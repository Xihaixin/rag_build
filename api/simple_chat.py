"""
流式聊天端点 — 使用 pgvector 后端的 HTTP SSE 流式聊天

替代原始 deepwiki-open 的 simple_chat.py，底层使用 rag_optimizer 的
PgvectorRetriever 进行检索，支持多种 LLM 提供者。

注意: call_llm_stream 及所有 _call_*_stream 函数已迁移至
core.utils.llm，此处仅做重导出以保持向后兼容。
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.config import configs
from api.prompts import (
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT,
)
from api.rag import Memory
from core.utils.llm import (
    call_llm_stream,
    _call_dashscope_stream,
    _call_google_stream,
    _call_openai_stream,
    _call_openrouter_stream,
    _call_ollama_stream,
)
from rag_optimizer.integration.deepwiki_adapter import (
    PgvectorRetriever,
    PgvectorDatabaseManager,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# 请求模型
# ============================================================


class ChatCompletionRequest(BaseModel):
    """聊天补全请求"""
    messages: List[Dict[str, str]] = Field(..., description="聊天消息列表")
    provider: str = Field(default="dashscope", description="LLM 提供者")
    model: Optional[str] = Field(default=None, description="模型名称")
    temperature: Optional[float] = Field(default=None, description="温度参数")
    top_p: Optional[float] = Field(default=None, description="Top-p 采样参数")
    stream: bool = Field(default=True, description="是否流式返回")
    repo_url: Optional[str] = Field(default=None, description="仓库 URL")
    repo_type: Optional[str] = Field(default="github", description="仓库类型")
    token: Optional[str] = Field(default=None, description="访问令牌")
    language: Optional[str] = Field(default="en", description="语言代码")
    file_path: Optional[str] = Field(default=None, description="文件路径")
    deep_research: bool = Field(default=False, description="是否使用深度研究模式")
    research_iterations: int = Field(default=5, description="深度研究迭代次数")
    excluded_dirs: Optional[List[str]] = Field(default=None, description="排除的目录")
    excluded_files: Optional[List[str]] = Field(default=None, description="排除的文件")
    included_dirs: Optional[List[str]] = Field(default=None, description="包含的目录")
    included_files: Optional[List[str]] = Field(default=None, description="包含的文件")


# ============================================================
# 辅助函数
# ============================================================


def get_language_name(language_code: str) -> str:
    """获取语言名称"""
    lang_config = configs.get("lang_config", {})
    supported = lang_config.get("supported_languages", {})
    return supported.get(language_code, "English")


def build_context_from_results(results: List[Any]) -> str:
    """从检索结果构建上下文文本"""
    context_parts = []
    for i, doc in enumerate(results):
        file_path = (
            getattr(doc, "meta", {}).get("file_path", "unknown")
            if hasattr(doc, "meta")
            else "unknown"
        )
        text = getattr(doc, "text", "") if hasattr(doc, "text") else ""
        context_parts.append(f"{i + 1}.\nFile Path: {file_path}\nContent: {text}")
    return "\n".join(context_parts)


def build_simple_chat_prompt(
    query: str,
    repo_url: str,
    repo_name: str,
    repo_type: str,
    language: str,
    contexts: Optional[List[Any]] = None,
    conversation_history: Optional[Dict] = None,
) -> str:
    """构建简单聊天提示词"""
    language_name = get_language_name(language)

    system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT.format(
        repo_type=repo_type,
        repo_url=repo_url,
        repo_name=repo_name,
        language_name=language_name,
    )

    prompt_parts = [f"<system>{system_prompt}</system>"]

    if conversation_history:
        prompt_parts.append("<conversation_history>")
        for key, turn in conversation_history.items():
            user_query = turn.get("user_query", {})
            assistant_response = turn.get("assistant_response", {})
            if isinstance(user_query, dict):
                prompt_parts.append(f"User: {user_query.get('data', str(user_query))}")
            else:
                prompt_parts.append(f"User: {str(user_query)}")
            if isinstance(assistant_response, dict):
                prompt_parts.append(
                    f"Assistant: {assistant_response.get('data', str(assistant_response))}"
                )
            else:
                prompt_parts.append(f"Assistant: {str(assistant_response)}")
        prompt_parts.append("</conversation_history>")

    if contexts:
        prompt_parts.append("<context>")
        prompt_parts.append(build_context_from_results(contexts))
        prompt_parts.append("</context>")

    prompt_parts.append(f"<user_query>{query}</user_query>")

    return "\n".join(prompt_parts)


def build_deep_research_prompt(
    query: str,
    repo_url: str,
    repo_name: str,
    repo_type: str,
    language: str,
    iteration: int,
    total_iterations: int,
    conversation_history: Optional[Dict] = None,
) -> str:
    """构建深度研究提示词"""
    language_name = get_language_name(language)

    if iteration == 1:
        prompt_template = DEEP_RESEARCH_FIRST_ITERATION_PROMPT
    elif iteration >= total_iterations:
        prompt_template = DEEP_RESEARCH_FINAL_ITERATION_PROMPT
    else:
        prompt_template = DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT

    system_prompt = prompt_template.format(
        repo_type=repo_type,
        repo_url=repo_url,
        repo_name=repo_name,
        language_name=language_name,
        research_iteration=iteration,
    )

    prompt_parts = [f"<system>{system_prompt}</system>"]

    if conversation_history:
        prompt_parts.append("<conversation_history>")
        for key, turn in conversation_history.items():
            user_query = turn.get("user_query", {})
            assistant_response = turn.get("assistant_response", {})
            if isinstance(user_query, dict):
                prompt_parts.append(f"User: {user_query.get('data', str(user_query))}")
            else:
                prompt_parts.append(f"User: {str(user_query)}")
            if isinstance(assistant_response, dict):
                prompt_parts.append(
                    f"Assistant: {assistant_response.get('data', str(assistant_response))}"
                )
            else:
                prompt_parts.append(f"Assistant: {str(assistant_response)}")
        prompt_parts.append("</conversation_history>")

    prompt_parts.append(f"<user_query>{query}</user_query>")

    return "\n".join(prompt_parts)


# ============================================================
# 聊天补全端点（SSE 流式）
# ============================================================


@router.post("/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """
    流式聊天补全端点

    支持多种 LLM 提供者，通过 SSE (Server-Sent Events) 流式返回结果。
    支持普通聊天和深度研究两种模式。
    """
    try:
        query = request.messages[-1]["content"] if request.messages else ""
        logger.info(
            f"Chat completion request: provider={request.provider}, "
            f"model={request.model or 'default'}, "
            f"deep_research={request.deep_research}, "
            f"query='{query[:50]}...'"
        )

        # 如果没有仓库 URL，直接调用 LLM
        if not request.repo_url:
            async def direct_stream():
                async for chunk in call_llm_stream(
                    provider=request.provider,
                    model=request.model,
                    messages=request.messages,
                ):
                    yield chunk

            return StreamingResponse(
                direct_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # 有仓库 URL，使用 RAG 检索
        repo_name = request.repo_url.rstrip("/").split("/")[-1]

        if request.deep_research:
            return await _handle_deep_research(request, query, repo_name)
        else:
            return await _handle_simple_chat(request, query, repo_name)

    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _handle_simple_chat(
    request: ChatCompletionRequest,
    query: str,
    repo_name: str,
) -> StreamingResponse:
    """处理普通聊天模式"""
    memory = Memory()

    async def response_stream():
        try:
            # 初始化检索器
            retriever = None
            try:
                db_manager = PgvectorDatabaseManager()
                project_id = db_manager.prepare_database(
                    repo_url_or_path=request.repo_url,
                    repo_type=request.repo_type or "github",
                    access_token=request.token,
                )
                retriever = PgvectorRetriever(
                    project_id=project_id,
                    retrieval_type="hybrid",
                    top_k=10,
                )
            except Exception as e:
                logger.warning(f"Could not initialize retriever: {e}")

            # 执行检索
            contexts = None
            if retriever:
                try:
                    results, _ = retriever(query, k=10) if hasattr(retriever, '__call__') else ([], {})
                    if results:
                        contexts = results
                except Exception as e:
                    logger.warning(f"Retrieval error: {e}")

            # 构建提示词
            prompt = build_simple_chat_prompt(
                query=query,
                repo_url=request.repo_url,
                repo_name=repo_name,
                repo_type=request.repo_type or "github",
                language=request.language or "en",
                contexts=contexts,
                conversation_history=None,
            )

            # 调用 LLM
            messages = [{"role": "user", "content": prompt}]
            async for chunk in call_llm_stream(
                provider=request.provider,
                model=request.model,
                messages=messages,
            ):
                yield chunk

            # 记录对话
            memory.add_dialog_turn(query, "[streaming response]")

        except Exception as e:
            logger.error(f"Simple chat stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        response_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _handle_deep_research(
    request: ChatCompletionRequest,
    query: str,
    repo_name: str,
) -> StreamingResponse:
    """处理深度研究模式"""
    memory = Memory()
    total_iterations = request.research_iterations

    async def research_stream():
        try:
            # 初始化检索器
            retriever = None
            try:
                db_manager = PgvectorDatabaseManager()
                project_id = db_manager.prepare_database(
                    repo_url_or_path=request.repo_url,
                    repo_type=request.repo_type or "github",
                    access_token=request.token,
                )
                retriever = PgvectorRetriever(
                    project_id=project_id,
                    retrieval_type="hybrid",
                    top_k=10,
                )
            except Exception as e:
                logger.warning(f"Could not initialize retriever: {e}")

            for iteration in range(1, total_iterations + 1):
                # 执行检索
                contexts = None
                if retriever:
                    try:
                        results, _ = retriever(query, k=10) if hasattr(retriever, '__call__') else ([], {})
                        if results:
                            contexts = results
                    except Exception as e:
                        logger.warning(f"Retrieval error at iteration {iteration}: {e}")

                # 构建深度研究提示词
                prompt = build_deep_research_prompt(
                    query=query,
                    repo_url=request.repo_url,
                    repo_name=repo_name,
                    repo_type=request.repo_type or "github",
                    language=request.language or "en",
                    iteration=iteration,
                    total_iterations=total_iterations,
                    conversation_history=memory() if hasattr(memory, '__call__') else None,
                )

                # 发送迭代标记
                yield f"data: {json.dumps({'type': 'iteration', 'iteration': iteration, 'total': total_iterations})}\n\n"

                # 调用 LLM
                messages = [{"role": "user", "content": prompt}]
                full_response = ""
                async for chunk in call_llm_stream(
                    provider=request.provider,
                    model=request.model,
                    messages=messages,
                ):
                    full_response += chunk
                    yield chunk

                # 记录对话
                memory.add_dialog_turn(
                    f"{query} (iteration {iteration}/{total_iterations})",
                    full_response,
                )

            # 发送完成标记
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Deep research stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        research_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
