"""
流式聊天端点 — 使用 pgvector 后端的 HTTP SSE 流式聊天

替代原始 deepwiki-open 的 simple_chat.py，底层使用 core/flows/ 中的
SimpleChatFlow 和 DeepResearchFlow 进行业务逻辑处理。

架构说明 (Phase 4 重构):
  - API 层只负责 HTTP 协议处理（请求解析、SSE 流式响应）
  - 业务逻辑（RAG 检索、prompt 构建）委托给 core/flows/ 中的 Flow 类
  - LLM 流式调用使用 core.utils.llm.call_llm_stream()
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.flows.chat_flow import SimpleChatFlow
from core.flows.research_flow import DeepResearchFlow
from core.utils.llm import call_llm_stream

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
# 聊天补全端点（SSE 流式）
# ============================================================


@router.post("/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """
    流式聊天补全端点

    支持多种 LLM 提供者，通过 SSE (Server-Sent Events) 流式返回结果。
    支持普通聊天和深度研究两种模式。

    业务逻辑委托给:
      - SimpleChatFlow — 普通聊天模式
      - DeepResearchFlow — 深度研究模式
    """
    try:
        query = request.messages[-1]["content"] if request.messages else ""
        logger.info(
            f"Chat completion request: provider={request.provider}, "
            f"model={request.model or 'default'}, "
            f"deep_research={request.deep_research}, "
            f"query='{query[:50]}...'"
        )

        # 如果没有仓库 URL，直接调用 LLM（无需 RAG）
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
    """
    处理普通聊天模式 — 委托给 SimpleChatFlow

    SimpleChatFlow 负责:
      1. 初始化 RAG 引擎（RAGEngine）
      2. 构建 RAG 上下文
      3. 构建 prompt（系统指令 + 上下文 + 用户问题）

    API 层负责:
      - 流式调用 LLM 并返回 SSE 响应
    """
    # 初始化 SimpleChatFlow（repo_url 在此路径下保证不为 None）
    repo_url: str = request.repo_url  # type: ignore[assignment]
    flow = SimpleChatFlow(
        repo_url=repo_url,
        provider=request.provider,
        model=request.model or "qwen-plus",
        language=request.language or "en",
        use_database=True,
    )


    async def response_stream():
        try:
            # 步骤 1: 初始化 RAG 引擎
            flow._init_rag_engine()

            # 步骤 2: 构建 RAG 上下文
            context = flow._build_context(query)

            # 步骤 3: 构建 prompt
            messages = flow._build_prompt(query, context)

            # 步骤 4: 流式调用 LLM
            async for chunk in call_llm_stream(
                provider=request.provider,
                model=request.model,
                messages=messages,
            ):
                yield chunk

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
    """
    处理深度研究模式 — 委托给 DeepResearchFlow

    DeepResearchFlow 负责:
      1. 初始化 RAG 引擎（RAGEngine）
      2. 构建 RAG 上下文
      3. 构建研究 prompt（根据迭代次数选择模板）

    API 层负责:
      - 迭代循环控制
      - 流式调用 LLM 并返回 SSE 响应
      - 发送迭代标记（iteration/done）
    """
    # 初始化 DeepResearchFlow（repo_url 在此路径下保证不为 None）
    repo_url: str = request.repo_url  # type: ignore[assignment]
    flow = DeepResearchFlow(
        repo_url=repo_url,
        provider=request.provider,
        model=request.model or "qwen-plus",
        language=request.language or "en",
        use_database=True,
    )

    total_iterations = request.research_iterations
    conversation_turns: List[Dict[str, str]] = []

    async def research_stream():
        try:
            # 初始化 RAG 引擎
            flow._init_rag_engine()

            for iteration in range(1, total_iterations + 1):
                # 构建 RAG 上下文
                context = flow._build_context(query)

                # 构建研究 prompt
                messages = flow._build_research_prompt(
                    query=query,
                    iteration=iteration,
                    context=context,
                )

                # 如果有对话历史，添加到 messages 中
                for turn in conversation_turns:
                    messages.append({"role": "assistant", "content": turn["assistant"]})
                    messages.append({"role": "user", "content": "[DEEP RESEARCH] Continue the research"})

                # 发送迭代标记
                yield f"data: {json.dumps({'type': 'iteration', 'iteration': iteration, 'total': total_iterations})}\n\n"

                # 流式调用 LLM
                full_response = ""
                async for chunk in call_llm_stream(
                    provider=request.provider,
                    model=request.model,
                    messages=messages,
                ):
                    full_response += chunk
                    yield chunk

                # 记录对话
                conversation_turns.append({
                    "user": f"{query} (iteration {iteration}/{total_iterations})",
                    "assistant": full_response,
                })

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
