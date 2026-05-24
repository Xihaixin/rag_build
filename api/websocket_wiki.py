"""
WebSocket 聊天处理 — 使用 pgvector 后端的 WebSocket 流式聊天

替代原始 deepwiki-open 的 websocket_wiki.py，底层使用 rag_optimizer 的
PgvectorRetriever 进行检索。

协议变更 (Phase 3):
  - WebSocket 发送纯文本分片（兼容 deepwiki-open 前端协议）
  - 去除 {"content": "..."} JSON 包装
  - 去除 6 个 provider 专用 _call_*_stream_ws 函数
  - 统一使用 core.utils.llm.call_llm_stream_raw() 进行 provider 分发
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from core.prompts.rag import (

    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT,
)
from core.utils.llm import call_llm_stream_raw
from core.utils.language import get_language_name
from rag_optimizer.integration.deepwiki_adapter import (
    PgvectorRetriever,
    PgvectorDatabaseManager,
)

logger = logging.getLogger(__name__)


# ============================================================
# 辅助函数
# ============================================================


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
# WebSocket 处理函数
# ============================================================


async def handle_websocket_chat(websocket: WebSocket):
    """
    处理 WebSocket 聊天连接

    接收 JSON 格式的请求，流式返回 LLM 响应（纯文本分片）。
    支持普通聊天和深度研究两种模式。

    协议 (Phase 3):
      - 接收: JSON 格式请求
      - 发送: 纯文本分片（逐 token），兼容 deepwiki-open 前端
      - 完成: 发送 "[DONE]" 标记
      - 错误: 发送 "[ERROR: ...]" 标记
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        # 接收请求数据
        request_data = await websocket.receive_json()

        repo_url = request_data.get("repo_url", "")
        repo_type = request_data.get("type", "github")
        token = request_data.get("token")
        provider = request_data.get("provider", "dashscope")
        model = request_data.get("model")
        language = request_data.get("language", "en")
        query = request_data.get("query", "")
        file_path = request_data.get("filePath")
        deep_research = request_data.get("deep_research", False)
        research_iterations = request_data.get("research_iterations", 5)
        excluded_dirs = request_data.get("excluded_dirs")
        excluded_files = request_data.get("excluded_files")
        included_dirs = request_data.get("included_dirs")
        included_files = request_data.get("included_files")

        # 语言验证 — 使用 core.utils.language 验证
        from core.utils.language import validate_language
        language = validate_language(language, default="en")

        # 提取仓库名
        repo_url = repo_url.rstrip("/")
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url

        # 准备检索器
        try:
            db_manager = PgvectorDatabaseManager()

            project_id = db_manager.prepare_database(
                repo_url_or_path=repo_url,
                repo_type=repo_type,
                access_token=token,
            )

            retriever = PgvectorRetriever(
                project_id=project_id,
                retrieval_type="hybrid",
                top_k=10,
            )
        except Exception as e:
            logger.warning(f"Could not prepare retriever: {e}")
            retriever = None

        # 处理深度研究模式
        if deep_research:
            await _handle_deep_research_ws(
                websocket=websocket,
                query=query,
                repo_url=repo_url,
                repo_name=repo_name,
                repo_type=repo_type,
                language=language,
                provider=provider,
                model=model,
                retriever=retriever,
                iterations=research_iterations,
            )
        else:
            # 普通聊天模式
            await _handle_simple_chat_ws(
                websocket=websocket,
                query=query,
                repo_url=repo_url,
                repo_name=repo_name,
                repo_type=repo_type,
                language=language,
                provider=provider,
                model=model,
                retriever=retriever,
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}", exc_info=True)
        try:
            await websocket.send_text(f"[ERROR: {e}]")
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def _handle_simple_chat_ws(
    websocket: WebSocket,
    query: str,
    repo_url: str,
    repo_name: str,
    repo_type: str,
    language: str,
    provider: str,
    model: Optional[str],
    retriever: Optional[PgvectorRetriever],
):
    """处理普通 WebSocket 聊天 — 纯文本分片推流"""
    try:
        # 检索相关文档
        contexts = None
        if retriever:
            try:
                contexts = retriever(query, k=10)
                logger.info(f"Retrieved {len(contexts)} documents")
            except Exception as e:
                logger.warning(f"Retrieval error: {e}")

        # 构建提示词
        prompt = build_simple_chat_prompt(
            query=query,
            repo_url=repo_url,
            repo_name=repo_name,
            repo_type=repo_type,
            language=language,
            contexts=contexts,
        )

        # 构建消息
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]

        # 流式调用 LLM — 纯文本分片
        async for chunk in call_llm_stream_raw(
            provider=provider,
            model=model,
            messages=messages,
        ):
            await websocket.send_text(chunk)

        # 发送完成信号
        await websocket.send_text("[DONE]")

    except Exception as e:
        logger.error(f"Simple chat error: {e}", exc_info=True)
        await websocket.send_text(f"[ERROR: {e}]")


async def _handle_deep_research_ws(
    websocket: WebSocket,
    query: str,
    repo_url: str,
    repo_name: str,
    repo_type: str,
    language: str,
    provider: str,
    model: Optional[str],
    retriever: Optional[PgvectorRetriever],
    iterations: int = 5,
):
    """处理深度研究 WebSocket 聊天 — 纯文本分片推流"""
    try:
        # 使用简单的内存对话跟踪（替代 api.rag.Memory）
        conversation_turns: List[Dict[str, str]] = []

        for i in range(1, iterations + 1):
            # 检索相关文档
            contexts = None
            if retriever:
                try:
                    contexts = retriever(query, k=10)
                except Exception as e:
                    logger.warning(f"Research iteration {i} retrieval error: {e}")

            # 构建深度研究提示词
            conversation_history = None
            if conversation_turns:
                conversation_history = {
                    str(idx): {
                        "user_query": {"data": turn["user"]},
                        "assistant_response": {"data": turn["assistant"]},
                    }
                    for idx, turn in enumerate(conversation_turns)
                }

            prompt = build_deep_research_prompt(
                query=query,
                repo_url=repo_url,
                repo_name=repo_name,
                repo_type=repo_type,
                language=language,
                iteration=i,
                total_iterations=iterations,
                conversation_history=conversation_history,
            )

            # 构建消息
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ]

            # 添加上下文
            if contexts:
                context_text = build_context_from_results(contexts)
                messages.insert(1, {"role": "user", "content": f"Context:\n{context_text}"})

            # 发送迭代开始信号
            await websocket.send_text(f"[ITERATION_START:{i}]")

            # 流式调用 LLM — 纯文本分片，同时收集完整响应
            full_response_chars: List[str] = []
            async for chunk in call_llm_stream_raw(
                provider=provider,
                model=model,
                messages=messages,
            ):
                full_response_chars.append(chunk)
                await websocket.send_text(chunk)

            # 保存到对话历史
            full_response = "".join(full_response_chars)
            conversation_turns.append({"user": query, "assistant": full_response})

            # 发送迭代完成信号
            await websocket.send_text(f"[ITERATION_DONE:{i}]")

        # 发送完成信号
        await websocket.send_text(f"[DONE:{iterations}]")

    except Exception as e:
        logger.error(f"Deep research error: {e}", exc_info=True)
        await websocket.send_text(f"[ERROR: {e}]")
