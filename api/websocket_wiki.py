"""
WebSocket 聊天处理 — 使用 pgvector 后端的 WebSocket 流式聊天

替代原始 deepwiki-open 的 websocket_wiki.py，底层使用 rag_optimizer 的
PgvectorRetriever 进行检索。
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from api.config import configs
from api.prompts import (
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT,
)
from api.rag import Memory
from rag_optimizer.integration.deepwiki_adapter import (
    PgvectorRetriever,
    PgvectorDatabaseManager,
)

logger = logging.getLogger(__name__)


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
# LLM 调用函数（WebSocket 版本）
# ============================================================


async def call_llm_stream_ws(
    websocket: WebSocket,
    provider: str,
    model: Optional[str],
    messages: List[Dict[str, str]],
):
    """
    调用 LLM 并通过 WebSocket 流式返回结果

    支持多个提供者：dashscope, google, openai, openrouter, ollama
    """
    try:
        from api.config import get_model_config

        model_config = get_model_config(provider=provider, model=model)
        model_kwargs = model_config.get("model_kwargs", {})
        actual_model = model_kwargs.get("model", model or "qwen-plus")

        if provider == "dashscope":
            await _call_dashscope_stream_ws(websocket, actual_model, messages, model_kwargs)
        elif provider == "google":
            await _call_google_stream_ws(websocket, actual_model, messages, model_kwargs)
        elif provider == "openai":
            await _call_openai_stream_ws(websocket, actual_model, messages, model_kwargs)
        elif provider == "openrouter":
            await _call_openrouter_stream_ws(websocket, actual_model, messages, model_kwargs)
        elif provider == "ollama":
            await _call_ollama_stream_ws(websocket, actual_model, messages, model_kwargs)
        else:
            await _call_dashscope_stream_ws(websocket, actual_model, messages, model_kwargs)

    except Exception as e:
        logger.error(f"LLM call error: {e}")
        await websocket.send_json({"error": str(e)})


async def _call_dashscope_stream_ws(
    websocket: WebSocket,
    model: str,
    messages: List[Dict[str, str]],
    model_kwargs: Dict[str, Any],
):
    """通过 WebSocket 调用 DashScope 流式 API"""
    from openai import AsyncOpenAI
    from rag_optimizer.config.settings import settings

    api_key = settings.embedding.dashscope_api_key
    if not api_key:
        await websocket.send_json({"error": "DASHSCOPE_API_KEY not configured"})
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=settings.embedding.dashscope_base_url,
    )

    temperature = model_kwargs.get("temperature", 0.7)
    top_p = model_kwargs.get("top_p", 0.8)

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            await websocket.send_json({"content": chunk.choices[0].delta.content})


async def _call_google_stream_ws(
    websocket: WebSocket,
    model: str,
    messages: List[Dict[str, str]],
    model_kwargs: Dict[str, Any],
):
    """通过 WebSocket 调用 Google Generative AI"""
    import os
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        await websocket.send_json({"error": "GOOGLE_API_KEY not configured"})
        return

    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)

    chat_messages = []
    for msg in messages:
        if msg["role"] != "system":
            chat_messages.append({"role": msg["role"], "parts": [msg["content"]]})

    chat = client.start_chat(history=chat_messages[:-1] if len(chat_messages) > 1 else [])
    response = await chat.send_message_async(
        chat_messages[-1]["parts"][0] if chat_messages else "",
    )
    await websocket.send_json({"content": response.text})


async def _call_openai_stream_ws(
    websocket: WebSocket,
    model: str,
    messages: List[Dict[str, str]],
    model_kwargs: Dict[str, Any],
):
    """通过 WebSocket 调用 OpenAI 流式 API"""
    from openai import AsyncOpenAI
    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        await websocket.send_json({"error": "OPENAI_API_KEY not configured"})
        return

    client = AsyncOpenAI(api_key=api_key)

    temperature = model_kwargs.get("temperature", 0.7)
    top_p = model_kwargs.get("top_p", 0.8)

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            await websocket.send_json({"content": chunk.choices[0].delta.content})


async def _call_openrouter_stream_ws(
    websocket: WebSocket,
    model: str,
    messages: List[Dict[str, str]],
    model_kwargs: Dict[str, Any],
):
    """通过 WebSocket 调用 OpenRouter 流式 API"""
    from openai import AsyncOpenAI
    import os

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        await websocket.send_json({"error": "OPENROUTER_API_KEY not configured"})
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    temperature = model_kwargs.get("temperature", 0.7)
    top_p = model_kwargs.get("top_p", 0.8)

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            await websocket.send_json({"content": chunk.choices[0].delta.content})


async def _call_ollama_stream_ws(
    websocket: WebSocket,
    model: str,
    messages: List[Dict[str, str]],
    model_kwargs: Dict[str, Any],
):
    """通过 WebSocket 调用 Ollama 流式 API"""
    import httpx
    import os

    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    options = model_kwargs.get("options", {})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": options.get("temperature", 0.7),
            "top_p": options.get("top_p", 0.8),
            "num_ctx": options.get("num_ctx", 32000),
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{ollama_base_url}/api/chat", json=payload) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            await websocket.send_json({"content": data["message"]["content"]})
                    except json.JSONDecodeError:
                        continue


# ============================================================
# WebSocket 处理函数
# ============================================================


async def handle_websocket_chat(websocket: WebSocket):
    """
    处理 WebSocket 聊天连接

    接收 JSON 格式的请求，流式返回 LLM 响应。
    支持普通聊天和深度研究两种模式。
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

        # 语言验证
        supported_langs = configs.get("lang_config", {}).get("supported_languages", {})
        if language not in supported_langs:
            language = "en"

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
            await websocket.send_json({"error": str(e)})
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
    """处理普通 WebSocket 聊天"""
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

        # 流式调用 LLM
        await call_llm_stream_ws(
            websocket=websocket,
            provider=provider,
            model=model,
            messages=messages,
        )

        # 发送完成信号
        await websocket.send_json({"done": True})

    except Exception as e:
        logger.error(f"Simple chat error: {e}", exc_info=True)
        await websocket.send_json({"error": str(e)})


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
    """处理深度研究 WebSocket 聊天"""
    try:
        memory = Memory()

        for i in range(1, iterations + 1):
            # 检索相关文档
            contexts = None
            if retriever:
                try:
                    contexts = retriever(query, k=10)
                except Exception as e:
                    logger.warning(f"Research iteration {i} retrieval error: {e}")

            # 构建深度研究提示词
            conversation_history = memory() if i > 1 else None
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
            await websocket.send_json({"iteration_start": i})

            # 流式调用 LLM
            full_response_chars = []
            original_send = websocket.send_json

            # 自定义发送以收集响应
            async def send_with_collection(data):
                if "content" in data:
                    full_response_chars.append(data["content"])
                await original_send(data)

            websocket.send_json = send_with_collection

            await call_llm_stream_ws(
                websocket=websocket,
                provider=provider,
                model=model,
                messages=messages,
            )

            # 恢复原始 send 方法
            websocket.send_json = original_send

            # 保存到记忆
            full_response = "".join(full_response_chars)
            memory.add_dialog_turn(query, full_response)

            # 发送迭代完成信号
            await websocket.send_json({"iteration_done": i})

        # 发送完成信号
        await websocket.send_json({"done": True, "iterations": iterations})

    except Exception as e:
        logger.error(f"Deep research error: {e}", exc_info=True)
        await websocket.send_json({"error": str(e)})
