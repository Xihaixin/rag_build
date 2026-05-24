"""
llm.py — LLM 客户端抽象
========================

统一封装多个 LLM 提供者的流式调用接口，支持：
  - dashscope (通义千问)
  - google (Gemini)
  - openai (GPT 系列)
  - openrouter (统一路由)
  - ollama (本地部署)

依赖:
  - core.config — get_model_config
  - rag_optimizer.config.settings — DashScope API 密钥
"""

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from core.config import get_model_config

logger = logging.getLogger("core.utils.llm")


# ══════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════


async def call_llm_stream(
    provider: str,
    model: Optional[str],
    messages: List[Dict[str, str]],
) -> AsyncGenerator[str, None]:
    """
    调用 LLM 并通过 SSE 流式返回结果。

    支持多个提供者：dashscope, google, openai, openrouter, ollama

    参数:
        provider: LLM 提供者名称
        model: 模型名称（可选，未指定时使用配置默认值）
        messages: 消息列表 [{"role": "user", "content": "..."}]

    生成:
        SSE 格式字符串: data: {"content":"文本块"}\\n\\n
    """
    try:
        model_config = get_model_config(provider=provider, model=model)
        model_kwargs = model_config.get("model_kwargs", {})
        actual_model = model_kwargs.get("model", model or "qwen-plus")

        if provider == "dashscope":
            async for chunk in _call_dashscope_stream(actual_model, messages, model_kwargs):
                yield chunk
        elif provider == "google":
            async for chunk in _call_google_stream(actual_model, messages, model_kwargs):
                yield chunk
        elif provider == "openai":
            async for chunk in _call_openai_stream(actual_model, messages, model_kwargs):
                yield chunk
        elif provider == "openrouter":
            async for chunk in _call_openrouter_stream(actual_model, messages, model_kwargs):
                yield chunk
        elif provider == "ollama":
            async for chunk in _call_ollama_stream(actual_model, messages, model_kwargs):
                yield chunk
        else:
            async for chunk in _call_dashscope_stream(actual_model, messages, model_kwargs):
                yield chunk

    except Exception as e:
        logger.error(f"LLM call error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ══════════════════════════════════════════════════════════════════════════
# Provider 实现
# ══════════════════════════════════════════════════════════════════════════


async def _call_dashscope_stream(
    model: str,
    messages: List[Dict[str, str]],
    model_kwargs: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """调用 DashScope (通义千问) 流式 API"""
    from openai import AsyncOpenAI
    from rag_optimizer.config.settings import settings

    api_key = settings.embedding.dashscope_api_key
    if not api_key:
        yield f"data: {json.dumps({'error': 'DASHSCOPE_API_KEY not configured'})}\n\n"
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
            yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"


async def _call_google_stream(
    model: str,
    messages: List[Dict[str, str]],
    model_kwargs: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """调用 Google Generative AI (Gemini)"""
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        yield f"data: {json.dumps({'error': 'GOOGLE_API_KEY not configured'})}\n\n"
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
    yield f"data: {json.dumps({'content': response.text})}\n\n"


async def _call_openai_stream(
    model: str,
    messages: List[Dict[str, str]],
    model_kwargs: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """调用 OpenAI 流式 API"""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        yield f"data: {json.dumps({'error': 'OPENAI_API_KEY not configured'})}\n\n"
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
            yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"


async def _call_openrouter_stream(
    model: str,
    messages: List[Dict[str, str]],
    model_kwargs: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """调用 OpenRouter 流式 API"""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        yield f"data: {json.dumps({'error': 'OPENROUTER_API_KEY not configured'})}\n\n"
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
            yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"


async def _call_ollama_stream(
    model: str,
    messages: List[Dict[str, str]],
    model_kwargs: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """调用 Ollama 流式 API"""
    import httpx

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
                            yield f"data: {json.dumps({'content': data['message']['content']})}\n\n"
                    except json.JSONDecodeError:
                        continue
