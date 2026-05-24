"""
sse.py — SSE 解析与聚合工具
=============================

提供 Server-Sent Events (SSE) 格式的解析和聚合功能，
用于处理 LLM 流式响应。

依赖:
  - core.utils.llm — call_llm_stream
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from core.utils.llm import call_llm_stream

logger = logging.getLogger("core.utils.sse")


def parse_sse_chunk(chunk: str) -> Optional[str]:
    """
    解析 call_llm_stream 返回的 SSE 格式字符串，提取实际文本内容。

    call_llm_stream 返回的格式为: data: {"content":"文本块"}\\n\\n
    或错误时: data: {"error":"错误信息"}\\n\\n

    参数:
        chunk: SSE 格式的字符串块

    返回:
        提取的文本内容，如果是错误块则返回 None
    """
    if not chunk or not chunk.strip():
        return None

    text = chunk.strip()
    if text.startswith("data: "):
        text = text[6:]

    try:
        data = json.loads(text)
        if "error" in data:
            logger.warning(f"LLM 返回错误: {data['error']}")
            return None
        return data.get("content", "")
    except json.JSONDecodeError:
        return chunk


async def call_llm_and_collect(
    provider: str,
    model: Optional[str],
    messages: List[Dict[str, str]],
) -> str:
    """
    调用 call_llm_stream 并自动解析 SSE 格式，返回完整的纯文本响应。

    这是 call_llm_stream 的便捷封装，自动处理 SSE 解析，
    避免在每个调用点重复编写 SSE 解析逻辑。

    参数:
        provider: LLM 提供者 (dashscope, google, openai, openrouter, ollama)
        model: 模型名称
        messages: 消息列表

    返回:
        完整的纯文本响应（不含 SSE 格式标记）
    """
    full_response = ""
    try:
        async for chunk in call_llm_stream(provider, model, messages):
            if chunk:
                text = parse_sse_chunk(chunk)
                if text:
                    full_response += text
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        raise

    return full_response
