"""
WebSocket 聊天处理 — 使用 pgvector 后端的 WebSocket 流式聊天

替代原始 deepwiki-open 的 websocket_wiki.py，底层使用 core/flows/ 中的
SimpleChatFlow 和 DeepResearchFlow 进行业务逻辑处理。

架构说明 (Phase 4 重构):
  - API 层只负责 WebSocket 协议处理（连接管理、纯文本分片推流）
  - 业务逻辑（RAG 检索、prompt 构建、QA 日志记录）委托给 core/flows/ 中的 Flow 类
  - Flow 类的完整方法 (chat() / research()) 负责完整的业务流程，包括 QA 日志记录
  - LLM 流式调用使用 core.utils.llm.call_llm_stream_raw()

协议:
  - 接收: JSON 格式请求
  - 发送: 纯文本分片（逐 token），兼容 deepwiki-open 前端
  - 完成: 发送 "[DONE]" 标记
  - 错误: 发送 "[ERROR: ...]" 标记
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from core.flows.chat_flow import SimpleChatFlow
from core.flows.research_flow import DeepResearchFlow
from core.utils.llm import call_llm_stream_raw
from core.utils.language import validate_language

logger = logging.getLogger(__name__)


# ============================================================
# WebSocket 处理函数
# ============================================================


async def handle_websocket_chat(websocket: WebSocket):
    """
    处理 WebSocket 聊天连接

    接收 JSON 格式的请求，流式返回 LLM 响应（纯文本分片）。
    支持普通聊天和深度研究两种模式。

    业务逻辑委托给:
      - SimpleChatFlow — 普通聊天模式
      - DeepResearchFlow — 深度研究模式
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        # 接收请求数据
        request_data = await websocket.receive_json()

        repo_url: str = request_data.get("repo_url", "")
        repo_type: str = request_data.get("type", "github")
        token = request_data.get("token")
        provider: str = request_data.get("provider", "dashscope")
        model = request_data.get("model")
        language: str = request_data.get("language", "en")
        query: str = request_data.get("query", "")
        deep_research: bool = request_data.get("deep_research", False)
        research_iterations: int = request_data.get("research_iterations", 5)

        # 语言验证
        language = validate_language(language, default="en")

        # 提取仓库名
        repo_url = repo_url.rstrip("/")
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url

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
):
    """
    处理普通 WebSocket 聊天 — 委托给 SimpleChatFlow.chat()

    SimpleChatFlow.chat() 负责完整的业务流程:
      1. 初始化 RAG 引擎（RAGEngine）
      2. 构建 RAG 上下文
      3. 构建 prompt（系统指令 + 上下文 + 用户问题）
      4. 调用 LLM 获取完整响应
      5. 记录问答日志到 qa_logs 表（通过 RAGEngine.log_qa()）

    API 层负责:
      - WebSocket 纯文本分片推流
      - 发送 [DONE] / [ERROR] 标记
    """
    try:
        # 初始化 SimpleChatFlow
        flow = SimpleChatFlow(
            repo_url=repo_url,
            provider=provider,
            model=model or "qwen-plus",
            language=language,
            use_database=True,
        )

        # 委托给 SimpleChatFlow.chat() 完整方法
        # chat() 内部执行: 初始化 RAG → 构建上下文 → 构建 prompt → 调用 LLM → 记录 QA 日志
        full_response = await flow.chat(query)

        # 将完整响应以纯文本分片推流
        # 按字符逐片发送，模拟流式输出
        chunk_size = 50  # 每片 50 字符
        for i in range(0, len(full_response), chunk_size):
            chunk = full_response[i:i + chunk_size]
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
    iterations: int = 5,
):
    """
    处理深度研究 WebSocket 聊天 — 委托给 DeepResearchFlow.research()

    DeepResearchFlow.research() 负责完整的业务流程:
      1. 初始化 RAG 引擎（RAGEngine）
      2. 迭代研究（最多 iterations 轮）
         - 构建 RAG 上下文
         - 构建研究 prompt（根据迭代次数选择模板）
         - 调用 LLM 获取响应
         - 检测是否完成
         - 提取研究阶段
      3. 记录问答日志到 qa_logs 表（通过 RAGEngine.log_qa()）

    API 层负责:
      - WebSocket 纯文本分片推流
      - 发送迭代标记（ITERATION_START / ITERATION_DONE / DONE / ERROR）
    """
    try:
        # 初始化 DeepResearchFlow，传入自定义迭代次数
        flow = DeepResearchFlow(
            repo_url=repo_url,
            provider=provider,
            model=model or "qwen-plus",
            language=language,
            use_database=True,
            max_iterations=iterations,
        )

        # 委托给 DeepResearchFlow.research() 完整方法
        # research() 内部执行: 初始化 RAG → 迭代研究 → 记录 QA 日志
        # 注意: research() 返回最终答案，但我们需要在迭代过程中发送标记
        # 因此这里我们仍然需要手动控制迭代流程以发送 WS 标记
        # 但业务逻辑（RAG、prompt 构建、完成检测）由 Flow 负责

        # 初始化 RAG 引擎
        flow._init_rag_engine()

        # 使用简单的内存对话跟踪
        conversation_turns: List[Dict[str, str]] = []

        for i in range(1, iterations + 1):
            # 构建 RAG 上下文
            context = flow._build_context(query)

            # 构建研究 prompt
            messages = flow._build_research_prompt(
                query=query,
                iteration=i,
                context=context,
            )

            # 如果有对话历史，按正确顺序添加到 messages 中
            for turn in conversation_turns:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
            # 添加继续研究的 user 消息
            messages.append({"role": "user", "content": "[DEEP RESEARCH] Continue the research"})

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

            # 检测是否完成（使用 Flow 的完成检测逻辑）
            if flow._check_if_complete(full_response):
                # 发送迭代完成信号
                await websocket.send_text(f"[ITERATION_DONE:{i}]")
                # 发送完成信号
                await websocket.send_text(f"[DONE:{i}]")
                return

            # 发送迭代完成信号
            await websocket.send_text(f"[ITERATION_DONE:{i}]")

        # 发送完成信号
        await websocket.send_text(f"[DONE:{iterations}]")

    except Exception as e:
        logger.error(f"Deep research error: {e}", exc_info=True)
        await websocket.send_text(f"[ERROR: {e}]")
