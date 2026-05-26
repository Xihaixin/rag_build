"""
流式聊天端点 — 使用 pgvector 后端的 HTTP SSE 流式聊天

替代原始 deepwiki-open 的 simple_chat.py，底层使用 core/flows/ 中的
SimpleChatFlow 和 DeepResearchFlow 进行业务逻辑处理。

架构说明 (Phase 4 重构):
  - API 层只负责 HTTP 协议处理（请求解析、SSE 流式响应）
  - 业务逻辑（RAG 检索、prompt 构建、QA 日志记录）委托给 core/flows/ 中的 Flow 类
  - Flow 类的完整方法 (chat() / research()) 负责完整的业务流程，包括 QA 日志记录
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

    架构说明:
      Flow 类的完整方法 (chat() / research()) 负责完整的业务流程，
      包括 RAG 检索、prompt 构建、LLM 调用、QA 日志记录。
      API 层将 Flow 返回的完整结果以 SSE 格式流式返回给客户端。
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
    处理普通聊天模式 — 委托给 SimpleChatFlow.chat()

    SimpleChatFlow.chat() 负责完整的业务流程:
      1. 初始化 RAG 引擎（RAGEngine）
      2. 构建 RAG 上下文
      3. 构建 prompt（系统指令 + 上下文 + 用户问题）
      4. 调用 LLM 获取完整响应
      5. 记录问答日志到 qa_logs 表（通过 RAGEngine.log_qa()）

    API 层负责:
      - 将 Flow 返回的完整结果以 SSE 格式流式返回
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
            # 委托给 SimpleChatFlow.chat() 完整方法
            # chat() 内部执行: 初始化 RAG → 构建上下文 → 构建 prompt → 调用 LLM → 记录 QA 日志
            full_response = await flow.chat(query)

            # 将完整响应以 SSE 格式流式返回
            # 模拟流式输出，每个句子/段落作为一个 chunk
            # 这样前端可以逐步显示内容
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            # 按行分割，逐行流式输出
            lines = full_response.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    chunk = line + ('\n' if i < len(lines) - 1 else '')
                    yield f"data: {json.dumps({'content': chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

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
    处理深度研究模式 — 委托给 DeepResearchFlow.research()

    DeepResearchFlow.research() 负责完整的业务流程:
      1. 初始化 RAG 引擎（RAGEngine）
      2. 迭代研究（最多 5 轮）
         - 构建 RAG 上下文
         - 构建研究 prompt（根据迭代次数选择模板）
         - 调用 LLM 获取响应
         - 检测是否完成
         - 提取研究阶段
      3. 记录问答日志到 qa_logs 表（通过 RAGEngine.log_qa()）

    API 层负责:
      - 将 Flow 返回的完整结果以 SSE 格式流式返回
      - 发送迭代标记（iteration/done）
    """
    # 初始化 DeepResearchFlow（repo_url 在此路径下保证不为 None）
    repo_url: str = request.repo_url  # type: ignore[assignment]
    total_iterations = request.research_iterations
    flow = DeepResearchFlow(
        repo_url=repo_url,
        provider=request.provider,
        model=request.model or "qwen-plus",
        language=request.language or "en",
        use_database=True,
        max_iterations=total_iterations,
    )

    async def research_stream():
        try:
            # 委托给 DeepResearchFlow.research() 完整方法
            # research() 内部执行: 初始化 RAG → 迭代研究 → 记录 QA 日志
            final_answer = await flow.research(query)

            # 发送研究阶段信息
            stages_data = []
            for stage in flow.research_stages:
                stages_data.append({
                    "title": stage.title,
                    "content": stage.content,
                    "iteration": stage.iteration,
                    "type": stage.type,
                })

            yield f"data: {json.dumps({'type': 'stages', 'stages': stages_data})}\n\n"

            # 将最终答案以 SSE 格式流式返回
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            lines = final_answer.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    chunk = line + ('\n' if i < len(lines) - 1 else '')
                    yield f"data: {json.dumps({'content': chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'iterations': flow.current_iteration})}\n\n"

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
