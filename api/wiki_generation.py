"""
Wiki 生成 API 端点 — 使用 WikiGenerationFlow 的 SSE 流式 Wiki 生成

将后端 core/flows/wiki_flow.py 中的 WikiGenerationFlow 集成到 API 层，
提供 SSE 流式端点供前端调用。

架构说明:
  - API 层只负责 HTTP 协议处理（请求解析、SSE 流式响应）
  - 业务逻辑（文件树获取、Wiki 结构确定、页面生成、数据库保存）
    委托给 core/flows/wiki_flow.py 中的 WikiGenerationFlow
  - WikiGenerationFlow.run() 负责完整的业务流程
  - SSE 事件类型:
    - structure: Wiki 结构确定完成
    - page_progress: 页面生成进度
    - page_complete: 单个页面生成完成
    - complete: 全部完成
    - error: 错误
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.flows.wiki_flow import WikiGenerationFlow
from core.models import WikiPage, WikiStructure

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# 请求/响应模型
# ============================================================


class WikiGenerateRequest(BaseModel):
    """Wiki 生成请求"""
    repo_url: str = Field(..., description="仓库 URL")
    repo_type: str = Field(default="github", description="仓库类型 (github/gitlab/bitbucket/local)")
    provider: str = Field(default="dashscope", description="LLM 提供者")
    model: Optional[str] = Field(default=None, description="模型名称")
    language: str = Field(default="zh", description="语言代码")
    comprehensive: bool = Field(default=True, description="是否生成综合 Wiki（更多页面）")
    local_path: Optional[str] = Field(default=None, description="本地仓库路径（repo_type=local 时使用）")


class WikiPageGenerateRequest(BaseModel):
    """单页 Wiki 生成请求"""
    repo_url: str = Field(..., description="仓库 URL")
    repo_type: str = Field(default="github", description="仓库类型")
    provider: str = Field(default="dashscope", description="LLM 提供者")
    model: Optional[str] = Field(default=None, description="模型名称")
    language: str = Field(default="zh", description="语言代码")
    page: WikiPage = Field(..., description="要生成的页面信息")
    file_tree: Optional[str] = Field(default=None, description="文件树（用于上下文）")
    readme: Optional[str] = Field(default=None, description="README 内容（用于上下文）")


# ============================================================
# Wiki 生成端点
# ============================================================


@router.post("/wiki/generate")
async def generate_wiki(request: WikiGenerateRequest):
    """
    生成完整的 Wiki 文档

    使用 WikiGenerationFlow 执行完整的 Wiki 生成流程：
      1. 获取仓库文件树和 README
      2. 调用 LLM 确定 Wiki 结构
      3. 逐页生成 Wiki 页面内容
      4. 保存到 PostgreSQL 数据库

    通过 SSE 流式返回生成进度。
    """
    try:
        logger.info(
            f"Wiki generation request: repo={request.repo_url}, "
            f"provider={request.provider}, model={request.model or 'default'}, "
            f"language={request.language}, comprehensive={request.comprehensive}"
        )

        # 初始化 WikiGenerationFlow
        flow = WikiGenerationFlow(
            repo_url=request.repo_url,
            provider=request.provider,
            model=request.model or "qwen-plus",
            language=request.language,
            comprehensive=request.comprehensive,
            use_database=True,
            local_path=request.local_path,
        )

        async def generate_stream():
            try:
                # 步骤 1: 获取仓库结构
                yield _sse_event("progress", {
                    "step": "fetch_structure",
                    "message": "正在获取仓库文件结构...",
                })

                file_tree, readme = flow.fetch_repository_structure()

                yield _sse_event("progress", {
                    "step": "fetch_structure_done",
                    "message": f"文件树获取完成 ({len(file_tree)} 字符)",
                    "file_tree_length": len(file_tree),
                    "readme_length": len(readme),
                })

                # 步骤 2: 确定 Wiki 结构
                yield _sse_event("progress", {
                    "step": "determine_structure",
                    "message": "正在调用 LLM 确定 Wiki 结构...",
                })

                wiki_structure = await flow.determine_wiki_structure()

                # 发送结构信息
                pages_info = []
                for p in wiki_structure.pages:
                    pages_info.append({
                        "id": p.id,
                        "title": p.title,
                        "importance": p.importance,
                        "filePaths": p.filePaths,
                        "relatedPages": p.relatedPages,
                    })

                yield _sse_event("structure", {
                    "id": wiki_structure.id,
                    "title": wiki_structure.title,
                    "description": wiki_structure.description,
                    "pages": pages_info,
                    "total_pages": len(wiki_structure.pages),
                })

                # 步骤 3: 逐页生成
                yield _sse_event("progress", {
                    "step": "generating_pages",
                    "message": f"开始生成 {len(wiki_structure.pages)} 个页面...",
                    "total_pages": len(wiki_structure.pages),
                })

                total_pages = len(wiki_structure.pages)
                for idx, page in enumerate(wiki_structure.pages):
                    yield _sse_event("page_progress", {
                        "page_id": page.id,
                        "page_title": page.title,
                        "index": idx + 1,
                        "total": total_pages,
                        "progress": int((idx / total_pages) * 100),
                        "message": f"正在生成页面 [{idx + 1}/{total_pages}]: {page.title}",
                    })

                    # 生成单个页面
                    content = await flow._generate_single_page(page)
                    flow.generated_pages[page.id] = content
                    page.content = content

                    yield _sse_event("page_complete", {
                        "page_id": page.id,
                        "page_title": page.title,
                        "content": content,
                        "index": idx + 1,
                        "total": total_pages,
                        "content_length": len(content),
                    })

                # 步骤 4: 保存到数据库
                yield _sse_event("progress", {
                    "step": "saving_to_database",
                    "message": "正在保存到数据库...",
                })

                saved_count = flow._save_to_database()

                # 完成
                yield _sse_event("complete", {
                    "message": "Wiki 生成完成",
                    "total_pages": total_pages,
                    "saved_count": saved_count,
                    "wiki_structure_id": wiki_structure.id,
                    "wiki_structure_title": wiki_structure.title,
                })

            except Exception as e:
                logger.error(f"Wiki generation error: {e}", exc_info=True)
                yield _sse_event("error", {"message": str(e)})

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"Wiki generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/wiki/generate/page")
async def generate_wiki_page(request: WikiPageGenerateRequest):
    """
    生成单个 Wiki 页面

    用于重新生成单个页面，不需要执行完整的 Wiki 生成流程。
    """
    try:
        logger.info(
            f"Single page generation: repo={request.repo_url}, "
            f"page={request.page.id} ({request.page.title})"
        )

        # 初始化 WikiGenerationFlow
        flow = WikiGenerationFlow(
            repo_url=request.repo_url,
            provider=request.provider,
            model=request.model or "qwen-plus",
            language=request.language,
            comprehensive=True,
            use_database=True,
        )

        # 设置文件树和 README（如果提供）
        if request.file_tree:
            flow.file_tree = request.file_tree
        if request.readme:
            flow.readme = request.readme

        async def page_stream():
            try:
                yield _sse_event("progress", {
                    "step": "generating",
                    "message": f"正在生成页面: {request.page.title}",
                })

                content = await flow._generate_single_page(request.page)

                yield _sse_event("page_complete", {
                    "page_id": request.page.id,
                    "page_title": request.page.title,
                    "content": content,
                    "content_length": len(content),
                })

                yield _sse_event("complete", {
                    "message": "页面生成完成",
                    "page_id": request.page.id,
                })

            except Exception as e:
                logger.error(f"Page generation error: {e}", exc_info=True)
                yield _sse_event("error", {"message": str(e)})

        return StreamingResponse(
            page_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"Page generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 辅助函数
# ============================================================


def _sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """构建 SSE 事件字符串"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
