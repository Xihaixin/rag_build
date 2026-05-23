"""
OpenWiki-Study API 端点

使用 PostgreSQL + pgvector 后端替代原始 LocalDB + .pkl 存储。
保持与原始 deepwiki-open 前端兼容的 API 接口。
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from api.config import configs, WIKI_AUTH_MODE, WIKI_AUTH_CODE
from rag_optimizer.db.repository import ProjectRepository

logger = logging.getLogger(__name__)


# ============================================================
# FastAPI 应用初始化
# ============================================================

app = FastAPI(
    title="OpenWiki API (RAG Optimized)",
    description="基于 PostgreSQL + pgvector 的 deepwiki-open 重构版 API",
    version="2.0.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 辅助函数
# ============================================================


def get_adalflow_default_root_path() -> str:
    """获取 Adalflow 默认根路径"""
    project_path = r"D:\ProgramFile2_OR\Python_Study_System\OpenStudy\DATA_SOURCE"
    return os.path.expanduser(os.path.join(project_path, ".adalflow"))


WIKI_CACHE_DIR = os.path.join(get_adalflow_default_root_path(), "wikicache")
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)


# ============================================================
# Pydantic 模型
# ============================================================


class WikiPage(BaseModel):
    """Wiki 页面模型"""
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str  # high, medium, low
    relatedPages: List[str] = []  # 相关页面 ID 列表


class ProcessedProjectEntry(BaseModel):
    """已处理项目条目"""
    id: str
    owner: str
    repo: str
    name: str
    repo_type: str
    submittedAt: int
    language: str


class RepoInfo(BaseModel):
    """仓库信息"""
    owner: str
    repo: str
    type: str
    token: Optional[str] = None
    localPath: Optional[str] = None
    repoUrl: Optional[str] = None


class WikiSection(BaseModel):
    """Wiki 章节"""
    id: str
    title: str
    pages: List[str]
    subsections: Optional[List[str]] = None


class WikiStructureModel(BaseModel):
    """Wiki 结构模型"""
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: Optional[List[WikiSection]] = None
    rootSections: Optional[List[str]] = None


class WikiCacheData(BaseModel):
    """Wiki 缓存数据"""
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None
    repo: Optional[RepoInfo] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class WikiCacheRequest(BaseModel):
    """Wiki 缓存请求"""
    repo: RepoInfo
    language: str
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    provider: str
    model: str


class WikiExportRequest(BaseModel):
    """Wiki 导出请求"""
    repo_url: str = Field(..., description="Repository URL")
    pages: List[WikiPage] = Field(..., description="Wiki pages to export")
    format: Literal["markdown", "json"] = Field(..., description="Export format")


class Model(BaseModel):
    """LLM 模型"""
    id: str
    name: str


class Provider(BaseModel):
    """LLM 提供者"""
    id: str
    name: str
    models: List[Model]
    supportsCustomModel: Optional[bool] = False


class ModelConfig(BaseModel):
    """模型配置"""
    providers: List[Provider]
    defaultProvider: str


class AuthorizationConfig(BaseModel):
    """授权配置"""
    code: str


# ============================================================
# API 端点
# ============================================================


@app.get("/lang/config")
async def get_lang_config():
    """获取语言配置"""
    return configs.get("lang_config", {
        "supported_languages": {"en": "English"},
        "default": "zh",
    })


@app.get("/auth/status")
async def get_auth_status():
    """检查是否需要认证"""
    return {"auth_required": WIKI_AUTH_MODE}


@app.post("/auth/validate")
async def validate_auth_code(request: AuthorizationConfig):
    """验证授权码"""
    return {"success": WIKI_AUTH_CODE == request.code}


@app.get("/models/config", response_model=ModelConfig)
async def get_model_config():
    """
    获取可用的模型提供者和模型列表

    从 generator.json 配置中读取提供者和模型信息。
    """
    try:
        logger.info("Fetching model configurations")

        providers = []
        default_provider = configs.get("default_provider", "dashscope")

        for provider_id, provider_config in configs.get("providers", {}).items():
            models = []
            for model_id in provider_config.get("models", {}).keys():
                models.append(Model(id=model_id, name=model_id))

            providers.append(
                Provider(
                    id=provider_id,
                    name=provider_id.capitalize(),
                    supportsCustomModel=provider_config.get("supportsCustomModel", False),
                    models=models,
                )
            )

        return ModelConfig(providers=providers, defaultProvider=default_provider)

    except Exception as e:
        logger.error(f"Error creating model configuration: {str(e)}")
        return ModelConfig(
            providers=[
                Provider(
                    id="dashscope",
                    name="DashScope",
                    supportsCustomModel=True,
                    models=[Model(id="qwen-plus", name="Qwen Plus")],
                )
            ],
            defaultProvider="dashscope",
        )


@app.post("/export/wiki")
async def export_wiki(request: WikiExportRequest):
    """
    导出 Wiki 内容为 Markdown 或 JSON

    Args:
        request: 导出请求

    Returns:
        可下载的文件
    """
    try:
        logger.info(f"Exporting wiki for {request.repo_url} in {request.format} format")

        repo_parts = request.repo_url.rstrip("/").split("/")
        repo_name = repo_parts[-1] if repo_parts else "wiki"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except Exception as e:
        error_msg = f"Error exporting wiki: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/local_repo/structure")
async def get_local_repo_structure(path: str = Query(None, description="Path to local repository")):
    """返回本地仓库的文件树和 README 内容"""
    if not path:
        return JSONResponse(
            status_code=400,
            content={"error": "No path provided. Please provide a 'path' query parameter."},
        )

    if not os.path.isdir(path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Directory not found: {path}"},
        )

    try:
        logger.info(f"Processing local repository at: {path}")
        file_tree_lines = []
        readme_content = ""

        for root, dirs, files in os.walk(path):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".")
                and d != "__pycache__"
                and d != "node_modules"
                and d != ".venv"
            ]
            for file in files:
                if file.startswith(".") or file == "__init__.py" or file == ".DS_Store":
                    continue
                rel_dir = os.path.relpath(root, path)
                rel_file = os.path.join(rel_dir, file) if rel_dir != "." else file
                file_tree_lines.append(rel_file)
                if file.lower() == "readme.md" and not readme_content:
                    try:
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read README.md: {str(e)}")
                        readme_content = ""

        file_tree_str = "\n".join(sorted(file_tree_lines))
        return {"file_tree": file_tree_str, "readme": readme_content}

    except Exception as e:
        logger.error(f"Error processing local repository: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing local repository: {str(e)}"},
        )


# ============================================================
# Wiki 导出辅助函数
# ============================================================


def generate_markdown_export(repo_url: str, pages: List[WikiPage]) -> str:
    """生成 Markdown 导出"""
    markdown = f"# Wiki Documentation for {repo_url}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    markdown += "## Table of Contents\n\n"
    for page in pages:
        markdown += f"- [{page.title}](#{page.id})\n"
    markdown += "\n"

    for page in pages:
        markdown += f"<a id='{page.id}'></a>\n\n"
        markdown += f"## {page.title}\n\n"

        if page.relatedPages and len(page.relatedPages) > 0:
            markdown += "### Related Pages\n\n"
            related_titles = []
            for related_id in page.relatedPages:
                related_page = next((p for p in pages if p.id == related_id), None)
                if related_page:
                    related_titles.append(f"[{related_page.title}](#{related_id})")
            if related_titles:
                markdown += "Related topics: " + ", ".join(related_titles) + "\n\n"

        markdown += f"{page.content}\n\n"
        markdown += "---\n\n"

    return markdown


def generate_json_export(repo_url: str, pages: List[WikiPage]) -> str:
    """生成 JSON 导出"""
    export_data = {
        "metadata": {
            "repository": repo_url,
            "generated_at": datetime.now().isoformat(),
            "page_count": len(pages),
        },
        "pages": [page.model_dump() for page in pages],
    }
    return json.dumps(export_data, indent=2)


# ============================================================
# 导入聊天和 WebSocket 端点
# ============================================================

from api.simple_chat import router as chat_router
from api.websocket_wiki import handle_websocket_chat

# 注册聊天路由
app.include_router(chat_router)

# 注册 WebSocket 端点
app.add_api_websocket_route("/ws/chat", handle_websocket_chat)


# ============================================================
# Wiki 缓存辅助函数
# ============================================================


def get_wiki_cache_path(owner: str, repo: str, repo_type: str, language: str) -> str:
    """生成 Wiki 缓存文件路径"""
    filename = f"OpenWiki_cache_{repo_type}_{owner}_{repo}_{language}.json"
    return os.path.join(WIKI_CACHE_DIR, filename)


async def read_wiki_cache(owner: str, repo: str, repo_type: str, language: str) -> Optional[WikiCacheData]:
    """从文件系统读取 Wiki 缓存"""
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"Error reading wiki cache from {cache_path}: {e}")
            return None
    return None


async def save_wiki_cache(data: WikiCacheRequest) -> bool:
    """保存 Wiki 缓存到文件系统"""
    cache_path = get_wiki_cache_path(data.repo.owner, data.repo.repo, data.repo.type, data.language)
    logger.info(f"Attempting to save wiki cache. Path: {cache_path}")
    try:
        payload = WikiCacheData(
            wiki_structure=data.wiki_structure,
            generated_pages=data.generated_pages,
            repo=data.repo,
            provider=data.provider,
            model=data.model,
        )
        try:
            payload_json = payload.model_dump_json()
            payload_size = len(payload_json.encode("utf-8"))
            logger.info(f"Payload prepared for caching. Size: {payload_size} bytes.")
        except Exception as ser_e:
            logger.warning(f"Could not serialize payload for size logging: {ser_e}")

        logger.info(f"Writing cache file to: {cache_path}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload.model_dump(), f, indent=2)
        logger.info(f"Wiki cache successfully saved to {cache_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving wiki cache to {cache_path}: {e.strerror} (errno: {e.errno})", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving wiki cache to {cache_path}: {e}", exc_info=True)
        return False


# ============================================================
# Wiki 缓存 API 端点
# ============================================================


@app.get("/api/wiki_cache", response_model=Optional[WikiCacheData])
async def get_cached_wiki(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content"),
):
    """获取缓存的 Wiki 数据"""
    supported_langs = configs.get("lang_config", {}).get("supported_languages", {})
    if language not in supported_langs:
        language = configs.get("lang_config", {}).get("default", "en")

    logger.info(f"Retrieving wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cached_data = await read_wiki_cache(owner, repo, repo_type, language)
    if cached_data:
        return cached_data
    else:
        logger.info(f"Wiki cache not found for {owner}/{repo} ({repo_type}), lang: {language}")
        return None


@app.post("/api/wiki_cache")
async def store_wiki_cache(request_data: WikiCacheRequest):
    """存储 Wiki 缓存"""
    supported_langs = configs.get("lang_config", {}).get("supported_languages", {})
    if request_data.language not in supported_langs:
        request_data.language = configs.get("lang_config", {}).get("default", "en")

    logger.info(
        f"Saving wiki cache for {request_data.repo.owner}/{request_data.repo.repo} "
        f"({request_data.repo.type}), lang: {request_data.language}"
    )
    success = await save_wiki_cache(request_data)
    if success:
        return {"message": "Wiki cache saved successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save wiki cache")


@app.delete("/api/wiki_cache")
async def delete_wiki_cache(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content"),
    authorization_code: Optional[str] = Query(None, description="Authorization code"),
):
    """删除 Wiki 缓存"""
    supported_langs = configs.get("lang_config", {}).get("supported_languages", {})
    if language not in supported_langs:
        raise HTTPException(status_code=400, detail="Language is not supported")

    if WIKI_AUTH_MODE:
        logger.info("Checking authorization code")
        if not authorization_code or WIKI_AUTH_CODE != authorization_code:
            raise HTTPException(status_code=401, detail="Authorization code is invalid")

    logger.info(f"Deleting wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"Successfully deleted wiki cache: {cache_path}")
            return {"message": f"Wiki cache for {owner}/{repo} ({language}) deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting wiki cache {cache_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete wiki cache: {str(e)}")
    else:
        logger.warning(f"Wiki cache not found, cannot delete: {cache_path}")
        raise HTTPException(status_code=404, detail="Wiki cache not found")


# ============================================================
# 健康检查和根端点
# ============================================================


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "openwiki-api",
        "version": "2.0.0",
    }


@app.get("/")
async def root():
    """根端点 — 列出所有可用端点"""
    endpoints = {}
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            if route.path in ["/openapi.json", "/docs", "/redoc", "/favicon.ico"]:
                continue
            path_parts = route.path.strip("/").split("/")
            group = path_parts[0].capitalize() if path_parts[0] else "Root"
            method_list = list(route.methods - {"HEAD", "OPTIONS"})
            for method in method_list:
                endpoints.setdefault(group, []).append(f"{method} {route.path}")

    for group in endpoints:
        endpoints[group].sort()

    return {
        "message": "Welcome to OpenWiki API (RAG Optimized)",
        "version": "2.0.0",
        "endpoints": endpoints,
    }


# ============================================================
# 已处理项目列表
# ============================================================


@app.get("/api/processed_projects", response_model=List[ProcessedProjectEntry])
async def get_processed_projects():
    """
    列出所有已处理的项目

    从 Wiki 缓存目录中扫描缓存文件，解析项目信息。
    同时从 PostgreSQL 数据库中获取项目列表作为补充。
    """
    project_entries: List[ProcessedProjectEntry] = []

    try:
        # 1. 从 Wiki 缓存目录扫描
        if os.path.exists(WIKI_CACHE_DIR):
            logger.info(f"Scanning for project cache files in: {WIKI_CACHE_DIR}")
            filenames = await asyncio.to_thread(os.listdir, WIKI_CACHE_DIR)

            for filename in filenames:
                if filename.startswith("deepwiki_cache_") and filename.endswith(".json"):
                    file_path = os.path.join(WIKI_CACHE_DIR, filename)
                    try:
                        stats = await asyncio.to_thread(os.stat, file_path)
                        parts = filename.replace("deepwiki_cache_", "").replace(".json", "").split("_")

                        if len(parts) >= 4:
                            repo_type = parts[0]
                            owner = parts[1]
                            language = parts[-1]
                            repo = "_".join(parts[2:-1])

                            project_entries.append(
                                ProcessedProjectEntry(
                                    id=filename,
                                    owner=owner,
                                    repo=repo,
                                    name=f"{owner}/{repo}",
                                    repo_type=repo_type,
                                    submittedAt=int(stats.st_mtime * 1000),
                                    language=language,
                                )
                            )
                        else:
                            logger.warning(f"Could not parse project details from filename: {filename}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue

        # 2. 从 PostgreSQL 数据库获取项目列表作为补充
        try:
            db_projects = ProjectRepository.list_all()
            for proj in db_projects:
                # 检查是否已存在（避免重复）
                existing_ids = {p.id for p in project_entries}
                proj_id = str(proj.get("id", ""))
                if proj_id not in existing_ids:
                    created_at = proj.get("created_at")
                    if isinstance(created_at, datetime):
                        timestamp = int(created_at.timestamp() * 1000)
                    else:
                        timestamp = 0

                    project_entries.append(
                        ProcessedProjectEntry(
                            id=proj_id,
                            owner=proj.get("owner") or "unknown",
                            repo=proj.get("name") or "unknown",
                            name=f"{proj.get('owner') or 'unknown'}/{proj.get('name') or 'unknown'}",
                            repo_type=proj.get("repo_type") or "github",
                            submittedAt=timestamp,
                            language="en",
                        )
                    )
        except Exception as e:
            logger.warning(f"Could not fetch projects from database: {e}")

        # 按时间排序（最新的在前）
        project_entries.sort(key=lambda p: p.submittedAt, reverse=True)
        logger.info(f"Found {len(project_entries)} processed project entries.")
        return project_entries

    except Exception as e:
        logger.error(f"Error listing processed projects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list processed projects.")
