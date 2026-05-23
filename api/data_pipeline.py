"""
数据处理管道 — 文档读取、分块、嵌入、存储

替代原始 deepwiki-open 中使用 LocalDB + .pkl 的数据处理管道，
底层使用 rag_optimizer 的 PostgreSQL + pgvector 存储。
"""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import adalflow as adal

from api.config import (
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    configs,
    get_embedder_config,
)
from rag_optimizer.config.settings import settings
from rag_optimizer.db.repository import ProjectRepository, DocumentRepository, ChunkRepository, EmbeddingRepository
from rag_optimizer.pipeline.ingestion import TextSplitter, Embedder, IngestionPipeline

logger = logging.getLogger(__name__)


# ============================================================
# Token 计数
# ============================================================


def count_tokens(text: str, embedder_type: Optional[str] = None, is_ollama_embedder: Optional[bool] = None) -> int:
    """
    计算文本的 token 数量

    Args:
        text: 文本内容
        embedder_type: 嵌入器类型
        is_ollama_embedder: 是否使用 Ollama 嵌入器

    Returns:
        int: token 数量
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # 回退：使用简单估算（约 4 字符/token）
        return len(text) // 4


# ============================================================
# 仓库下载
# ============================================================


def download_repo(
    repo_url: str,
    local_path: str,
    repo_type: Optional[str] = None,
    access_token: Optional[str] = None,
) -> str:
    """
    下载仓库到本地

    Args:
        repo_url: 仓库 URL
        local_path: 本地路径
        repo_type: 仓库类型 (github, gitlab, bitbucket, gitee)
        access_token: 访问令牌

    Returns:
        str: 本地路径
    """
    logger.info(f"Downloading repo: {repo_url} to {local_path}")

    # 如果本地路径已存在，跳过下载
    if os.path.exists(local_path) and os.listdir(local_path):
        logger.info(f"Local path already exists: {local_path}")
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        # 构建带认证的 URL
        if access_token:
            parsed = urlparse(repo_url)
            auth_url = f"{parsed.scheme}://{access_token}@{parsed.netloc}{parsed.path}"
        else:
            auth_url = repo_url

        # 执行 git clone
        result = subprocess.run(
            ["git", "clone", "--depth=1", auth_url, local_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 分钟超时
        )

        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr}")
            raise RuntimeError(f"Failed to clone repository: {result.stderr}")

        logger.info(f"Repository cloned successfully to {local_path}")
        return local_path

    except subprocess.TimeoutExpired:
        logger.error("Git clone timed out")
        raise RuntimeError("Repository clone timed out")
    except Exception as e:
        logger.error(f"Error downloading repo: {e}")
        raise


# ============================================================
# 文档读取
# ============================================================


def read_all_documents(
    path: str,
    embedder_type: Optional[str] = None,
    is_ollama_embedder: Optional[bool] = None,
    excluded_dirs: Optional[List[str]] = None,
    excluded_files: Optional[List[str]] = None,
    included_dirs: Optional[List[str]] = None,
    included_files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    递归读取目录中的所有文档文件

    Args:
        path: 目录路径
        embedder_type: 嵌入器类型
        is_ollama_embedder: 是否使用 Ollama 嵌入器
        excluded_dirs: 排除的目录列表
        excluded_files: 排除的文件列表
        included_dirs: 包含的目录列表
        included_files: 包含的文件列表

    Returns:
        List[Dict]: 文档列表，每项包含 file_path, content, file_type
    """
    excluded_dirs = excluded_dirs or DEFAULT_EXCLUDED_DIRS
    excluded_files = excluded_files or DEFAULT_EXCLUDED_FILES

    documents: List[Dict[str, Any]] = []

    # 规范化排除目录
    normalized_excluded_dirs = []
    for d in excluded_dirs:
        d = d.strip("./").strip("/")
        if d:
            normalized_excluded_dirs.append(d)

    def should_process_file(
        file_path: str,
        use_inclusion: bool,
        included_dirs_list: List[str],
        included_files_list: List[str],
        excluded_dirs_list: List[str],
        excluded_files_list: List[str],
    ) -> bool:
        """判断文件是否应该被处理"""
        rel_path = os.path.relpath(file_path, path).replace("\\", "/")

        # 检查排除目录
        for excl_dir in excluded_dirs_list:
            if rel_path.startswith(excl_dir + "/") or rel_path == excl_dir:
                return False

        # 检查排除文件
        for excl_file in excluded_files_list:
            if excl_file.startswith("*."):
                # 通配符匹配
                ext = excl_file[1:]
                if rel_path.endswith(ext):
                    return False
            elif excl_file == os.path.basename(rel_path):
                return False

        # 包含模式
        if use_inclusion:
            in_included = False
            for inc_dir in included_dirs_list:
                if rel_path.startswith(inc_dir + "/") or rel_path == inc_dir:
                    in_included = True
                    break
            for inc_file in included_files_list:
                if inc_file == os.path.basename(rel_path):
                    in_included = True
                    break
            return in_included

        return True

    use_inclusion_mode = bool(included_dirs or included_files)
    included_dirs_list = included_dirs or []
    included_files_list = included_files or []

    # 支持的文件扩展名
    text_extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".hpp",
        ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".dart",
        ".md", ".mdx", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
        ".cfg", ".conf", ".xml", ".html", ".css", ".scss", ".less", ".sql",
        ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".dockerfile",
        ".gradle", ".sbt", ".clj", ".ex", ".exs", ".erl", ".hrl",
        ".lua", ".r", ".m", ".mm", ".pl", ".pm", ".t", ".pod",
        ".vue", ".svelte", ".astro", ".graphql", ".gql", ".proto",
        ".cmake", ".makefile", ".gnumakefile", ".dockerignore",
        ".env.example", ".env.sample",
    }

    for root, dirs, files in os.walk(path):
        # 过滤排除目录
        rel_root = os.path.relpath(root, path).replace("\\", "/")
        dirs[:] = [
            d for d in dirs
            if d not in normalized_excluded_dirs
            and not d.startswith(".")
        ]

        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, path).replace("\\", "/")

            # 检查是否应该处理
            if not should_process_file(
                file_path,
                use_inclusion_mode,
                included_dirs_list,
                included_files_list,
                normalized_excluded_dirs,
                excluded_files,
            ):
                continue

            # 检查文件扩展名
            ext = os.path.splitext(file)[1].lower()
            if ext not in text_extensions and file not in (
                "Dockerfile", "Makefile", "GNUmakefile",
                "docker-compose.yml", "docker-compose.yaml",
            ):
                continue

            # 读取文件内容
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if content.strip():
                    documents.append({
                        "file_path": rel_path,
                        "content": content,
                        "file_type": ext.lstrip(".") if ext else "text",
                    })
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")

    logger.info(f"Read {len(documents)} documents from {path}")
    return documents


# ============================================================
# 数据管道
# ============================================================


def prepare_data_pipeline(
    embedder_type: Optional[str] = None,
    is_ollama_embedder: Optional[bool] = None,
) -> adal.Sequential:
    """
    创建 Adalflow 数据处理管道

    Args:
        embedder_type: 嵌入器类型
        is_ollama_embedder: 是否使用 Ollama 嵌入器

    Returns:
        adal.Sequential: 数据处理管道
    """
    # 使用 rag_optimizer 的 TextSplitter
    text_splitter = TextSplitter(
        chunk_size=settings.chunk.default_chunk_size,
        chunk_overlap=settings.chunk.default_chunk_overlap,
        split_by=settings.chunk.default_split_by,
    )

    # 使用 rag_optimizer 的 Embedder
    embedder = Embedder(
        model_name=settings.embedding.default_model,
        api_key=settings.embedding.dashscope_api_key,
    )

    # 创建管道
    pipeline = adal.Sequential(
        text_splitter,
        embedder,
    )

    logger.info("Data pipeline created with TextSplitter -> Embedder")
    return pipeline


def transform_documents_and_save_to_db(
    documents: List[Dict[str, Any]],
    project_id: str,
    embedder_type: Optional[str] = None,
    is_ollama_embedder: Optional[bool] = None,
) -> Dict[str, int]:
    """
    转换文档并保存到 PostgreSQL 数据库

    Args:
        documents: 文档列表
        project_id: 项目 ID
        embedder_type: 嵌入器类型
        is_ollama_embedder: 是否使用 Ollama 嵌入器

    Returns:
        Dict: 处理统计信息
    """
    stats = {
        "total_documents": len(documents),
        "processed_documents": 0,
        "total_chunks": 0,
        "total_embeddings": 0,
        "errors": 0,
    }

    # 使用 IngestionPipeline
    pipeline = IngestionPipeline(project_id=project_id)

    # 处理每个文档
    for doc in documents:
        try:
            chunks_count, embeddings_count = pipeline.process_document(
                file_path=doc["file_path"],
                content=doc["content"],
                file_type=doc.get("file_type"),
            )
            stats["processed_documents"] += 1
            stats["total_chunks"] += chunks_count
            stats["total_embeddings"] += embeddings_count
        except Exception as e:
            logger.error(f"Error processing document {doc['file_path']}: {e}")
            stats["errors"] += 1

    logger.info(
        f"Document processing complete: "
        f"{stats['processed_documents']}/{stats['total_documents']} docs, "
        f"{stats['total_chunks']} chunks, {stats['total_embeddings']} embeddings"
    )
    return stats


# ============================================================
# 文件内容获取（GitHub/GitLab/Bitbucket API）
# ============================================================


def get_github_file_content(repo_url: str, file_path: str, access_token: Optional[str] = None) -> str:
    """通过 GitHub API 获取文件内容"""
    import urllib.request
    import json

    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip("/").split("/")

    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {repo_url}")

    owner, repo = path_parts[0], path_parts[1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path.lstrip('/')}"

    headers = {
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "DeepWiki-Open",
    }
    if access_token:
        headers["Authorization"] = f"token {access_token}"

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Error fetching GitHub file {file_path}: {e}")
        raise


def get_gitlab_file_content(repo_url: str, file_path: str, access_token: Optional[str] = None) -> str:
    """通过 GitLab API 获取文件内容"""
    import urllib.request
    import urllib.parse

    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip("/").split("/")

    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitLab URL: {repo_url}")

    project_path = urllib.parse.quote("/".join(path_parts), safe="")
    encoded_file_path = urllib.parse.quote(file_path.lstrip("/"), safe="")
    api_url = f"https://gitlab.com/api/v4/projects/{project_path}/repository/files/{encoded_file_path}/raw"

    headers = {"User-Agent": "DeepWiki-Open"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Error fetching GitLab file {file_path}: {e}")
        raise


def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: Optional[str] = None) -> str:
    """通过 Bitbucket API 获取文件内容"""
    import urllib.request
    import urllib.parse

    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip("/").split("/")

    if len(path_parts) < 2:
        raise ValueError(f"Invalid Bitbucket URL: {repo_url}")

    owner, repo = path_parts[0], path_parts[1]
    encoded_path = urllib.parse.quote(file_path.lstrip("/"), safe="")
    api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/src/master/{encoded_path}"

    headers = {"User-Agent": "DeepWiki-Open"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Error fetching Bitbucket file {file_path}: {e}")
        raise


def get_file_content(
    repo_url: str,
    file_path: str,
    repo_type: Optional[str] = None,
    access_token: Optional[str] = None,
) -> str:
    """
    从远程仓库获取文件内容

    Args:
        repo_url: 仓库 URL
        file_path: 文件路径
        repo_type: 仓库类型 (github, gitlab, bitbucket)
        access_token: 访问令牌

    Returns:
        str: 文件内容
    """
    if repo_type == "gitlab" or "gitlab.com" in repo_url:
        return get_gitlab_file_content(repo_url, file_path, access_token)
    elif repo_type == "bitbucket" or "bitbucket.org" in repo_url:
        return get_bitbucket_file_content(repo_url, file_path, access_token)
    else:
        return get_github_file_content(repo_url, file_path, access_token)


# ============================================================
# DatabaseManager — 兼容原始接口
# ============================================================


class DatabaseManager:
    """
    数据库管理器 — 兼容原始 deepwiki-open 的 DatabaseManager 接口

    底层使用 rag_optimizer 的 PgvectorDatabaseManager 和 PostgreSQL。
    """

    def __init__(self):
        from rag_optimizer.integration.deepwiki_adapter import PgvectorDatabaseManager
        self._impl = PgvectorDatabaseManager()

    @property
    def project_id(self) -> Optional[str]:
        return self._impl.project_id

    @project_id.setter
    def project_id(self, value: Optional[str]):
        self._impl.project_id = value

    @property
    def repo_name(self) -> Optional[str]:
        return self._impl.repo_name

    def prepare_database(
        self,
        repo_url_or_path: str,
        repo_type: str = "gitee",
        access_token: Optional[str] = None,
    ) -> str:
        """准备数据库"""
        return self._impl.prepare_database(repo_url_or_path, repo_type, access_token)

    def reset_database(self):
        """重置数据库"""
        self._impl.reset_database()

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        """从 URL 提取仓库名"""
        if repo_type in ("gitee", "github") and ("gitee.com" in repo_url_or_path or "github.com" in repo_url_or_path):
            return repo_url_or_path.rstrip("/").split("/")[-1]
        return Path(repo_url_or_path).name

    def _create_repo(
        self,
        repo_url_or_path: str,
        repo_type: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> None:
        """创建仓库（下载到本地）"""
        adalflow_root = os.path.expanduser(os.path.join("~", ".adalflow"))
        repos_dir = os.path.join(adalflow_root, "repos")
        repo_name = self._extract_repo_name_from_url(repo_url_or_path, repo_type or "github")
        local_path = os.path.join(repos_dir, repo_name)

        download_repo(repo_url_or_path, local_path, repo_type, access_token)

    def prepare_db_index(
        self,
        embedder_type: Optional[str] = None,
        is_ollama_embedder: Optional[bool] = None,
        excluded_dirs: Optional[List[str]] = None,
        excluded_files: Optional[List[str]] = None,
        included_dirs: Optional[List[str]] = None,
        included_files: Optional[List[str]] = None,
    ) -> bool:
        """
        准备数据库索引 — 读取文档、分块、嵌入、存储到 PostgreSQL

        Returns:
            bool: 是否成功
        """
        if not self.project_id or not self.repo_name:
            logger.error("Database not prepared. Call prepare_database first.")
            return False

        try:
            adalflow_root = os.path.expanduser(os.path.join("~", ".adalflow"))
            repo_path = os.path.join(adalflow_root, "repos", self.repo_name)

            if not os.path.exists(repo_path):
                logger.error(f"Repository path not found: {repo_path}")
                return False

            # 读取文档
            documents = read_all_documents(
                path=repo_path,
                embedder_type=embedder_type,
                is_ollama_embedder=is_ollama_embedder,
                excluded_dirs=excluded_dirs,
                excluded_files=excluded_files,
                included_dirs=included_dirs,
                included_files=included_files,
            )

            if not documents:
                logger.warning("No documents found to index")
                return False

            # 转换并保存到数据库
            stats = transform_documents_and_save_to_db(
                documents=documents,
                project_id=self.project_id,
                embedder_type=embedder_type,
                is_ollama_embedder=is_ollama_embedder,
            )

            logger.info(f"Database index prepared: {stats}")
            return True

        except Exception as e:
            logger.error(f"Error preparing database index: {e}", exc_info=True)
            return False

    def prepare_retriever(
        self,
        repo_url_or_path: str,
        repo_type: str = "github",
        access_token: Optional[str] = None,
    ):
        """
        准备检索器 — 兼容原始接口

        Returns:
            PgvectorRetriever 实例
        """
        project_id = self.prepare_database(repo_url_or_path, repo_type, access_token)

        from rag_optimizer.integration.deepwiki_adapter import PgvectorRetriever
        return PgvectorRetriever(
            project_id=project_id,
            retrieval_type="hybrid",
            top_k=10,
        )
