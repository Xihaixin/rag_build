"""
数据处理管道 — 文档读取、分块、嵌入、存储

替代原始 deepwiki-open 中使用 LocalDB + .pkl 的数据处理管道，
底层使用 rag_optimizer 的 PostgreSQL + pgvector 存储。

注意: download_repo, read_all_documents, count_tokens, get_file_content
及系列函数已迁移至 core.utils.repo 和 core.utils.documents，
此处仅做重导出以保持向后兼容。
"""

import logging
import os
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
from core.utils.repo import (
    download_repo,
    get_file_content,
    get_github_file_content,
    get_gitlab_file_content,
    get_bitbucket_file_content,
)
from core.utils.documents import count_tokens, read_all_documents
from rag_optimizer.config.settings import settings
from rag_optimizer.db.repository import ProjectRepository, DocumentRepository, ChunkRepository, EmbeddingRepository
from rag_optimizer.pipeline.ingestion import TextSplitter, Embedder, IngestionPipeline

logger = logging.getLogger(__name__)


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
