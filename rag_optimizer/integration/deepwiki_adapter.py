"""
deepwiki-open 集成适配器

提供与 deepwiki-open (adalflow_processing.py) 的兼容层，使现有代码
能够无缝切换到 pgvector 后端，而无需修改原有业务逻辑。

核心功能：
1. PgvectorRetriever — 替代 FAISSRetriever 的兼容类
2. PgvectorDatabaseManager — 替代 DatabaseManager 的兼容类
3. patch_adalflow() — 一键注入 pgvector 后端的 monkey-patch

用法：
    # 方式 1: 直接使用 PgvectorRetriever
    from rag_optimizer.integration.deepwiki_adapter import PgvectorRetriever
    retriever = PgvectorRetriever(project_id="xxx")
    results = retriever(query, k=5)

    # 方式 2: Monkey-patch adalflow_processing
    from rag_optimizer.integration.deepwiki_adapter import patch_adalflow
    patch_adalflow()
    # 之后 adalflow_processing.RAG 将自动使用 pgvector

    # 方式 3: 替换 RAG 类的 retriever
    from rag_optimizer.integration.deepwiki_adapter import create_pgvector_rag
    rag = create_pgvector_rag(project_id="xxx")
    results = rag("查询文本")
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_optimizer.config.settings import settings
from rag_optimizer.db.connection import sync_conn
from rag_optimizer.pipeline.ingestion import Embedder
from rag_optimizer.retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievalResult,
    RetrievalStats,
)

logger = logging.getLogger(__name__)


# ============================================================
# PgvectorRetriever — 兼容 FAISSRetriever 接口
# ============================================================

class PgvectorRetriever:
    """
    pgvector 检索器 — 兼容 adalflow FAISSRetriever 接口

    实现 __call__ 方法，使其可以像 FAISSRetriever 一样被调用：
        retriever(query: str, k: int = 5) -> List[Document]

    同时保留 HybridRetriever 的全部能力。
    """

    def __init__(
        self,
        project_id: str,
        model_name: Optional[str] = None,
        retrieval_type: str = "hybrid",
        top_k: int = 10,
    ):
        """
        Args:
            project_id: 项目 ID
            model_name: 嵌入模型名称
            retrieval_type: 检索类型 (vector_only, keyword_only, hybrid, rrf)
            top_k: 默认返回结果数
        """
        self.project_id = project_id
        self.model_name = model_name or settings.embedding.default_model
        self.retrieval_type = retrieval_type
        self.top_k = top_k

        self._hybrid_retriever = HybridRetriever(
            project_id=project_id,
            model_name=self.model_name,
        )
        self._embedder = Embedder(model_name=self.model_name)

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本的嵌入向量

        Args:
            text: 查询文本

        Returns:
            嵌入向量列表，获取失败时返回 None
        """
        try:
            return self._embedder.embed_one(text)
        except Exception as e:
            logger.error(f"Failed to get embedding for query '{text[:50]}': {e}", exc_info=True)
            return None

    # ----------------------------------------------------------
    # 兼容 FAISSRetriever 接口
    # ----------------------------------------------------------

    def __call__(self, query: str, k: Optional[int] = None) -> List[Any]:
        """
        兼容 FAISSRetriever 的调用方式。

        Args:
            query: 查询文本
            k: 返回结果数

        Returns:
            List[Document] — 兼容 adalflow Document 格式（异常时返回空列表）
        """
        top_k = k or self.top_k
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            logger.warning(f"PgvectorRetriever: embedding failed for query '{query[:50]}', returning empty")
            return []

        try:
            results, stats = self._hybrid_retriever.search(
                query_text=query,
                query_embedding=query_embedding,
                retrieval_type=self.retrieval_type,
                top_k=top_k,
                log_retrieval=True,
            )
        except Exception as e:
            logger.error(f"PgvectorRetriever: search failed: {e}", exc_info=True)
            return []

        # 转换为兼容的 Document 格式
        documents = []
        for r in results:
            doc = _create_compat_document(r)
            documents.append(doc)

        logger.debug(
            f"PgvectorRetriever: query='{query[:50]}', "
            f"{len(documents)} results, {stats.latency_ms:.1f}ms"
        )
        return documents

    # ----------------------------------------------------------
    # 原生接口
    # ----------------------------------------------------------

    def search(
        self,
        query: str,
        retrieval_type: Optional[str] = None,
        top_k: Optional[int] = None,
        file_pattern: Optional[str] = None,
    ) -> Tuple[List[RetrievalResult], Any]:
        """使用原生接口检索

        Returns:
            (检索结果列表, 检索统计) — 异常时返回 ([], stats)
        """
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            logger.warning(f"PgvectorRetriever.search: embedding failed for query '{query[:50]}'")
            empty_stats = RetrievalStats(
                retrieval_type=retrieval_type or self.retrieval_type,
                top_k=top_k or self.top_k,
                latency_ms=0,
                total_results=0,
                query_text=query,
            )
            return [], empty_stats

        try:
            return self._hybrid_retriever.search(
                query_text=query,
                query_embedding=query_embedding,
                retrieval_type=retrieval_type or self.retrieval_type,
                top_k=top_k or self.top_k,
                file_pattern=file_pattern,
                log_retrieval=True,
            )
        except Exception as e:
            logger.error(f"PgvectorRetriever.search failed: {e}", exc_info=True)
            empty_stats = RetrievalStats(
                retrieval_type=retrieval_type or self.retrieval_type,
                top_k=top_k or self.top_k,
                latency_ms=0,
                total_results=0,
                query_text=query,
            )
            return [], empty_stats

    def vector_search(self, query: str, top_k: Optional[int] = None):
        """纯向量检索"""
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return []
        return self._hybrid_retriever.vector_search(query_embedding, top_k=top_k or self.top_k)

    def keyword_search(self, query: str, top_k: Optional[int] = None):
        """纯关键词检索"""
        return self._hybrid_retriever.keyword_search(query, top_k=top_k or self.top_k)

    def hybrid_search(self, query: str, top_k: Optional[int] = None):
        """加权融合检索"""
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return []
        return self._hybrid_retriever.hybrid_search(query, query_embedding, top_k=top_k or self.top_k)

    def rrf_search(self, query: str, top_k: Optional[int] = None):
        """RRF 融合检索"""
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return []
        return self._hybrid_retriever.rrf_search(query, query_embedding, top_k=top_k or self.top_k)


# ============================================================
# 兼容 Document 类
# ============================================================

def _create_compat_document(result: RetrievalResult) -> Any:
    """
    将 RetrievalResult 转换为兼容 adalflow Document 格式的对象。

    返回的对象具有以下属性（与 adalflow Document 兼容）：
        - id: str
        - text: str (内容)
        - vector: Optional[List[float]]
        - meta: Dict (包含 file_path, score 等)
    """
    class CompatDocument:
        def __init__(self, r: RetrievalResult):
            self.id = r.chunk_id
            self.text = r.content
            self.vector = None  # 不存储向量以减少内存
            self.meta = {
                "file_path": r.file_path,
                "score": r.final_score,
                "vector_score": r.vector_score,
                "keyword_score": r.keyword_score,
            }

        def __repr__(self):
            return f"Document(id={self.id}, meta={self.meta})"

    return CompatDocument(result)


# ============================================================
# PgvectorDatabaseManager — 兼容 DatabaseManager 接口
# ============================================================

class PgvectorDatabaseManager:
    """
    pgvector 数据库管理器 — 兼容 adalflow_processing.DatabaseManager 接口

    提供 prepare_database, prepare_db_index 等方法，使现有代码
    可以无缝切换到 pgvector 后端。
    """

    def __init__(self):
        self.project_id: Optional[str] = None
        self.repo_name: Optional[str] = None

    def prepare_database(
        self,
        repo_url_or_path: str,
        repo_type: str = "gitee",
        access_token: Optional[str] = None,
    ) -> str:
        """
        准备数据库 — 创建或获取项目记录。

        Returns:
            project_id (str)
        """
        from rag_optimizer.db.repository import ProjectRepository

        # 提取仓库名
        if repo_type == "gitee" and "gitee.com" in repo_url_or_path:
            repo_name = repo_url_or_path.rstrip("/").split("/")[-1]
        elif "github.com" in repo_url_or_path:
            repo_name = repo_url_or_path.rstrip("/").split("/")[-1]
        else:
            repo_name = Path(repo_url_or_path).name

        self.repo_name = repo_name

        # 创建或获取项目
        project = ProjectRepository.get_or_create(
            name=repo_name,
            repo_url=repo_url_or_path,
        )
        self.project_id = str(project["id"])

        logger.info(f"Database prepared: project={repo_name}, id={self.project_id}")
        return self.project_id

    def prepare_db_index(self, *args, **kwargs):
        """
        准备数据库索引 — pgvector 索引在 schema 创建时已建立。
        此方法仅用于兼容性。
        """
        logger.info("pgvector indexes already created in schema.")
        return True

    def reset_database(self):
        """重置数据库 — 清空项目相关数据"""
        if not self.project_id:
            logger.warning("No project_id set, cannot reset.")
            return

        try:
            sync_conn.execute(
                "DELETE FROM qa_logs WHERE project_id = %s", (self.project_id,)
            )
            sync_conn.execute(
                "DELETE FROM retrieval_results WHERE retrieval_id IN "
                "(SELECT id FROM retrieval_logs WHERE project_id = %s)",
                (self.project_id,),
            )
            sync_conn.execute(
                "DELETE FROM retrieval_logs WHERE project_id = %s",
                (self.project_id,),
            )
            sync_conn.execute(
                "DELETE FROM chunk_embeddings_dim256 WHERE chunk_id IN "
                "(SELECT id FROM document_chunks WHERE document_id IN "
                "(SELECT id FROM raw_documents WHERE project_id = %s))",
                (self.project_id,),
            )
            sync_conn.execute(
                "DELETE FROM document_chunks WHERE document_id IN "
                "(SELECT id FROM raw_documents WHERE project_id = %s)",
                (self.project_id,),
            )
            sync_conn.execute(
                "DELETE FROM raw_documents WHERE project_id = %s",
                (self.project_id,),
            )
            logger.info(f"Database reset for project {self.project_id}")
        except Exception as e:
            logger.error(f"Database reset error: {e}")


# ============================================================
# Monkey-patch 工具
# ============================================================

def patch_adalflow():
    """
    一键注入 pgvector 后端的 monkey-patch。

    替换 adalflow_processing 中的关键组件：
    1. DatabaseManager → PgvectorDatabaseManager
    2. RAG.prepare_retriever → 使用 PgvectorRetriever
    3. RAG.call → 使用 pgvector 检索

    调用后，adalflow_processing.RAG 将自动使用 pgvector 后端。
    """
    try:
        import adalflow_processing
    except ImportError:
        logger.error(
            "adalflow_processing not found. Make sure you're running "
            "from the project root directory."
        )
        return False

    original_prepare_retriever = adalflow_processing.RAG.prepare_retriever
    original_call = adalflow_processing.RAG.call

    def _patched_prepare_retriever(self, *args, **kwargs):
        """
        替换后的 prepare_retriever 方法。

        使用 PgvectorRetriever 替代 FAISSRetriever。
        """
        logger.info("[pgvector] Patching RAG.prepare_retriever...")

        # 获取 project_id
        if hasattr(self, 'db_manager') and hasattr(self.db_manager, 'project_id'):
            project_id = self.db_manager.project_id
        else:
            # 尝试从已有数据获取
            project_id = kwargs.get("project_id", args[0] if args else None)

        if not project_id:
            logger.warning("[pgvector] No project_id found, falling back to original")
            return original_prepare_retriever(self, *args, **kwargs)

        # 创建 PgvectorRetriever
        pg_retriever = PgvectorRetriever(
            project_id=project_id,
            retrieval_type="hybrid",
            top_k=10,
        )

        # 替换 retriever
        self.retriever = pg_retriever
        self._pgvector_mode = True

        logger.info(f"[pgvector] PgvectorRetriever initialized for project {project_id}")
        return True

    def _patched_call(self, query: str, language: str = "zh") -> Tuple[List]:
        """
        替换后的 call 方法。

        使用 pgvector 检索替代 FAISS 检索。
        """
        if not getattr(self, '_pgvector_mode', False):
            return original_call(self, query, language)

        logger.debug(f"[pgvector] RAG.call: query='{query[:50]}'")

        try:
            # 使用 pgvector 检索
            if hasattr(self, 'retriever') and isinstance(self.retriever, PgvectorRetriever):
                results = self.retriever(query, k=10)
            else:
                results = []

            # 构建返回格式（兼容原有逻辑）
            if hasattr(self, 'memory'):
                memory_output = self.memory()
            else:
                memory_output = {}

            return results, memory_output

        except Exception as e:
            logger.error(f"[pgvector] RAG.call error: {e}")
            return [], {}

    # 应用 patch
    adalflow_processing.RAG.prepare_retriever = _patched_prepare_retriever
    adalflow_processing.RAG.call = _patched_call

    # 替换 DatabaseManager
    adalflow_processing.DatabaseManager = PgvectorDatabaseManager

    logger.info("[pgvector] adalflow_processing patched successfully!")
    logger.info("[pgvector]   DatabaseManager -> PgvectorDatabaseManager")
    logger.info("[pgvector]   RAG.prepare_retriever -> pgvector")
    logger.info("[pgvector]   RAG.call -> pgvector")

    return True


# ============================================================
# 便捷工厂函数
# ============================================================

def create_pgvector_rag(
    project_id: str,
    retrieval_type: str = "hybrid",
    top_k: int = 10,
) -> Any:
    """
    创建使用 pgvector 后端的 RAG 实例。

    返回一个兼容 adalflow RAG 接口的对象，可以直接调用：
        rag = create_pgvector_rag("project_id")
        results = rag("查询文本")

    Args:
        project_id: 项目 ID
        retrieval_type: 检索类型
        top_k: 返回结果数

    Returns:
        PgvectorRAG 实例
    """
    from rag_optimizer.pipeline.rag_engine import RAGEngine

    class PgvectorRAG:
        """
        兼容 adalflow RAG 接口的 pgvector RAG 包装器。
        """

        def __init__(self, project_id: str, retrieval_type: str, top_k: int):
            self.project_id = project_id
            self.retrieval_type = retrieval_type
            self.top_k = top_k
            self.engine = RAGEngine(project_id=project_id)
            self.retriever = PgvectorRetriever(
                project_id=project_id,
                retrieval_type=retrieval_type,
                top_k=top_k,
            )
            self.memory = lambda: {}

        def __call__(self, query: str, language: str = "zh") -> Tuple[List]:
            """
            兼容 adalflow RAG.call 接口。

            Returns:
                (results, memory_output)
            """
            ctx = self.engine.answer(
                query=query,
                retrieval_type=self.retrieval_type,
                top_k=self.top_k,
                language=language,
                use_semantic_cache=True,
            )

            # 转换为兼容格式
            results = [_create_compat_document(r) for r in ctx.results]
            memory_output = self.memory()

            return results, memory_output

        def prepare_retriever(self, *args, **kwargs):
            """兼容接口"""
            return True

    return PgvectorRAG(
        project_id=project_id,
        retrieval_type=retrieval_type,
        top_k=top_k,
    )


# ============================================================
# CLI 入口
# ============================================================

def main():
    """CLI 入口：测试集成"""
    import argparse

    parser = argparse.ArgumentParser(description="deepwiki-open 集成测试")
    parser.add_argument("--project-id", required=True, help="项目 ID")
    parser.add_argument("--query", default="这个项目的主要功能是什么？", help="测试查询")
    parser.add_argument("--mode", choices=["direct", "patch", "factory"], default="direct")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.mode == "patch":
        # Monkey-patch 模式
        logger.info("Testing monkey-patch mode...")
        patch_adalflow()
        try:
            import adalflow_processing
            rag = adalflow_processing.RAG()
            rag.db_manager = PgvectorDatabaseManager()
            rag.db_manager.project_id = args.project_id
            rag.prepare_retriever()
            results, memory = rag(args.query)
            logger.info(f"Results: {len(results)} documents")
            for r in results[:3]:
                logger.info(f"  - {r.meta.get('file_path', 'unknown')}: {r.text[:100]}...")
        except Exception as e:
            logger.error(f"Patch mode error: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == "factory":
        # 工厂模式
        logger.info("Testing factory mode...")
        rag = create_pgvector_rag(project_id=args.project_id)
        results, memory = rag(args.query)
        logger.info(f"Results: {len(results)} documents")
        for r in results[:3]:
            logger.info(f"  - {r.meta.get('file_path', 'unknown')}: {r.text[:100]}...")

    else:
        # 直接模式
        logger.info("Testing direct mode...")
        retriever = PgvectorRetriever(project_id=args.project_id)
        results = retriever(args.query, k=5)
        logger.info(f"Results: {len(results)} documents")
        for r in results[:3]:
            logger.info(f"  - {r.meta.get('file_path', 'unknown')}: {r.text[:100]}...")


if __name__ == "__main__":
    main()
