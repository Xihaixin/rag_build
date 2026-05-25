"""
deepwiki-open 集成适配器

提供与 deepwiki-open (adalflow_processing.py) 的兼容层，使现有代码
能够无缝切换到 pgvector 后端，而无需修改原有业务逻辑。

核心功能：
1. PgvectorRetriever — 替代 FAISSRetriever 的兼容类，用于 WikiGenerationFlow
2. create_pgvector_rag() — 便捷工厂函数，创建兼容 adalflow RAG 接口的实例

用法：
    # 方式 1: 直接使用 PgvectorRetriever
    from rag_optimizer.integration.deepwiki_adapter import PgvectorRetriever
    retriever = PgvectorRetriever(project_id="xxx")
    results = retriever(query, k=5)

    # 方式 2: 使用工厂函数（底层使用 RAGEngine）
    from rag_optimizer.integration.deepwiki_adapter import create_pgvector_rag
    rag = create_pgvector_rag(project_id="xxx")
    results, memory = rag("查询文本")

架构定位:
    PgvectorRetriever 是一个薄适配层，核心价值是将 HybridRetriever.search()
    返回的 RetrievalResult 转换为 adalflow 兼容的 Document 格式。
    它不包含 LLM 生成、缓存等能力 — 这些由 core.rag_engine.RAGEngine 提供。

    与 RAGEngine 的职责区分:
    - RAGEngine: 高层 RAG 编排（检索 + 缓存 + LLM 生成 + qa_logs）
    - PgvectorRetriever: 底层检索适配器（仅检索 + 格式转换）
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

    用于 WikiGenerationFlow 等需要原生 search() 接口的场景。
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

    底层使用 core.rag_engine.RAGEngine 进行完整的 RAG 问答流程。

    Args:
        project_id: 项目 ID
        retrieval_type: 检索类型
        top_k: 返回结果数

    Returns:
        PgvectorRAG 实例
    """
    from core.rag_engine import RAGEngine

    class PgvectorRAG:
        """
        兼容 adalflow RAG 接口的 pgvector RAG 包装器。

        底层使用 RAGEngine 进行检索 + LLM 生成 + 缓存 + qa_logs。
        """

        def __init__(self, project_id: str, retrieval_type: str, top_k: int):
            self.project_id = project_id
            self.retrieval_type = retrieval_type
            self.top_k = top_k
            self.engine = RAGEngine(project_id=project_id)
            self.memory = lambda: {}

        def __call__(self, query: str, language: str = "zh") -> Tuple[List, Dict]:
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
    parser.add_argument("--mode", choices=["direct", "factory"], default="direct")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.mode == "factory":
        # 工厂模式（使用 RAGEngine）
        logger.info("Testing factory mode (RAGEngine)...")
        rag = create_pgvector_rag(project_id=args.project_id)
        results, memory = rag(args.query)
        logger.info(f"Results: {len(results)} documents")
        for r in results[:3]:
            logger.info(f"  - {r.meta.get('file_path', 'unknown')}: {r.text[:100]}...")

    else:
        # 直接模式（使用 PgvectorRetriever）
        logger.info("Testing direct mode (PgvectorRetriever)...")
        retriever = PgvectorRetriever(project_id=args.project_id)
        results = retriever(args.query, k=5)
        logger.info(f"Results: {len(results)} documents")
        for r in results[:3]:
            logger.info(f"  - {r.meta.get('file_path', 'unknown')}: {r.text[:100]}...")


if __name__ == "__main__":
    main()
