"""
混合检索引擎 — 向量相似度 + 全文检索 + RRF 融合

核心功能：
1. 纯向量检索（pgvector HNSW）
2. 纯关键词检索（tsvector GIN）
3. 混合检索（加权融合 + RRF 倒数排名融合）
4. 检索质量评估（MRR、NDCG、Recall）
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag_optimizer.config.settings import settings
from rag_optimizer.db.connection import sync_conn
from rag_optimizer.db.repository import (
    RetrievalRepository,
    compute_sha256,
    vector_to_str,
)

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class RetrievalResult:
    """检索结果"""
    chunk_id: str
    content: str
    file_path: str
    chunk_index: int
    vector_score: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalStats:
    """检索统计"""
    retrieval_type: str
    top_k: int
    latency_ms: int
    total_results: int
    query_text: str


# ============================================================
# 向量工具
# ============================================================

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算余弦相似度"""
    import math
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ============================================================
# 混合检索器
# ============================================================

class HybridRetriever:
    """
    混合检索器

    支持三种检索模式：
    - vector_only: 仅向量检索
    - keyword_only: 仅关键词检索
    - hybrid: 向量 + 关键词加权融合
    """

    def __init__(self, project_id: str, model_name: Optional[str] = None):
        """
        初始化检索器

        Args:
            project_id: 项目 ID
            model_name: 嵌入模型名称，默认使用配置中的默认模型

        Raises:
            ValueError: 嵌入模型在数据库中不存在时抛出
            RuntimeError: 数据库查询失败时抛出
        """
        self.project_id = project_id
        self.model_name = model_name or settings.embedding.default_model
        self.top_k = settings.retrieval.default_top_k
        self.vector_weight = settings.retrieval.hybrid_vector_weight
        self.keyword_weight = settings.retrieval.hybrid_keyword_weight

        # 获取模型 ID
        try:
            result = sync_conn.execute(
                "SELECT id, dimensions FROM embedding_models WHERE name = %s",
                (self.model_name,)
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query embedding model '{self.model_name}': {e}"
            ) from e

        if result:
            self.model_id = str(result[0]["id"])
            self.dimensions = result[0]["dimensions"]
        else:
            raise ValueError(f"Embedding model '{self.model_name}' not found in database")

        logger.info(f"HybridRetriever initialized: project={project_id}, model={self.model_name}")

    # ----------------------------------------------------------
    # 向量检索
    # ----------------------------------------------------------

    def vector_search(self, query_embedding: List[float], top_k: Optional[int] = None,
                      file_pattern: Optional[str] = None) -> List[RetrievalResult]:
        """
        纯向量检索（pgvector HNSW）

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数
            file_pattern: 可选的文件路径过滤模式

        Returns:
            检索结果列表（异常时返回空列表）
        """
        k = top_k or self.top_k
        embedding_str = vector_to_str(query_embedding)

        # 构建查询
        where_clauses = [
            "ce.project_id = %s",
            "ce.model_id = %s",
        ]
        # 注意：SQL 中 %s 的顺序是：%s::vector (<=>) → project_id → model_id → [file_pattern] → LIMIT
        params = [embedding_str, self.project_id, self.model_id]

        if file_pattern:
            where_clauses.append("ce.file_path LIKE %s")
            params.append(file_pattern)

        where_sql = " AND ".join(where_clauses)
        params.append(k)

        query = f"""
            SELECT
                ce.id,
                ce.chunk_id,
                ce.content,
                ce.file_path,
                ce.chunk_index,
                1 - (ce.embedding <=> %s::vector) AS vector_score
            FROM chunk_embeddings_dim256 ce
            WHERE {where_sql}
            ORDER BY vector_score DESC
            LIMIT %s
        """

        # 重新构建参数
        # 注意：SQL 中 %s 的顺序是：%s::vector (<=>) → project_id → model_id → [file_pattern] → LIMIT
        params = [embedding_str, self.project_id, self.model_id]
        if file_pattern:
            params.append(file_pattern)
        params.append(k)

        try:
            result = sync_conn.execute(query, tuple(params))
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return []

        if not result:
            return []

        return [
            RetrievalResult(
                chunk_id=str(r["chunk_id"]),
                content=r["content"] or "",
                file_path=r["file_path"] or "",
                chunk_index=r["chunk_index"] or 0,
                vector_score=float(r["vector_score"]),
            )
            for r in result
        ]

    # ----------------------------------------------------------
    # 关键词检索
    # ----------------------------------------------------------

    def keyword_search(self, query_text: str, top_k: Optional[int] = None,
                       file_pattern: Optional[str] = None) -> List[RetrievalResult]:
        """
        纯关键词检索（tsvector GIN）

        Args:
            query_text: 查询文本
            top_k: 返回结果数
            file_pattern: 可选的文件路径过滤模式

        Returns:
            检索结果列表（异常时返回空列表）
        """
        k = top_k or self.top_k

        where_clauses = [
            "ce.project_id = %s",
            "ce.model_id = %s",
        ]
        # 注意：SQL 中 %s 的顺序是：
        #   plainto_tsquery('simple', %s) [ts_rank] →
        #   project_id → model_id → [file_pattern] →
        #   plainto_tsquery('simple', %s) [@@] →
        #   LIMIT %s
        params = [query_text, self.project_id, self.model_id]

        if file_pattern:
            where_clauses.append("ce.file_path LIKE %s")
            params.append(file_pattern)

        where_sql = " AND ".join(where_clauses)
        params.append(query_text)  # 第二个 plainto_tsquery
        params.append(k)

        query = f"""
            SELECT
                ce.id,
                ce.chunk_id,
                ce.content,
                ce.file_path,
                ce.chunk_index,
                ts_rank(ce.fts_text, plainto_tsquery('simple', %s)) AS keyword_score
            FROM chunk_embeddings_dim256 ce
            WHERE {where_sql}
              AND ce.fts_text @@ plainto_tsquery('simple', %s)
            ORDER BY keyword_score DESC
            LIMIT %s
        """

        try:
            result = sync_conn.execute(query, tuple(params))
        except Exception as e:
            logger.error(f"Keyword search failed: {e}", exc_info=True)
            return []

        if not result:
            return []

        return [
            RetrievalResult(
                chunk_id=str(r["chunk_id"]),
                content=r["content"] or "",
                file_path=r["file_path"] or "",
                chunk_index=r["chunk_index"] or 0,
                keyword_score=float(r["keyword_score"]),
            )
            for r in result
        ]

    # ----------------------------------------------------------
    # 混合检索
    # ----------------------------------------------------------

    def hybrid_search(self, query_text: str, query_embedding: List[float],
                      top_k: Optional[int] = None,
                      vector_weight: Optional[float] = None,
                      keyword_weight: Optional[float] = None,
                      file_pattern: Optional[str] = None) -> List[RetrievalResult]:
        """
        混合检索：向量 + 关键词加权融合

        使用加权求和方式融合两种检索结果。
        当某一种检索失败时（返回空列表），仅使用另一种检索的结果。

        Args:
            query_text: 查询文本（用于关键词检索）
            query_embedding: 查询向量（用于向量检索）
            top_k: 返回结果数
            vector_weight: 向量权重（默认 0.7）
            keyword_weight: 关键词权重（默认 0.3）
            file_pattern: 可选的文件路径过滤模式

        Returns:
            融合后的检索结果列表
        """
        k = top_k or self.top_k
        vw = vector_weight if vector_weight is not None else self.vector_weight
        kw = keyword_weight if keyword_weight is not None else self.keyword_weight

        # 分别执行两种检索（取更多结果以充分融合）
        # 注意：vector_search/keyword_search 内部已捕获异常并返回空列表
        vector_results = self.vector_search(query_embedding, top_k=k * 3, file_pattern=file_pattern)
        keyword_results = self.keyword_search(query_text, top_k=k * 3, file_pattern=file_pattern)

        if not vector_results and not keyword_results:
            logger.warning("Hybrid search: both vector and keyword searches returned no results")
            return []

        if not vector_results:
            logger.warning("Hybrid search: vector search failed, falling back to keyword-only results")
            return keyword_results[:k]
        if not keyword_results:
            logger.warning("Hybrid search: keyword search failed, falling back to vector-only results")
            return vector_results[:k]

        # 融合结果
        merged: Dict[str, RetrievalResult] = {}

        for r in vector_results:
            merged[r.chunk_id] = RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                file_path=r.file_path,
                chunk_index=r.chunk_index,
                vector_score=r.vector_score,
                keyword_score=0.0,
                final_score=r.vector_score * vw,
            )

        for r in keyword_results:
            if r.chunk_id in merged:
                merged[r.chunk_id].keyword_score = r.keyword_score
                merged[r.chunk_id].final_score = (
                    merged[r.chunk_id].vector_score * vw +
                    r.keyword_score * kw
                )
            else:
                merged[r.chunk_id] = RetrievalResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    file_path=r.file_path,
                    chunk_index=r.chunk_index,
                    vector_score=0.0,
                    keyword_score=r.keyword_score,
                    final_score=r.keyword_score * kw,
                )

        # 按综合得分排序
        results = sorted(merged.values(), key=lambda x: x.final_score, reverse=True)

        # 分配排名
        for i, r in enumerate(results):
            r.rank = i + 1

        return results[:k]

    # ----------------------------------------------------------
    # RRF 融合（倒数排名融合）
    # ----------------------------------------------------------

    def rrf_search(self, query_text: str, query_embedding: List[float],
                   top_k: Optional[int] = None, k_const: int = 60,
                   file_pattern: Optional[str] = None) -> List[RetrievalResult]:
        """
        RRF (Reciprocal Rank Fusion) 融合检索

        RRF 分数 = Σ(1 / (k + rank_i))
        其中 k 是常数（通常为 60），rank_i 是结果在第 i 个排序中的排名。
        当某一种检索失败时，仅使用另一种检索的结果。

        Args:
            query_text: 查询文本
            query_embedding: 查询向量
            top_k: 返回结果数
            k_const: RRF 常数（默认 60）
            file_pattern: 可选的文件路径过滤模式

        Returns:
            RRF 融合后的结果
        """
        k = top_k or self.top_k

        # 分别检索（取更多结果）
        # 注意：vector_search/keyword_search 内部已捕获异常并返回空列表
        vector_results = self.vector_search(query_embedding, top_k=k * 5, file_pattern=file_pattern)
        keyword_results = self.keyword_search(query_text, top_k=k * 5, file_pattern=file_pattern)

        if not vector_results and not keyword_results:
            logger.warning("RRF search: both vector and keyword searches returned no results")
            return []

        if not vector_results:
            logger.warning("RRF search: vector search failed, falling back to keyword-only results")
            return keyword_results[:k]
        if not keyword_results:
            logger.warning("RRF search: keyword search failed, falling back to vector-only results")
            return vector_results[:k]

        # RRF 分数计算
        rrf_scores: Dict[str, Tuple[float, RetrievalResult]] = {}

        for rank, r in enumerate(vector_results):
            score = 1.0 / (k_const + rank + 1)
            rrf_scores[r.chunk_id] = (
                score,
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    file_path=r.file_path,
                    chunk_index=r.chunk_index,
                    vector_score=r.vector_score,
                    final_score=score,
                )
            )

        for rank, r in enumerate(keyword_results):
            score = 1.0 / (k_const + rank + 1)
            if r.chunk_id in rrf_scores:
                existing_score, existing_r = rrf_scores[r.chunk_id]
                rrf_scores[r.chunk_id] = (
                    existing_score + score,
                    RetrievalResult(
                        chunk_id=r.chunk_id,
                        content=r.content,
                        file_path=r.file_path,
                        chunk_index=r.chunk_index,
                        vector_score=existing_r.vector_score,
                        keyword_score=r.keyword_score,
                        final_score=existing_score + score,
                    )
                )
            else:
                rrf_scores[r.chunk_id] = (
                    score,
                    RetrievalResult(
                        chunk_id=r.chunk_id,
                        content=r.content,
                        file_path=r.file_path,
                        chunk_index=r.chunk_index,
                        keyword_score=r.keyword_score,
                        final_score=score,
                    )
                )

        # 按 RRF 分数排序
        results = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        results = [r for _, r in results]

        for i, r in enumerate(results):
            r.rank = i + 1

        return results[:k]

    # ----------------------------------------------------------
    # 统一检索接口
    # ----------------------------------------------------------

    def search(self, query_text: str, query_embedding: List[float],
               retrieval_type: str = "hybrid", top_k: Optional[int] = None,
               file_pattern: Optional[str] = None,
               log_retrieval: bool = True) -> Tuple[List[RetrievalResult], RetrievalStats]:
        """
        统一检索接口

        内部捕获所有异常，确保不会因检索失败而中断调用方流程。
        当检索执行失败时，返回空结果列表和统计信息。

        Args:
            query_text: 查询文本
            query_embedding: 查询向量
            retrieval_type: 检索类型 (vector_only, keyword_only, hybrid, rrf)
            top_k: 返回结果数
            file_pattern: 可选的文件路径过滤模式
            log_retrieval: 是否记录检索日志

        Returns:
            (检索结果列表, 检索统计) — 异常时返回 ([], stats)
        """
        start_time = time.time()
        k = top_k or self.top_k

        # 执行检索（各子方法内部已捕获异常并返回空列表）
        try:
            if retrieval_type == "vector_only":
                results = self.vector_search(query_embedding, top_k=k, file_pattern=file_pattern)
            elif retrieval_type == "keyword_only":
                results = self.keyword_search(query_text, top_k=k, file_pattern=file_pattern)
            elif retrieval_type == "rrf":
                results = self.rrf_search(query_text, query_embedding, top_k=k, file_pattern=file_pattern)
            else:  # hybrid (default)
                results = self.hybrid_search(query_text, query_embedding, top_k=k, file_pattern=file_pattern)
        except Exception as e:
            logger.error(
                f"Search [{retrieval_type}] failed unexpectedly: {e}",
                exc_info=True
            )
            results = []

        latency_ms = int((time.time() - start_time) * 1000)

        # 记录检索日志
        if log_retrieval:
            try:
                retrieval_id = RetrievalRepository.log_retrieval(
                    project_id=self.project_id,
                    query_text=query_text,
                    query_embedding=query_embedding,
                    top_k=k,
                    retrieval_type=retrieval_type,
                    hybrid_weight=self.vector_weight,
                    latency_ms=latency_ms,
                )
                # 记录结果明细
                result_data = [
                    {
                        "chunk_id": r.chunk_id,
                        "rank": r.rank,
                        "vector_score": r.vector_score,
                        "keyword_score": r.keyword_score,
                        "final_score": r.final_score,
                        "metadata": {"file_path": r.file_path},
                    }
                    for r in results
                ]
                RetrievalRepository.log_results(retrieval_id, result_data)
            except Exception as e:
                logger.warning(f"Failed to log retrieval: {e}")

        stats = RetrievalStats(
            retrieval_type=retrieval_type,
            top_k=k,
            latency_ms=latency_ms,
            total_results=len(results),
            query_text=query_text,
        )

        logger.info(
            f"Search [{retrieval_type}] top-{k}: "
            f"{len(results)} results in {latency_ms}ms"
        )

        return results, stats

    # ----------------------------------------------------------
    # 检索质量评估
    # ----------------------------------------------------------

    @staticmethod
    def evaluate(results: List[RetrievalResult], relevant_chunk_ids: List[str],
                 top_k: int = 5) -> Dict[str, float]:
        """
        评估检索质量

        Args:
            results: 检索结果列表
            relevant_chunk_ids: 相关文档的 chunk_id 列表
            top_k: 评估的 top-k

        Returns:
            评估指标字典
        """
        if not results or not relevant_chunk_ids:
            return {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "precision": 0.0}

        relevant_set = set(relevant_chunk_ids)
        top_results = results[:top_k]

        # Recall@k
        retrieved_relevant = sum(1 for r in top_results if r.chunk_id in relevant_set)
        recall = retrieved_relevant / len(relevant_set) if relevant_set else 0.0

        # Precision@k
        precision = retrieved_relevant / len(top_results) if top_results else 0.0

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, r in enumerate(top_results):
            if r.chunk_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break

        # NDCG@k (Normalized Discounted Cumulative Gain)
        dcg = 0.0
        idcg = 0.0
        for i, r in enumerate(top_results):
            rel = 1.0 if r.chunk_id in relevant_set else 0.0
            if i == 0:
                dcg = rel
                idcg = 1.0
            else:
                dcg += rel / (i + 1)
                idcg += 1.0 / (i + 1)

        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "mrr": round(mrr, 4),
            "ndcg": round(ndcg, 4),
        }


# ============================================================
# 便捷函数
# ============================================================

def create_retriever(project_id: str, model_name: Optional[str] = None) -> HybridRetriever:
    """创建检索器实例"""
    return HybridRetriever(project_id=project_id, model_name=model_name)
