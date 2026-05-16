"""
语义缓存 — 相似问题直接返回缓存答案

当用户提出的问题与之前的问题语义相似时（余弦相似度 > 阈值），
直接返回缓存的答案，避免重复调用 LLM。
"""

import hashlib
import json
import logging
from typing import List, Optional, Tuple

from rag_optimizer.cache.redis_client import redis_client
from rag_optimizer.config.settings import settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    语义缓存

    存储 (query, answer, query_vector) 三元组。
    新查询到来时，在缓存中搜索语义相似的历史查询。
    """

    def __init__(self):
        self.ttl = settings.redis.semantic_cache_ttl
        self.threshold = settings.retrieval.semantic_cache_threshold

    def _make_key(self, repo_id: str, query_hash: str) -> str:
        return f"semantic:{repo_id}:{query_hash}"

    def _make_pattern(self, repo_id: str) -> str:
        return f"semantic:{repo_id}:*"

    def _compute_hash(self, query: str) -> str:
        return hashlib.md5(query.encode("utf-8")).hexdigest()[:16]

    def get_exact(self, repo_id: str, query: str) -> Optional[str]:
        """精确匹配缓存"""
        query_hash = self._compute_hash(query)
        key = self._make_key(repo_id, query_hash)
        cached = redis_client.get(key)
        if cached:
            try:
                data = json.loads(cached)
                return data.get("answer")
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def set_exact(self, repo_id: str, query: str, answer: str,
                  query_vector: Optional[List[float]] = None):
        """精确匹配缓存"""
        query_hash = self._compute_hash(query)
        key = self._make_key(repo_id, query_hash)
        data = {
            "query": query,
            "answer": answer,
            "vector": query_vector,
        }
        redis_client.set(key, json.dumps(data), ttl=self.ttl)

    def search_similar(self, repo_id: str, query_vector: List[float],
                       threshold: Optional[float] = None) -> Optional[Tuple[str, str, float]]:
        """
        在缓存中搜索语义相似的查询

        Args:
            repo_id: 仓库 ID
            query_vector: 查询向量
            threshold: 相似度阈值

        Returns:
            (query, answer, similarity) 或 None
        """
        thr = threshold if threshold is not None else self.threshold
        pattern = self._make_pattern(repo_id)
        keys = redis_client.keys(pattern)

        if not keys:
            return None

        best_match = None
        best_sim = 0.0

        for key in keys:
            cached = redis_client.get(key)
            if not cached:
                continue

            try:
                data = json.loads(cached)
                cached_vector = data.get("vector")
                if not cached_vector:
                    continue

                similarity = self._cosine_similarity(query_vector, cached_vector)
                if similarity > best_sim:
                    best_sim = similarity
                    best_match = (data["query"], data["answer"], similarity)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        if best_match and best_sim >= thr:
            logger.info(f"Semantic cache HIT: similarity={best_sim:.4f}")
            return best_match

        return None

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """计算余弦相似度"""
        import math
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# 全局单例
semantic_cache = SemanticCache()
