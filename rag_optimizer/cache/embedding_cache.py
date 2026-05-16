"""
Embedding 缓存 — 内容哈希缓存

避免对相同文本重复调用 Embedding API，节省 API 调用费用。
"""

import hashlib
import json
import logging
from typing import List, Optional

from rag_optimizer.cache.redis_client import redis_client
from rag_optimizer.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Embedding 缓存

    使用 SHA-256 内容哈希作为缓存键，TTL 24 小时。
    """

    def __init__(self):
        self.ttl = settings.redis.embedding_cache_ttl

    def _make_key(self, model: str, content: str) -> str:
        """生成缓存键"""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"embed:{model}:{content_hash}"

    def get(self, model: str, content: str) -> Optional[List[float]]:
        """获取缓存的向量"""
        key = self._make_key(model, content)
        cached = redis_client.get(key)
        if cached:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def set(self, model: str, content: str, embedding: List[float]):
        """缓存向量"""
        key = self._make_key(model, content)
        redis_client.set(key, json.dumps(embedding), ttl=self.ttl)

    def get_or_compute(self, model: str, content: str,
                       compute_fn) -> List[float]:
        """
        获取缓存或计算新向量

        Args:
            model: 模型名称
            content: 文本内容
            compute_fn: 计算向量的函数

        Returns:
            向量
        """
        cached = self.get(model, content)
        if cached is not None:
            logger.debug(f"Embedding cache HIT: {model}")
            return cached

        logger.debug(f"Embedding cache MISS: {model}")
        embedding = compute_fn(content)
        self.set(model, content, embedding)
        return embedding

    def get_or_compute_batch(self, model: str, texts: List[str],
                              compute_fn) -> List[List[float]]:
        """
        批量获取缓存或计算新向量

        Args:
            model: 模型名称
            texts: 文本列表
            compute_fn: 计算向量的函数（接收文本列表）

        Returns:
            向量列表
        """
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = self.get(model, text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)  # placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            logger.info(f"Computing {len(uncached_texts)} uncached embeddings...")
            new_embeddings = compute_fn(uncached_texts)
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding
                self.set(model, texts[idx], embedding)

        return [r for r in results if r is not None]


# 全局单例
embedding_cache = EmbeddingCache()
