"""
Redis 连接管理
"""

import json
import logging
from typing import Any, Optional

from rag_optimizer.config.settings import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 客户端封装"""

    def __init__(self):
        self._client = None
        self._config = settings.redis

    def connect(self):
        """建立 Redis 连接"""
        if self._client is not None:
            try:
                self._client.ping()
                return
            except Exception:
                self._client = None

        try:
            import redis
            self._client = redis.Redis(
                host=self._config.host,
                port=self._config.port,
                db=self._config.db,
                password=self._config.password,
                decode_responses=self._config.decode_responses,
            )
            self._client.ping()
            logger.info(f"Redis connected: {self._config.host}:{self._config.port}/{self._config.db}")
        except ImportError:
            logger.warning("redis package not installed. Cache disabled.")
            self._client = None
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Cache disabled.")
            self._client = None

    def close(self):
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self):
        """获取 Redis 客户端（自动连接）"""
        if self._client is None:
            self.connect()
        return self._client

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        if self._client is None:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    # 便捷方法
    def get(self, key: str) -> Optional[str]:
        """获取值"""
        try:
            c = self.client
            return c.get(key) if c else None
        except Exception as e:
            logger.debug(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        """设置值"""
        try:
            c = self.client
            if c:
                if ttl:
                    c.setex(key, ttl, value)
                else:
                    c.set(key, value)
        except Exception as e:
            logger.debug(f"Redis set error: {e}")

    def delete(self, key: str):
        """删除键"""
        try:
            c = self.client
            if c:
                c.delete(key)
        except Exception as e:
            logger.debug(f"Redis delete error: {e}")

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            c = self.client
            return bool(c.exists(key)) if c else False
        except Exception:
            return False

    def keys(self, pattern: str) -> list:
        """匹配键"""
        try:
            c = self.client
            return c.keys(pattern) if c else []
        except Exception:
            return []


# 全局单例
redis_client = RedisClient()
