"""
仓库处理分布式锁 — 防止并发处理同一仓库
"""

import json
import logging
import uuid
from typing import Optional

from rag_optimizer.cache.redis_client import redis_client
from rag_optimizer.config.settings import settings

logger = logging.getLogger(__name__)


class RepoLock:
    """
    仓库处理分布式锁

    使用 Redis SET NX + EX 实现分布式锁。
    只有锁的持有者才能释放锁（通过 Lua 脚本保证原子性）。
    """

    def __init__(self):
        self.ttl = settings.redis.repo_lock_ttl

    def _make_key(self, repo_id: str) -> str:
        return f"lock:repo:{repo_id}"

    def acquire(self, repo_id: str, session_id: Optional[str] = None,
                timeout: Optional[int] = None) -> Optional[str]:
        """
        获取锁

        Args:
            repo_id: 仓库 ID
            session_id: 会话 ID（自动生成如果未提供）
            timeout: 锁超时时间（秒）

        Returns:
            成功返回 session_id，失败返回 None
        """
        key = self._make_key(repo_id)
        sid = session_id or str(uuid.uuid4())
        ttl = timeout or self.ttl

        try:
            c = redis_client.client
            if c is None:
                # Redis 不可用，返回模拟锁
                return sid

            acquired = c.set(key, sid, nx=True, ex=ttl)
            if acquired:
                logger.info(f"Lock acquired: repo={repo_id}, session={sid}")
                return sid
            else:
                holder = c.get(key)
                logger.debug(f"Lock held by: {holder}")
                return None
        except Exception as e:
            logger.warning(f"Lock acquire error: {e}")
            return sid  # 降级：允许执行

    def release(self, repo_id: str, session_id: str) -> bool:
        """
        释放锁（只有持有者才能释放）

        Args:
            repo_id: 仓库 ID
            session_id: 会话 ID

        Returns:
            是否成功释放
        """
        key = self._make_key(repo_id)

        try:
            c = redis_client.client
            if c is None:
                return True

            # Lua 脚本：原子性检查并删除
            lua_script = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            end
            return 0
            """
            result = c.eval(lua_script, 1, key, session_id)
            if result:
                logger.info(f"Lock released: repo={repo_id}, session={session_id}")
                return True
            else:
                logger.warning(f"Lock release failed: not holder. repo={repo_id}")
                return False
        except Exception as e:
            logger.warning(f"Lock release error: {e}")
            return True

    def is_locked(self, repo_id: str) -> bool:
        """检查锁是否被持有"""
        key = self._make_key(repo_id)
        return redis_client.exists(key)


# 全局单例
repo_lock = RepoLock()
