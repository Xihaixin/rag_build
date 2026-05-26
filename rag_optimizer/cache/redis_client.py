"""
Redis 连接管理
"""

import json
import logging
import os
import subprocess
import time
import socket
from typing import Any, Optional

from rag_optimizer.config.settings import settings

logger = logging.getLogger(__name__)

# Redis 本地安装路径（Windows）
_REDIS_INSTALL_DIR = r"D:\Redis\Redis-8.4.0-Windows-x64-msys2"
_REDIS_SERVER_EXE = os.path.join(_REDIS_INSTALL_DIR, "redis-server.exe")
_REDIS_CONF = os.path.join(_REDIS_INSTALL_DIR, "redis.conf")


def _start_redis_server() -> bool:
    if not os.path.isfile(_REDIS_SERVER_EXE):
        logger.error("redis-server.exe not found")
        return False

    try:
        args = [_REDIS_SERVER_EXE]
        if os.path.isfile(_REDIS_CONF):
            args.append("redis.conf")

        process = subprocess.Popen(
            args,
            cwd=_REDIS_INSTALL_DIR,
            shell=False,
            stdout=subprocess.PIPE,   # 先别 DEVNULL，便于排错
            stderr=subprocess.PIPE,
            creationflags=(
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
            ),
        )

        # 1. 看是否秒崩
        time.sleep(0.5)

        if process.poll() is not None:
            stderr = process.stderr.read().decode(errors="ignore")
            logger.error(
                f"Redis exited immediately. stderr={stderr}"
            )
            return False

        # 2. 等待端口 ready
        for _ in range(10):
            try:
                with socket.create_connection(
                    ("127.0.0.1", 6379),
                    timeout=1
                ):
                    logger.info("Redis is ready.")
                    return True
            except OSError:
                time.sleep(0.5)

        logger.error("Redis failed to listen on port 6379")
        return False

    except Exception as e:
        logger.error(f"Failed to start Redis: {e}")
        return False


"""
Redis 连接管理
"""

import json
import logging
import threading
import time
from typing import Any, Optional

from rag_optimizer.config.settings import settings
from rag_optimizer.infrastructure.redis_server import _start_redis_server

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 客户端封装（线程安全 + 自动重连 + 自动启动）"""

    def __init__(self):
        self._client = None
        self._config = settings.redis
        self._lock = threading.Lock()

    # =========================
    # 内部连接逻辑
    # =========================
    def _create_client(self):
        """创建 redis client"""
        import redis

        return redis.Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            password=self._config.password,
            decode_responses=self._config.decode_responses,
            socket_connect_timeout=3,
            socket_timeout=3,
            health_check_interval=30,
            retry_on_timeout=True,
        )

    def _is_alive(self) -> bool:
        """检测连接是否可用"""
        if self._client is None:
            return False

        try:
            self._client.ping()
            return True
        except Exception:
            return False

    # =========================
    # 连接管理
    # =========================
    def connect(self):
        """建立 Redis 连接"""

        if self._is_alive():
            return

        with self._lock:
            # double-check
            if self._is_alive():
                return

            try:
                self._client = self._create_client()
                self._client.ping()

                logger.info(
                    "Redis connected: %s:%s/%s",
                    self._config.host,
                    self._config.port,
                    self._config.db,
                )
                return

            except ImportError:
                logger.warning("redis package not installed.")
                self._client = None
                return

            except Exception as e:
                logger.warning(
                    "Redis connection failed: %s, try auto-start...",
                    e,
                )
                self._client = None

            # =========================
            # 自动启动 Redis
            # =========================
            if not _start_redis_server():
                logger.error("Redis auto-start failed.")
                return

            # 等待 ready
            for i in range(5):
                time.sleep(1)

                try:
                    self._client = self._create_client()
                    self._client.ping()

                    logger.info(
                        "Redis connected after auto-start "
                        "(attempt=%s)",
                        i + 1,
                    )
                    return

                except Exception:
                    logger.debug(
                        "Waiting Redis ready (%s/5)...",
                        i + 1,
                    )

            logger.error("Redis failed to become ready.")
            self._client = None

    def reconnect(self):
        """强制重连"""
        self.close()
        self.connect()

    def close(self):
        """关闭连接"""
        with self._lock:
            if self._client:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    # =========================
    # 属性
    # =========================
    @property
    def client(self):
        """获取 client（自动重连）"""
        if not self._is_alive():
            self.connect()
        return self._client

    @property
    def is_connected(self) -> bool:
        return self._is_alive()

    # =========================
    # KV 操作
    # =========================
    def get(self, key: str, default=None):
        """获取值（自动 JSON 反序列化）"""
        try:
            c = self.client
            if not c:
                return default

            value = c.get(key)
            if value is None:
                return default

            try:
                return json.loads(value)
            except Exception:
                return value

        except Exception as e:
            logger.debug("Redis get error: %s", e)
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """设置值（自动 JSON 序列化）"""
        try:
            c = self.client
            if not c:
                return False

            if not isinstance(value, str):
                value = json.dumps(
                    value,
                    ensure_ascii=False,
                )

            if ttl:
                c.setex(key, ttl, value)
            else:
                c.set(key, value)

            return True

        except Exception as e:
            logger.debug("Redis set error: %s", e)
            return False

    def delete(self, key: str) -> bool:
        """删除 key"""
        try:
            c = self.client
            return bool(c.delete(key)) if c else False
        except Exception as e:
            logger.debug("Redis delete error: %s", e)
            return False

    def exists(self, key: str) -> bool:
        """key 是否存在"""
        try:
            c = self.client
            return bool(c.exists(key)) if c else False
        except Exception:
            return False

    def ttl(self, key: str) -> int:
        """获取 TTL"""
        try:
            c = self.client
            return c.ttl(key) if c else -2
        except Exception:
            return -2

    # =========================
    # 计数器
    # =========================
    def incr(self, key: str, amount: int = 1) -> Optional[int]:
        try:
            c = self.client
            return c.incr(key, amount) if c else None
        except Exception:
            return None

    def decr(self, key: str, amount: int = 1) -> Optional[int]:
        try:
            c = self.client
            return c.decr(key, amount) if c else None
        except Exception:
            return None

    # =========================
    # scan（替代 keys）
    # =========================
    def scan(
        self,
        pattern: str = "*",
        count: int = 100,
    ) -> list[str]:
        """非阻塞扫描 key"""
        try:
            c = self.client
            if not c:
                return []

            cursor = 0
            keys = []

            while True:
                cursor, batch = c.scan(
                    cursor=cursor,
                    match=pattern,
                    count=count,
                )
                keys.extend(batch)

                if cursor == 0:
                    break

            return keys

        except Exception:
            return []


# 全局单例
redis_client = RedisClient()
