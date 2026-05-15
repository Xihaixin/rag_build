"""
数据库连接管理 — 异步连接池（asyncpg）+ 同步连接（psycopg2）

支持两种模式：
1. 异步模式：使用 asyncpg，适用于 FastAPI 等异步框架
2. 同步模式：使用 psycopg2，适用于脚本和迁移工具
"""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator, Optional

from rag_optimizer.config.settings import settings

logger = logging.getLogger(__name__)


# ============================================================
# 异步连接池（asyncpg）
# ============================================================

class AsyncDatabasePool:
    """异步 PostgreSQL 连接池（基于 asyncpg）"""

    def __init__(self):
        self._pool: Any = None  # asyncpg.Pool
        self._config = settings.postgresql

    async def init_pool(self):
        """初始化连接池"""
        if self._pool is not None:
            return

        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                host=self._config.host,
                port=self._config.port,
                database=self._config.database,
                user=self._config.user,
                password=self._config.password,
                min_size=self._config.min_connections,
                max_size=self._config.max_connections,
                command_timeout=self._config.command_timeout,
            )
            logger.info(
                f"Async PostgreSQL pool initialized: "
                f"{self._config.host}:{self._config.port}/{self._config.database}"
            )
        except ImportError:
            logger.error("asyncpg is not installed. Run: pip install asyncpg")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize async PostgreSQL pool: {e}")
            raise

    async def close_pool(self):
        """关闭连接池"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Async PostgreSQL pool closed")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Any, None]:
        """获取异步连接（上下文管理器）"""
        if self._pool is None:
            await self.init_pool()
        async with self._pool.acquire() as conn:
            yield conn

    @property
    def is_initialized(self) -> bool:
        return self._pool is not None


# ============================================================
# 同步连接（psycopg2）
# ============================================================

class SyncDatabaseConnection:
    """同步 PostgreSQL 连接（基于 psycopg2）"""

    def __init__(self):
        self._config = settings.postgresql
        self._conn: Any = None  # psycopg2 connection

    def connect(self):
        """建立同步连接"""
        if self._conn is not None and not self._conn.closed:
            return

        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            self._conn = psycopg2.connect(
                host=self._config.host,
                port=self._config.port,
                database=self._config.database,
                user=self._config.user,
                password=self._config.password,
            )
            self._real_dict_cursor = RealDictCursor
            logger.info(
                f"Sync PostgreSQL connection established: "
                f"{self._config.host}:{self._config.port}/{self._config.database}"
            )
        except ImportError:
            logger.error("psycopg2 is not installed. Run: pip install psycopg2-binary")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def close(self):
        """关闭同步连接"""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None
            logger.info("Sync PostgreSQL connection closed")

    @contextmanager
    def get_cursor(self, dict_cursor: bool = True) -> Generator[Any, None, None]:
        """获取游标（上下文管理器，自动提交/回滚）"""
        self.connect()
        from psycopg2.extras import RealDictCursor
        cursor = self._conn.cursor(
            cursor_factory=RealDictCursor if dict_cursor else None
        )
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def execute(self, query: str, params: Optional[tuple] = None, dict_cursor: bool = True) -> Optional[list]:
        """执行 SQL 并返回结果"""
        with self.get_cursor(dict_cursor=dict_cursor) as cur:
            cur.execute(query, params)
            if cur.description:  # 有返回结果（SELECT）
                return cur.fetchall()
            return None

    def execute_many(self, query: str, params_list: list):
        """批量执行 SQL"""
        self.connect()
        cursor = self._conn.cursor()
        try:
            cursor.executemany(query, params_list)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()


# ============================================================
# 全局单例
# ============================================================

# 异步连接池（用于 FastAPI 等异步场景）
async_pool = AsyncDatabasePool()

# 同步连接（用于脚本和迁移）
sync_conn = SyncDatabaseConnection()


# ============================================================
# 便捷函数
# ============================================================

async def async_execute(query: str, *args):
    """异步执行 SQL"""
    async with async_pool.get_connection() as conn:
        return await conn.fetch(query, *args)


def sync_execute(query: str, params: Optional[tuple] = None):
    """同步执行 SQL"""
    return sync_conn.execute(query, params)
