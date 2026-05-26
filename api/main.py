"""
OpenWiki-open 重构版 — 应用入口点

使用 PostgreSQL + pgvector 后端，替代原始 LocalDB + .pkl 存储。
启动 FastAPI 应用服务器，配置日志、数据库连接池等。

Usage:
    python -m api.main
    # 或
    uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 配置日志（必须在导入其他模块之前）
from core.config.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

# ============================================================
# 环境变量加载
# ============================================================

try:
    from dotenv import load_dotenv

    env_path = _project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.info("No .env file found, using system environment variables")
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env loading")

# ============================================================
# 应用生命周期管理
# ============================================================


@asynccontextmanager
async def lifespan(app):
    """
    应用生命周期管理

    - startup: 初始化数据库连接池、配置 Google Generative AI 等
    - shutdown: 清理资源
    """
    # --- Startup ---
    logger.info("Starting OpenWiki API (RAG Optimized)...")

    # 初始化异步数据库连接池
    _db_pool = None
    try:
        from rag_optimizer.db.connection import AsyncDatabasePool

        _db_pool = AsyncDatabasePool()
        await _db_pool.init_pool()
        app.state.db_pool = _db_pool
        logger.info("Async database pool initialized")
    except Exception as e:
        logger.warning(f"Could not initialize async database pool: {e}")
        logger.warning("Database-dependent features will be unavailable")
        app.state.db_pool = None

    # 配置 Google Generative AI（如果可用）
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=google_api_key)
            logger.info("Google Generative AI configured")
        except ImportError:
            logger.warning("google.generativeai not installed, Google provider unavailable")
        except Exception as e:
            logger.warning(f"Failed to configure Google Generative AI: {e}")
    else:
        logger.info("GOOGLE_API_KEY not set, Google provider will be unavailable")

    yield

    # --- Shutdown ---
    logger.info("Shutting down OpenWiki API...")
    if _db_pool is not None:
        try:
            await _db_pool.close_pool()
            logger.info("Database connection pool closed")
        except Exception as e:
            logger.warning(f"Error closing database pool: {e}")


# ============================================================
# FastAPI 应用
# ============================================================

# 导入 api.api 模块（这会创建 FastAPI app 实例并注册所有路由）
# 然后我们重新创建 app 以应用 lifespan
from api.api import app as _original_app

# 创建新的 app 实例，应用 lifespan
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="OpenWiki API (RAG Optimized)",
    description="基于 PostgreSQL + pgvector 的 OpenWiki-open 重构版 API",
    version="2.0.0",
    lifespan=lifespan,
)

# 复制 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 复制所有路由
for route in _original_app.routes:
    app.routes.append(route)

# 复制异常处理器
for exc_class_or_status_code, handler in _original_app.exception_handlers.items():
    app.exception_handlers[exc_class_or_status_code] = handler

# 复制中间件
app.user_middleware = _original_app.user_middleware
app.middleware_stack = _original_app.middleware_stack

logger.info(f"App initialized with {len(app.routes)} routes")


# ============================================================
# 入口点
# ============================================================

def main():
    """启动应用服务器"""
    import uvicorn

    host = os.environ.get("OpenWiki_HOST", "0.0.0.0")
    port = int(os.environ.get("OpenWiki_PORT", "8001"))
    reload = os.environ.get("OpenWiki_RELOAD", "true").lower() == "true"

    logger.info(f"Starting server on {host}:{port} (reload={reload})")
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
