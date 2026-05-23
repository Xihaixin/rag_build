"""
毕业设计：透明化 RAG 系统优化
配置文件 — 集中管理所有配置项
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # rag_build/
RAG_OPTIMIZER_DIR = PROJECT_ROOT / "rag_optimizer"


# ============================================================
# 数据库配置
# ============================================================
@dataclass
class PostgreSQLConfig:
    """PostgreSQL + pgvector 连接配置"""
    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    database: str = os.getenv("PGDATABASE", "rag_optimizer")
    user: str = os.getenv("PGUSER", "postgres")
    password: str = os.getenv("PGPASSWORD", "postgres")
    min_connections: int = 2
    max_connections: int = 10
    command_timeout: int = 60  # seconds

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# ============================================================
# Redis 配置
# ============================================================
@dataclass
class RedisConfig:
    """Redis 缓存配置"""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = 0
    password: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    decode_responses: bool = True

    # 缓存 TTL 配置（秒）
    embedding_cache_ttl: int = 86400       # 24h: Embedding 缓存
    semantic_cache_ttl: int = 3600         # 1h: 语义缓存
    repo_lock_ttl: int = 300               # 5min: 仓库处理锁
    progress_ttl: int = 600                # 10min: 进度缓存

    @property
    def dsn(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


# ============================================================
# 嵌入模型配置
# ============================================================
@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    # 默认嵌入模型
    default_model: str = "text-embedding-v4"
    default_provider: str = "dashscope"
    default_dimensions: int = 256

    # DashScope 配置
    dashscope_api_key: Optional[str] = os.getenv("DASHSCOPE_API_KEY", None)
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 批处理配置
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0


# ============================================================
# LLM 配置
# ============================================================
@dataclass
class LLMConfig:
    """大语言模型配置"""
    default_model: str = "qwen-plus"
    default_provider: str = "dashscope"
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 0.9


# ============================================================
# 检索配置
# ============================================================
@dataclass
class RetrievalConfig:
    """检索配置"""
    # 默认检索参数
    default_top_k: int = 5
    default_retrieval_type: str = "hybrid"  # vector_only, hybrid, keyword_only

    # 混合检索权重
    hybrid_vector_weight: float = 0.7   # 语义搜索权重
    hybrid_keyword_weight: float = 0.3  # 关键词搜索权重

    # 语义缓存相似度阈值
    semantic_cache_threshold: float = 0.95


# ============================================================
# 分块配置
# ============================================================
@dataclass
class ChunkConfig:
    """文本分块配置"""
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    default_split_by: str = "word"  # word, sentence, code


# ============================================================
# 存储后端配置
# ============================================================
@dataclass
class StorageConfig:
    """存储后端配置"""
    backend: str = os.getenv("STORAGE_BACKEND", "pgvector")  # pgvector 或 faiss
    faiss_index_path: str = str(PROJECT_ROOT / "vector_stores")


# ============================================================
# 全局配置单例
# ============================================================
@dataclass
class Settings:
    """全局配置"""
    postgresql: PostgreSQLConfig = field(default_factory=PostgreSQLConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # 日志配置
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = str(PROJECT_ROOT / "config" / "logs" / "rag_optimizer.log")


# 全局单例
settings = Settings()
