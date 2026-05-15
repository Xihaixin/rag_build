"""
rag_optimizer — 基于 PostgreSQL + pgvector 的透明化 RAG 系统优化

毕业设计项目：将 deepwiki-open 的 RAG 系统从 FAISS + pickle 升级为
PostgreSQL + pgvector，实现数据透明化、混合检索和异步处理。

核心模块：
- db:        数据库连接、ORM 模型、数据访问层
- retrieval: 混合检索引擎（向量 + 全文检索）
- pipeline:  数据摄取管道（读取、分块、嵌入）
- cache:     Redis 缓存层（Embedding 缓存、语义缓存）
- migration: 数据迁移工具（从 pkl 导入 PostgreSQL）
"""

__version__ = "2.0.0"
