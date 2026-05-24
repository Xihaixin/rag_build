"""
core — 核心业务模块

将 scripts/simulate/ 下的三个业务流（Wiki/Chat/Research）和数据摄取管道
提升为项目级核心模块，实现与原始 deepwiki-open API 层的解耦。

子模块:
    flows/       — 业务流（WikiGenerationFlow, SimpleChatFlow, DeepResearchFlow）
    ingestion/   — 数据摄取管道（DataIngestor）
    models/      — 数据模型（dataclass）
    prompts/     — Prompt 构建
    utils/       — 工具函数（SSE 解析、LLM 调用、仓库 URL 解析）
"""
