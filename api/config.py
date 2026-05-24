"""
api.config — 向后兼容配置入口（委托给 core.config）
=======================================================

本文件是重构后的向后兼容入口，所有核心配置逻辑已迁移到:
  core/config/__init__.py

保留此文件以确保现有导入不受影响。
"""

import logging
from typing import Any, Dict, List, Optional

from core.config import (
    # 环境变量
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    OPENROUTER_API_KEY,
    DASHSCOPE_API_KEY,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    AWS_REGION,
    AWS_ROLE_ARN,
    WIKI_AUTH_MODE,
    WIKI_AUTH_CODE,
    EMBEDDER_TYPE,
    CONFIG_DIR,

    # 配置加载函数
    replace_env_placeholders,
    load_json_config,
    load_generator_config,
    load_embedder_config,
    load_lang_config,
    load_repo_config,
    load_configs,

    # 默认排除列表
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,

    # 辅助函数
    get_embedder_config,
    get_embedder_type,
    get_model_config,
)

logger = logging.getLogger(__name__)

# ============================================================
# 全局配置缓存（保持向后兼容）
# ============================================================

configs: Dict[str, Any] = {}

# 加载所有配置
generator_config = load_generator_config()
embedder_config = load_embedder_config()
repo_config = load_repo_config()
lang_config = load_lang_config()

# 更新全局配置
if generator_config:
    configs["default_provider"] = generator_config.get("default_provider", "dashscope")
    configs["providers"] = generator_config.get("providers", {})

if embedder_config:
    for key in ["embedder", "embedder_ollama", "embedder_google", "embedder_bedrock", "retriever", "text_splitter"]:
        if key in embedder_config:
            configs[key] = embedder_config[key]

if repo_config:
    for key in ["file_filters", "repository"]:
        if key in repo_config:
            configs[key] = repo_config[key]

if lang_config:
    configs["lang_config"] = lang_config
