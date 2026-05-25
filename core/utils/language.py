"""
language.py — 语言工具函数
===========================

从 api/simple_chat.py 和 api/websocket_wiki.py 中提取的公共语言工具。
供 core/flows/ 使用，不再依赖 api/ 模块。
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_language_name(language_code: str) -> str:
    """
    获取语言代码对应的语言名称。

    从 core.config 加载语言配置，返回语言名称。
    如果未找到，返回 "English" 作为默认值。

    Args:
        language_code: 语言代码（如 "zh", "en", "ja"）

    Returns:
        语言名称（如 "中文", "English", "日本語"）
    """
    try:
        from core.config import load_lang_config
        lang_config = load_lang_config()
        supported = lang_config.get("supported_languages", {})
        return supported.get(language_code, "English")
    except Exception as e:
        logger.warning(f"加载语言配置失败: {e}")
        return "English"


def get_supported_languages() -> Dict[str, str]:
    """
    获取所有支持的语言映射。

    Returns:
        Dict[str, str] — 语言代码 → 语言名称 的映射
    """
    try:
        from core.config import load_lang_config
        lang_config = load_lang_config()
        return lang_config.get("supported_languages", {"en": "English"})
    except Exception as e:
        logger.warning(f"加载语言配置失败: {e}")
        return {"en": "English"}


def validate_language(language_code: str, default: str = "en") -> str:
    """
    验证语言代码是否受支持，不受支持时返回默认值。

    Args:
        language_code: 语言代码
        default: 默认语言代码

    Returns:
        有效的语言代码
    """
    supported = get_supported_languages()
    if language_code not in supported:
        return default
    return language_code
