"""
配置管理模块 — 适配 OPENWIKI-open 的配置系统到 rag_optimizer 配置

提供与原始 deepwiki-open 兼容的配置接口，但底层使用 rag_optimizer 的配置系统。
支持从 JSON 配置文件和环境变量加载配置。
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rag_optimizer.config.settings import settings

logger = logging.getLogger(__name__)

# ============================================================
# 环境变量
# ============================================================

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or settings.embedding.dashscope_api_key

# AWS
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.environ.get("AWS_SESSION_TOKEN")
AWS_REGION = os.environ.get("AWS_REGION")
AWS_ROLE_ARN = os.environ.get("AWS_ROLE_ARN")

# Wiki authentication
raw_auth_mode = os.environ.get("OPENWIKI_AUTH_MODE", "False")
WIKI_AUTH_MODE = raw_auth_mode.lower() in ("true", "1", "t")
WIKI_AUTH_CODE = os.environ.get("OPENWIKI_AUTH_CODE", "")

# Embedder type
EMBEDDER_TYPE = os.environ.get("OPENWIKI_EMBEDDER_TYPE", "openai").lower()

# Configuration directory
CONFIG_DIR = os.environ.get("OPENWIKI_CONFIG_DIR", None)

# ============================================================
# 配置加载
# ============================================================

# 默认配置目录
_DEFAULT_CONFIG_DIR = Path(__file__).parent / "config"


def replace_env_placeholders(config: Union[Dict[str, Any], List[Any], str, Any]) -> Union[Dict[str, Any], List[Any], str, Any]:
    """
    递归替换配置中的环境变量占位符 ${VAR_NAME}
    """
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def replacer(match: re.Match[str]) -> str:
        env_var_name = match.group(1)
        original_placeholder = match.group(0)
        env_var_value = os.environ.get(env_var_name)
        if env_var_value is None:
            logger.warning(
                f"Environment variable placeholder '{original_placeholder}' was not found. "
                f"The placeholder string will be used as is."
            )
            return original_placeholder
        return env_var_value

    if isinstance(config, dict):
        return {k: replace_env_placeholders(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_placeholders(item) for item in config]
    elif isinstance(config, str):
        return pattern.sub(replacer, config)
    else:
        return config


def load_json_config(filename: str) -> dict:
    """加载 JSON 配置文件"""
    try:
        if CONFIG_DIR:
            config_path = Path(CONFIG_DIR) / filename
        else:
            config_path = _DEFAULT_CONFIG_DIR / filename

        logger.info(f"Loading configuration from {config_path}")

        if not config_path.exists():
            logger.warning(f"Configuration file {config_path} does not exist")
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            config = replace_env_placeholders(config)
            return config
    except Exception as e:
        logger.error(f"Error loading configuration file {filename}: {str(e)}")
        return {}


def load_generator_config() -> dict:
    """加载生成器模型配置"""
    generator_config = load_json_config("generator.json")

    # 从 rag_optimizer 设置中补充默认值
    if not generator_config.get("providers"):
        generator_config.setdefault("providers", {})

    # 确保 dashscope 提供者存在（rag_optimizer 默认使用 dashscope）
    if "dashscope" not in generator_config.get("providers", {}):
        generator_config.setdefault("providers", {})
        generator_config["providers"]["dashscope"] = {
            "default_model": settings.llm.default_model or "qwen-plus",
            "supportsCustomModel": True,
            "models": {
                settings.llm.default_model or "qwen-plus": {
                    "temperature": settings.llm.temperature,
                    "top_p": settings.llm.top_p,
                }
            },
        }

    if not generator_config.get("default_provider"):
        generator_config["default_provider"] = settings.llm.default_provider or "dashscope"

    return generator_config


def load_embedder_config() -> dict:
    """加载嵌入器配置"""
    embedder_config = load_json_config("embedder.json")

    # 从 rag_optimizer 设置中补充默认值
    if not embedder_config.get("embedder"):
        embedder_config["embedder"] = {
            "batch_size": settings.embedding.batch_size,
            "model_kwargs": {
                "model": settings.embedding.default_model,
                "dimensions": settings.embedding.default_dimensions,
            },
        }

    if not embedder_config.get("text_splitter"):
        embedder_config["text_splitter"] = {
            "split_by": settings.chunk.default_split_by,
            "chunk_size": settings.chunk.default_chunk_size,
            "chunk_overlap": settings.chunk.default_chunk_overlap,
        }

    if not embedder_config.get("retriever"):
        embedder_config["retriever"] = {
            "top_k": settings.retrieval.default_top_k,
        }

    return embedder_config


def load_lang_config() -> dict:
    """加载语言配置"""
    default_config = {
        "supported_languages": {
            "en": "English",
            "ja": "Japanese (日本語)",
            "zh": "Mandarin Chinese (中文)",
            "zh-tw": "Traditional Chinese (繁體中文)",
            "es": "Spanish (Español)",
            "kr": "Korean (한국어)",
            "vi": "Vietnamese (Tiếng Việt)",
            "pt-br": "Brazilian Portuguese (Português Brasileiro)",
            "fr": "Français (French)",
            "ru": "Русский (Russian)",
        },
        "default": "en",
    }

    loaded_config = load_json_config("lang.json")
    if not loaded_config:
        return default_config

    if "supported_languages" not in loaded_config or "default" not in loaded_config:
        logger.warning("Language configuration file 'lang.json' is malformed. Using default.")
        return default_config

    return loaded_config


def load_repo_config() -> dict:
    """加载仓库配置"""
    return load_json_config("repo.json")


# ============================================================
# 默认排除目录和文件
# ============================================================

DEFAULT_EXCLUDED_DIRS: List[str] = [
    "./.venv/", "./venv/", "./env/", "./virtualenv/",
    "./node_modules/", "./bower_components/", "./jspm_packages/",
    "./.git/", "./.svn/", "./.hg/", "./.bzr/",
    "./__pycache__/", "./.pytest_cache/", "./.mypy_cache/", "./.ruff_cache/", "./.coverage/",
    "./dist/", "./build/", "./out/", "./target/", "./bin/", "./obj/",
    "./docs/", "./_docs/", "./site-docs/", "./_site/",
    "./.idea/", "./.vscode/", "./.vs/", "./.eclipse/", "./.settings/",
    "./logs/", "./log/", "./tmp/", "./temp/",
]

DEFAULT_EXCLUDED_FILES: List[str] = [
    "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock",
    "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
    ".lock", ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk", ".env",
    ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv", ".gitignore",
    ".gitattributes", ".gitmodules", ".github", ".gitlab-ci.yml",
    ".prettierrc", ".eslintrc", ".eslintignore", ".stylelintrc",
    ".editorconfig", ".jshintrc", ".pylintrc", ".flake8", "mypy.ini",
    "pyproject.toml", "tsconfig.json", "webpack.config.js", "babel.config.js",
    "rollup.config.js", "jest.config.js", "karma.conf.js", "vite.config.js",
    "next.config.js", "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css",
    "*.map", "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z", "*.iso",
    "*.dmg", "*.img", "*.msix", "*.appx", "*.appxbundle", "*.xap", "*.ipa",
    "*.deb", "*.rpm", "*.msi", "*.exe", "*.dll", "*.so", "*.dylib", "*.o",
    "*.obj", "*.jar", "*.war", "*.ear", "*.jsm", "*.class", "*.pyc", "*.pyd",
    "*.pyo", "__pycache__", "*.a", "*.lib", "*.lo", "*.la", "*.slo", "*.dSYM",
    "*.egg", "*.egg-info", "*.dist-info", "*.eggs", "node_modules",
    "bower_components", "jspm_packages", "lib-cov", "coverage", "htmlcov",
    ".nyc_output", ".tox", "dist", "build", "bld", "out", "bin", "target",
    "packages/*/dist", "packages/*/build", ".output",
]

# ============================================================
# 全局配置
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


# ============================================================
# 辅助函数
# ============================================================

def get_embedder_config() -> dict:
    """获取当前嵌入器配置"""
    embedder_type = EMBEDDER_TYPE
    if embedder_type == "bedrock" and "embedder_bedrock" in configs:
        return configs.get("embedder_bedrock", {})
    elif embedder_type == "google" and "embedder_google" in configs:
        return configs.get("embedder_google", {})
    elif embedder_type == "ollama" and "embedder_ollama" in configs:
        return configs.get("embedder_ollama", {})
    else:
        return configs.get("embedder", {})


def get_embedder_type() -> str:
    """获取嵌入器类型"""
    return EMBEDDER_TYPE


def get_model_config(provider: str = "google", model: Optional[str] = None) -> dict:
    """
    获取指定提供者和模型的配置

    Returns:
        dict: 包含 model_client, model_kwargs 等配置
    """
    if "providers" not in configs:
        raise ValueError("Provider configuration not loaded")

    provider_config = configs["providers"].get(provider)
    if not provider_config:
        raise ValueError(f"Configuration for provider '{provider}' not found")

    # 如果未指定模型，使用默认模型
    if not model:
        model = provider_config.get("default_model")
        if not model:
            raise ValueError(f"No default model specified for provider '{provider}'")

    # 获取模型参数
    model_params = {}
    if model in provider_config.get("models", {}):
        model_params = provider_config["models"][model]
    else:
        default_model = provider_config.get("default_model")
        if default_model and default_model in provider_config.get("models", {}):
            model_params = provider_config["models"][default_model]

    # 构建配置
    result: Dict[str, Any] = {}

    # Provider-specific adjustments
    if provider == "ollama":
        if "options" in model_params:
            result["model_kwargs"] = {"model": model, **model_params["options"]}
        else:
            result["model_kwargs"] = {"model": model}
    else:
        result["model_kwargs"] = {"model": model, **model_params}

    return result
