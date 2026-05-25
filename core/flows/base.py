"""
base.py — 业务流公共基类
=========================

所有业务流（Wiki 生成、Q&A 聊天、深度研究）的公共基类，
封装了以下公共逻辑：

  1. 仓库信息解析（parse_repo_url）
  2. 配置加载（load_configs）
  3. LLM 调用封装（call_llm_and_collect, parse_sse_chunk）
  4. RAG 检索器初始化（_init_retriever）
  5. 项目查找与 project_id 解析

依赖:
  - core.config — 统一配置加载
  - core.utils.llm — call_llm_stream
  - core.utils.sse — parse_sse_chunk, call_llm_and_collect
  - core.utils.language — get_language_name
  - rag_optimizer.integration.deepwiki_adapter — PgvectorRetriever
  - rag_optimizer.db.repository — ProjectRepository
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from core.config import (
    load_generator_config, load_embedder_config, load_lang_config,
)
from core.utils.llm import call_llm_stream
from core.utils.sse import parse_sse_chunk, call_llm_and_collect
from core.utils.language import get_language_name as _get_language_name
from rag_optimizer.integration.deepwiki_adapter import PgvectorRetriever
from rag_optimizer.db.repository import ProjectRepository

logger = logging.getLogger("core.flows.base")


# ══════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════


def parse_repo_url(repo_url: str) -> Dict[str, str]:
    """
    解析仓库 URL，提取 owner, repo, repo_type。

    参数:
        repo_url: 仓库 URL（如 https://github.com/user/repo）

    返回:
        {"owner": str, "repo": str, "repo_type": str}
    """
    parsed = urlparse(repo_url)
    path_parts = parsed.path.strip("/").split("/")

    if "github" in parsed.netloc:
        repo_type = "github"
    elif "gitlab" in parsed.netloc:
        repo_type = "gitlab"
    elif "bitbucket" in parsed.netloc:
        repo_type = "bitbucket"
    else:
        repo_type = "local"

    owner = path_parts[0] if len(path_parts) > 0 else ""
    repo = path_parts[1].replace(".git", "") if len(path_parts) > 1 else ""

    return {"owner": owner, "repo": repo, "repo_type": repo_type}


def load_configs() -> Dict[str, Any]:
    """
    加载所有配置文件（生成器、语言、嵌入器）。

    返回:
        {"generator": dict, "lang": dict, "embedder": dict}
    """
    configs: Dict[str, Any] = {}

    try:
        configs["generator"] = load_generator_config()
        logger.info("✓ 已加载生成器配置")
    except Exception as e:
        logger.warning(f"加载生成器配置失败: {e}")
        configs["generator"] = {
            "default_provider": "dashscope",
            "default_model": "qwen-plus",
        }

    try:
        configs["lang"] = load_lang_config()
        logger.info("✓ 已加载语言配置")
    except Exception as e:
        logger.warning(f"加载语言配置失败: {e}")
        configs["lang"] = {
            "default": "zh",
            "options": [
                {"code": "zh", "name": "中文"},
                {"code": "en", "name": "English"},
            ],
        }

    try:
        configs["embedder"] = load_embedder_config()
        logger.info("✓ 已加载嵌入器配置")
    except Exception as e:
        logger.warning(f"加载嵌入器配置失败: {e}")
        configs["embedder"] = {}

    return configs


def get_cache_key(
    owner: str, repo: str, repo_type: str,
    language: str, comprehensive: bool = True,
) -> str:
    """
    生成 Wiki 缓存键。

    对应前端 page.tsx 中的 getCacheKey() 函数。
    """
    mode = "comprehensive" if comprehensive else "concise"
    return f"deepwiki_cache_{repo_type}_{owner}_{repo}_{language}_{mode}"


def generate_file_url(file_path: str, repo_url: str, repo_type: str = "github") -> str:
    """
    生成平台特定的文件 URL。

    对应前端 page.tsx 中的 generateFileUrl() 函数。
    """
    clean_path = file_path.lstrip("./").lstrip("/")

    if repo_type == "github":
        return f"{repo_url.rstrip('/')}/blob/main/{clean_path}"
    elif repo_type == "gitlab":
        return f"{repo_url.rstrip('/')}/-/blob/main/{clean_path}"
    elif repo_type == "bitbucket":
        return f"{repo_url.rstrip('/')}/src/main/{clean_path}"
    else:
        return f"{repo_url.rstrip('/')}/{clean_path}"


# ══════════════════════════════════════════════════════════════════════════
# BaseFlow — 业务流公共基类
# ══════════════════════════════════════════════════════════════════════════


class BaseFlow:
    """
    业务流公共基类。

    所有业务流（WikiGenerationFlow, SimpleChatFlow, DeepResearchFlow）
    继承此类，获得以下公共能力：

      - 仓库信息解析（repo_url → owner, repo, repo_type）
      - 配置加载（generator, lang, embedder）
      - LLM 调用（call_llm_and_collect）
      - RAG 检索器初始化（_init_retriever）
      - 项目查找（_find_project_id）

    子类需实现:
      - run() — 执行流程主入口
    """

    def __init__(
        self,
        repo_url: str,
        provider: str = "dashscope",
        model: str = "qwen-plus",
        language: str = "zh",
        use_database: bool = True,
        local_path: Optional[str] = None,
    ):
        self.repo_url = repo_url
        self.provider = provider
        self.model = model
        self.language = language
        self.use_database = use_database
        self.local_path = local_path

        # 解析仓库信息
        repo_info = parse_repo_url(repo_url)
        self.owner = repo_info["owner"]
        self.repo = repo_info["repo"]
        self.repo_type = repo_info["repo_type"]

        # 加载配置
        self.configs = load_configs()
        self.lang_config = self.configs.get("lang", {})
        self.language_name = _get_language_name(language)

        # RAG 组件
        self.retriever: Optional[PgvectorRetriever] = None
        self.project_id: Optional[str] = None

    # ── 项目查找 ──────────────────────────────────────────────────────────

    def _find_project_id(self) -> Optional[str]:
        """
        在数据库中查找与 repo_url 或 local_path 匹配的项目 ID。

        匹配优先级:
          1. repo_url 精确/包含匹配
          2. 项目名称匹配（从 repo_url 解析出的 repo 名）
          3. local_path 匹配（从本地路径提取的目录名）

        返回:
            匹配到的 project_id，未找到则返回 None
        """
        try:
            projects = ProjectRepository.list_all()
            for proj in projects:
                # 1. 按 repo_url 匹配
                proj_url = proj.get("repo_url", "") or proj.get("url", "")
                if proj_url and (self.repo_url in proj_url or proj_url in self.repo_url):
                    pid = proj.get("id") or proj.get("project_id")
                    self.project_id = str(pid) if pid else None
                    return self.project_id

                # 2. 按项目名称匹配（从 repo_url 解析出的 repo 名）
                proj_name = proj.get("name", "")
                if self.repo and self.repo.lower() in proj_name.lower():
                    pid = proj.get("id") or proj.get("project_id")
                    self.project_id = str(pid) if pid else None
                    return self.project_id

                # 3. 按 local_path 匹配（从本地路径提取目录名）
                if self.local_path:
                    local_dirname = os.path.basename(os.path.normpath(self.local_path.rstrip("/\\")))
                    if local_dirname.lower() == proj_name.lower():
                        pid = proj.get("id") or proj.get("project_id")
                        self.project_id = str(pid) if pid else None
                        logger.info(f"✓ 通过 local_path 匹配到项目: {proj_name} (id={self.project_id})")
                        return self.project_id

            logger.warning(f"未找到匹配的项目: repo_url={self.repo_url}, local_path={self.local_path}")
            return None
        except Exception as e:
            logger.warning(f"查找项目失败: {e}")
            return None

    # ── RAG 检索器初始化 ──────────────────────────────────────────────────

    def _init_retriever(self, top_k: int = 5) -> Optional[PgvectorRetriever]:
        """
        初始化 RAG 检索器。

        使用 PgvectorRetriever 进行混合检索（向量 + 关键词）。
        如果 use_database=False 或未找到 project_id，则跳过。

        参数:
            top_k: 检索返回的 top-k 结果数

        返回:
            PgvectorRetriever 实例，失败则返回 None
        """
        if not self.use_database:
            logger.info("跳过 RAG 检索器初始化（use_database=False）")
            return None

        # 如果还没有 project_id，尝试查找
        if not self.project_id:
            self._find_project_id()

        if not self.project_id:
            logger.warning(f"未找到项目: {self.repo_url}，跳过 RAG 检索")
            return None

        try:
            self.retriever = PgvectorRetriever(
                project_id=self.project_id,
                retrieval_type="hybrid",
                top_k=top_k,
            )
            logger.info(
                f"✓ RAG 检索器已初始化 "
                f"(project_id={self.project_id}, top_k={top_k})"
            )
            return self.retriever
        except Exception as e:
            logger.warning(f"初始化 RAG 检索器失败: {e}")
            return None

    # ── 子类需实现 ────────────────────────────────────────────────────────

    async def run(self) -> Any:
        """
        执行流程主入口。

        子类必须实现此方法，定义具体的业务流程。
        """
        raise NotImplementedError("子类必须实现 run() 方法")
