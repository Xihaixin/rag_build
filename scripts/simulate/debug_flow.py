"""
debug_flow.py — DeepWiki-open 业务逻辑流独立调试程序
=====================================================

本程序是一个 **独立可调试** 的 Python 脚本，完整复现 DeepWiki-open 的三大核心业务逻辑流：

  1. Wiki 文档生成 (WikiGenerationFlow)
     - 获取仓库文件树和 README
     - 调用 LLM 确定 Wiki 结构（解析 XML）
     - 逐页生成 Wiki 页面内容

  2. 用户 Q&A 简单聊天 (SimpleChatFlow)
     - 构建 RAG 上下文
     - 组装 prompt 并调用 LLM 流式回答

  3. 深度研究 (DeepResearchFlow)
     - 多轮迭代研究（最多 5 轮）
     - 自动检测研究是否完成
     - 提取研究阶段（计划/更新/结论）

设计目标：
  - 不是 API，不是 FastAPI 应用
  - 直接导入项目已有的后端组件（数据管理器、RAG 检索器、prompt 模板）
  - 使用数据库中已有的数据
  - 可通过命令行参数选择运行模式
  - 适合在 VS Code 中设置断点调试，理解完整业务逻辑流

用法：
  python scripts/simulate/debug_flow.py --mode wiki --repo-url https://github.com/user/repo
  python scripts/simulate/debug_flow.py --mode chat --repo-url https://github.com/user/repo --query "如何配置项目？"
  python scripts/simulate/debug_flow.py --mode research --repo-url https://github.com/user/repo --query "架构设计原理"

依赖：
  - 项目后端组件（api/, rag_optimizer/）
  - PostgreSQL 数据库（含已导入的项目数据）
  - LLM 配置（api/config/generator.json）
"""

import asyncio
import json
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ── 项目路径 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── 日志配置 ──────────────────────────────────────────────────────────────
from config.logging_config import setup_logging
setup_logging()
logger = logging.getLogger("debug_flow")

load_dotenv()

# ── 项目后端导入 ──────────────────────────────────────────────────────────
# 数据模型
from rag_optimizer.db.models import (
    Project, RawDocument, DocumentChunk, WikiPage as DBWikiPage,
)

# 数据仓库
from rag_optimizer.db.repository import (
    ProjectRepository, DocumentRepository, ChunkRepository,
)

# 数据库连接
from rag_optimizer.db.connection import SyncDatabaseConnection

# 配置
from api.config import (
    load_generator_config, load_embedder_config, load_lang_config,
    get_model_config, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES,
)

# Prompt 模板
from api.prompts import (
    RAG_SYSTEM_PROMPT, RAG_TEMPLATE,
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT,
)

# RAG 检索器
from rag_optimizer.integration.deepwiki_adapter import (
    PgvectorRetriever, PgvectorDatabaseManager,
)

# LLM 调用
from api.simple_chat import call_llm_stream

# ── 测试数据 ──────────────────────────────────────────────────────────────
from scripts.simulate.fixtures import (
    SAMPLE_FILE_TREE, SAMPLE_README, SAMPLE_WIKI_STRUCTURE_XML,
    SAMPLE_GENERATED_PAGE_CONTENT, SAMPLE_REPO_INFO,
)


# ══════════════════════════════════════════════════════════════════════════
# SSE 解析辅助函数
# ══════════════════════════════════════════════════════════════════════════

def _parse_sse_chunk(chunk: str) -> Optional[str]:
    """
    解析 call_llm_stream 返回的 SSE 格式字符串，提取实际文本内容。
    
    call_llm_stream 返回的格式为: data: {"content":"文本块"}\n\n
    或错误时: data: {"error":"错误信息"}\n\n
    
    参数:
        chunk: SSE 格式的字符串块
        
    返回:
        提取的文本内容，如果是错误块则返回 None
    """
    if not chunk or not chunk.strip():
        return None
    
    # 移除 "data: " 前缀和尾部的 "\n\n"
    text = chunk.strip()
    if text.startswith("data: "):
        text = text[6:]  # 去掉 "data: " 前缀
    
    try:
        data = json.loads(text)
        if "error" in data:
            logger.warning(f"LLM 返回错误: {data['error']}")
            return None
        return data.get("content", "")
    except json.JSONDecodeError:
        # 如果不是 JSON 格式，直接返回原始文本（兼容非 SSE 格式）
        return chunk


async def _call_llm_and_collect(
    provider: str,
    model: Optional[str],
    messages: List[Dict[str, str]],
) -> str:
    """
    调用 call_llm_stream 并自动解析 SSE 格式，返回完整的纯文本响应。
    
    这是 call_llm_stream 的便捷封装，自动处理 SSE 解析，
    避免在每个调用点重复编写 SSE 解析逻辑。
    
    参数:
        provider: LLM 提供者 (dashscope, google, openai, openrouter, ollama)
        model: 模型名称
        messages: 消息列表
        
    返回:
        完整的纯文本响应（不含 SSE 格式标记）
    """
    full_response = ""
    try:
        async for chunk in call_llm_stream(provider, model, messages):
            if chunk:
                text = _parse_sse_chunk(chunk)
                if text:
                    full_response += text
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        raise
    
    return full_response


# ══════════════════════════════════════════════════════════════════════════
# Part 1: 数据模型
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class WikiPage:
    """Wiki 页面 — 对应前端 page.tsx 中的 WikiPage 接口"""
    id: str
    title: str
    content: str = ""
    filePaths: List[str] = field(default_factory=list)
    importance: str = "medium"  # 'high' | 'medium' | 'low'
    relatedPages: List[str] = field(default_factory=list)
    parentId: Optional[str] = None
    isSection: bool = False
    children: List[str] = field(default_factory=list)


@dataclass
class WikiSection:
    """Wiki 章节 — 对应前端 page.tsx 中的 WikiSection 接口"""
    id: str
    title: str
    pages: List[str] = field(default_factory=list)
    subsections: List[str] = field(default_factory=list)


@dataclass
class WikiStructure:
    """Wiki 结构 — 对应前端 page.tsx 中的 WikiStructure 接口"""
    id: str
    title: str
    description: str = ""
    pages: List[WikiPage] = field(default_factory=list)
    sections: List[WikiSection] = field(default_factory=list)
    rootSections: List[str] = field(default_factory=list)


@dataclass
class Message:
    """聊天消息 — 对应前端 Ask.tsx 中的 Message 接口"""
    role: str  # 'user' | 'assistant' | 'system'
    content: str


@dataclass
class ResearchStage:
    """研究阶段 — 对应前端 Ask.tsx 中的 ResearchStage 接口"""
    title: str
    content: str
    iteration: int
    type: str  # 'plan' | 'update' | 'conclusion'


# ══════════════════════════════════════════════════════════════════════════
# Part 2: 配置加载
# ══════════════════════════════════════════════════════════════════════════

def load_configs() -> Dict[str, Any]:
    """
    加载所有配置文件，模拟前端从 /models/config 和 /lang/config 获取配置的过程。
    
    对应前端 page.tsx 中:
      - fetch('/models/config') → 获取 provider, model 列表
      - fetch('/lang/config') → 获取语言配置
    """
    configs = {}
    
    # 加载生成器配置（provider/model）
    try:
        generator_config = load_generator_config()
        configs["generator"] = generator_config
        logger.info("✓ 已加载生成器配置")
    except Exception as e:
        logger.warning(f"加载生成器配置失败: {e}")
        configs["generator"] = {"default_provider": "google", "default_model": "gemini-2.0-flash-exp"}
    
    # 加载语言配置
    try:
        lang_config = load_lang_config()
        configs["lang"] = lang_config
        logger.info("✓ 已加载语言配置")
    except Exception as e:
        logger.warning(f"加载语言配置失败: {e}")
        configs["lang"] = {"default": "zh", "options": [{"code": "zh", "name": "中文"}, {"code": "en", "name": "English"}]}
    
    # 加载嵌入器配置
    try:
        embedder_config = load_embedder_config()
        configs["embedder"] = embedder_config
        logger.info("✓ 已加载嵌入器配置")
    except Exception as e:
        logger.warning(f"加载嵌入器配置失败: {e}")
        configs["embedder"] = {}
    
    return configs


def get_language_name(language_code: str, lang_config: Optional[Dict] = None) -> str:
    """
    根据语言代码获取语言名称。
    
    对应前端 page.tsx 中的 getLanguageName() 函数。
    """
    if lang_config is None:
        lang_config = load_lang_config()
    
    options = lang_config.get("options", [])
    for opt in options:
        if opt.get("code") == language_code:
            return opt.get("name", language_code)
    return language_code


def get_cache_key(owner: str, repo: str, repo_type: str, language: str, comprehensive: bool = True) -> str:
    """
    生成 Wiki 缓存键。
    
    对应前端 page.tsx 中的 getCacheKey() 函数:
      `deepwiki_cache_{repoType}_{owner}_{repo}_{language}_{comprehensive|concise}`
    """
    mode = "comprehensive" if comprehensive else "concise"
    return f"deepwiki_cache_{repo_type}_{owner}_{repo}_{language}_{mode}"


def generate_file_url(file_path: str, repo_url: str, repo_type: str = "github") -> str:
    """
    生成平台特定的文件 URL。
    
    对应前端 page.tsx 中的 generateFileUrl() 函数。
    """
    # 移除开头的 ./ 或 /
    clean_path = file_path.lstrip("./").lstrip("/")
    
    if repo_type == "github":
        return f"{repo_url.rstrip('/')}/blob/main/{clean_path}"
    elif repo_type == "gitlab":
        return f"{repo_url.rstrip('/')}/-/blob/main/{clean_path}"
    elif repo_type == "bitbucket":
        return f"{repo_url.rstrip('/')}/src/main/{clean_path}"
    else:
        return f"{repo_url.rstrip('/')}/{clean_path}"


def parse_repo_url(repo_url: str) -> Dict[str, str]:
    """
    解析仓库 URL，提取 owner, repo, repo_type。
    
    对应前端 page.tsx 中的 repoInfo 计算逻辑。
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


# ══════════════════════════════════════════════════════════════════════════
# Part 3: Wiki 文档生成流
# ══════════════════════════════════════════════════════════════════════════

class WikiGenerationFlow:
    """
    Wiki 文档生成流 — 完整复现前端 page.tsx 中的 Wiki 生成逻辑。
    
    流程:
      1. fetch_repository_structure()  → 获取文件树和 README
      2. determine_wiki_structure()    → 调用 LLM 确定 Wiki 结构
      3. _generate_all_pages()         → 逐页生成 Wiki 页面内容
      4. print_summary()               → 打印结果摘要
    
    对应前端 page.tsx 中的:
      - fetchRepositoryStructure()
      - determineWikiStructure()
      - generatePageContent()
    """
    
    def __init__(
        self,
        repo_url: str,
        provider: str = "google",
        model: str = "gemini-2.0-flash-exp",
        language: str = "zh",
        comprehensive: bool = True,
        use_database: bool = True,
    ):
        self.repo_url = repo_url
        self.provider = provider
        self.model = model
        self.language = language
        self.comprehensive = comprehensive
        self.use_database = use_database
        
        # 解析仓库信息
        repo_info = parse_repo_url(repo_url)
        self.owner = repo_info["owner"]
        self.repo = repo_info["repo"]
        self.repo_type = repo_info["repo_type"]
        
        # 加载配置
        self.configs = load_configs()
        self.lang_config = self.configs.get("lang", {})
        self.language_name = get_language_name(language, self.lang_config)
        
        # 状态
        self.file_tree: Optional[str] = None
        self.readme: Optional[str] = None
        self.wiki_structure: Optional[WikiStructure] = None
        self.generated_pages: Dict[str, str] = {}  # page_id → content
        self.project_id: Optional[str] = None
        
        # 缓存键（对应前端 getCacheKey）
        self.cache_key = get_cache_key(
            self.owner, self.repo, self.repo_type, self.language, self.comprehensive
        )
        
        logger.info(f"初始化 WikiGenerationFlow:")
        logger.info(f"  仓库: {repo_url}")
        logger.info(f"  提供者: {provider}/{model}")
        logger.info(f"  语言: {self.language_name} ({language})")
        logger.info(f"  模式: {'comprehensive' if comprehensive else 'concise'}")
        logger.info(f"  缓存键: {self.cache_key}")
    
    # ── 步骤 1: 获取仓库结构 ──────────────────────────────────────────────
    
    def fetch_repository_structure(self) -> Tuple[str, str]:
        """
        获取仓库文件树和 README 内容。
        
        对应前端 page.tsx 中的 fetchRepositoryStructure():
          - GitHub: GET https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1
          - GitLab: GET https://gitlab.com/api/v4/projects/{id}/repository/tree?recursive=true
          - Bitbucket: GET https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/src
          - 本地: GET /local_repo/structure?path=...
        
        本实现支持两种模式:
          1. use_database=True  → 从 PostgreSQL 数据库查询已有数据
          2. use_database=False → 使用 fixtures 中的样本数据
        """
        logger.info("\n" + "=" * 60)
        logger.info("步骤 1: 获取仓库结构 (fetch_repository_structure)")
        logger.info("=" * 60)
        
        if self.use_database:
            logger.info("尝试从数据库获取仓库结构...")
            result = self._fetch_from_database()
            if result is not None:
                self.file_tree, self.readme = result
                logger.info(f"✓ 从数据库获取文件树 ({len(self.file_tree)} 字符)")
                logger.info(f"✓ 从数据库获取 README ({len(self.readme)} 字符)")
                return self.file_tree, self.readme
        
        # 回退到 fixtures 样本数据
        logger.info("使用 fixtures 样本数据...")
        self.file_tree = SAMPLE_FILE_TREE
        self.readme = SAMPLE_README
        logger.info(f"✓ 使用样本文件树 ({len(self.file_tree)} 字符)")
        logger.info(f"✓ 使用样本 README ({len(self.readme)} 字符)")
        
        return self.file_tree, self.readme
    
    def _fetch_from_database(self) -> Optional[Tuple[str, str]]:
        """
        从 PostgreSQL 数据库查询项目数据。
        
        对应后端 api/api.py 中的 /local_repo/structure 端点逻辑。
        使用 ProjectRepository 和 DocumentRepository 获取数据。
        """
        try:
            # 查找项目 — 使用 list_all() 遍历（注意：没有 get_by_name 方法）
            projects = ProjectRepository.list_all()
            target_project = None
            for proj in projects:
                proj_url = proj.get("repo_url", "") or proj.get("url", "")
                if self.repo_url in proj_url or proj_url in self.repo_url:
                    target_project = proj
                    break
                # 也匹配名称
                proj_name = proj.get("name", "")
                if self.repo.lower() in proj_name.lower():
                    target_project = proj
                    break
            
            if target_project is None:
                logger.warning(f"未找到匹配的项目: {self.repo_url}")
                logger.info(f"  可用项目: {[p.get('name', '?') for p in projects[:5]]}")
                return None
            
            raw_id = target_project.get("id") or target_project.get("project_id")
            self.project_id = str(raw_id) if raw_id else ""
            logger.info(f"  找到项目: {target_project.get('name')} (id={self.project_id})")
            
            # 获取文档列表
            if not self.project_id:
                logger.warning("项目 ID 为空")
                return None
            documents = DocumentRepository.get_by_project(self.project_id)
            logger.info(f"  项目文档数: {len(documents)}")
            
            # 构建文件树
            file_tree_lines = []
            readme_content = None
            
            for doc in documents:
                file_path = doc.get("file_path", "")
                if file_path == "README.md" or file_path.lower().endswith("readme.md"):
                    readme_content = doc.get("content", "")
                
                # 构建文件树行（兼容 Windows 反斜杠路径）
                depth = file_path.replace("\\", "/").count("/")
                indent = "  " * depth
                file_tree_lines.append(f"{indent}{file_path}")
            
            if not file_tree_lines:
                logger.warning("数据库中没有文档记录")
                return None
            
            file_tree = "\n".join(sorted(file_tree_lines))
            readme = readme_content or "# No README found"
            
            return file_tree, readme
            
        except Exception as e:
            logger.error(f"从数据库获取数据失败: {e}", exc_info=True)
            return None
    
    # ── 步骤 2: 确定 Wiki 结构 ────────────────────────────────────────────
    
    async def determine_wiki_structure(self) -> WikiStructure:
        """
        调用 LLM 确定 Wiki 文档结构。
        
        对应前端 page.tsx 中的 determineWikiStructure():
          1. 构建 prompt（包含文件树 + README）
          2. 通过 WebSocket 发送到 ws://localhost:8001/ws/chat
          3. 解析 LLM 返回的 XML 响应
          4. 生成 WikiStructure 对象
        
        本实现直接调用 call_llm_stream() 而非通过 WebSocket。
        """
        logger.info("\n" + "=" * 60)
        logger.info("步骤 2: 确定 Wiki 结构 (determine_wiki_structure)")
        logger.info("=" * 60)
        
        # 确保已有文件树和 README
        if not self.file_tree or not self.readme:
            self.fetch_repository_structure()
        
        # 构建 prompt
        prompt = self._build_structure_prompt()
        logger.info(f"构建的结构 prompt ({len(prompt)} 字符)")
        logger.debug(f"Prompt 前 200 字符: {prompt[:200]}...")
        
        # 调用 LLM
        logger.info(f"调用 LLM ({self.provider}/{self.model})...")
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        full_response = ""
        try:
            full_response = await _call_llm_and_collect(self.provider, self.model, messages)
            logger.info(f"✓ LLM 返回响应 ({len(full_response)} 字符)")
        except Exception as e:
            logger.warning(f"LLM 调用失败，使用样本数据: {e}")
            full_response = SAMPLE_WIKI_STRUCTURE_XML
        
        # 解析 XML
        self.wiki_structure = self._parse_wiki_xml(full_response)
        logger.info(f"✓ 解析 Wiki 结构: {self.wiki_structure.title}")
        logger.info(f"  页面数: {len(self.wiki_structure.pages)}")
        logger.info(f"  章节数: {len(self.wiki_structure.sections)}")
        
        return self.wiki_structure
    
    def _build_structure_prompt(self) -> str:
        """
        构建 Wiki 结构 prompt。
        
        对应前端 page.tsx 中 determineWikiStructure() 里的 prompt 构建逻辑（约 lines 712-832）。
        
        综合模式 (comprehensive):
          - 8-12 个页面
          - 包含 sections 分组
          - 每个页面有 filePaths, importance, relatedPages
        
        简洁模式 (concise):
          - 4-6 个页面
          - 无 sections
          - 简化的页面结构
        """
        if self.comprehensive:
            prompt = f"""You are a technical documentation expert. Analyze the following repository structure and README to create a comprehensive wiki structure.

Repository URL: {self.repo_url}
Language: {self.language_name}

## File Tree
```
{self.file_tree}
```

## README
{self.readme}

## Task
Create a comprehensive wiki structure with 8-12 pages that covers all important aspects of this project.

For each page, provide:
- id: unique identifier (use kebab-case)
- title: page title
- filePaths: relevant source files from the file tree
- importance: high/medium/low
- relatedPages: ids of related pages

Group related pages into sections. Each section can have subsections.

## Output Format
Return ONLY valid XML with this exact structure:
<wiki_structure>
  <title>Project Title</title>
  <description>Brief project description</description>
  <sections>
    <section>
      <id>section-id</id>
      <title>Section Title</title>
      <pages>
        <page>
          <id>page-id</id>
          <title>Page Title</title>
          <filePaths>
            <path>src/file1.py</path>
            <path>src/file2.py</path>
          </filePaths>
          <importance>high</importance>
          <relatedPages>
            <id>related-page-id</id>
          </relatedPages>
        </page>
      </pages>
      <subsections>
        <section>
          <id>subsection-id</id>
          <title>Subsection Title</title>
          <pages>
            <page>
              <id>sub-page-id</id>
              <title>Sub Page Title</title>
              <filePaths>
                <path>src/file.py</path>
              </filePaths>
              <importance>medium</importance>
              <relatedPages>
                <id>another-page-id</id>
              </relatedPages>
            </page>
          </pages>
        </section>
      </subsections>
    </section>
  </sections>
</wiki_structure>"""
        else:
            prompt = f"""You are a technical documentation expert. Analyze the following repository structure and README to create a concise wiki structure.

Repository URL: {self.repo_url}
Language: {self.language_name}

## File Tree
```
{self.file_tree}
```

## README
{self.readme}

## Task
Create a concise wiki structure with 4-6 pages covering the most important aspects.

For each page, provide:
- id: unique identifier (use kebab-case)
- title: page title
- filePaths: relevant source files
- importance: high/medium/low
- relatedPages: ids of related pages

## Output Format
Return ONLY valid XML with this exact structure:
<wiki_structure>
  <title>Project Title</title>
  <description>Brief project description</description>
  <pages>
    <page>
      <id>page-id</id>
      <title>Page Title</title>
      <filePaths>
        <path>src/file.py</path>
      </filePaths>
      <importance>high</importance>
      <relatedPages>
        <id>related-page-id</id>
      </relatedPages>
    </page>
  </pages>
</wiki_structure>"""
        
        return prompt
    
    async def _call_llm_for_structure(self, prompt: str) -> str:
        """
        调用 LLM 获取 Wiki 结构。
        
        对应前端 page.tsx 中 determineWikiStructure() 里的 WebSocket 调用逻辑（lines 852-899）。
        前端通过 WebSocket 发送，这里直接调用 call_llm_stream。
        """
        messages = [{"role": "user", "content": prompt}]
        
        full_response = ""
        try:
            full_response = await _call_llm_and_collect(self.provider, self.model, messages)
            return full_response
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise
    
    def _parse_wiki_xml(self, xml_text: str) -> WikiStructure:
        """
        解析 LLM 返回的 XML 响应为 WikiStructure 对象。
        
        对应前端 page.tsx 中的 XML 解析逻辑（lines 942-1083）：
          1. 使用 DOMParser 解析 XML
          2. 提取 <title>, <description>
          3. 遍历 <pages> 下的 <page> 元素
          4. 遍历 <sections> 下的 <section> 元素
          5. 构建 WikiSection 和 WikiPage 对象
        """
        # 清理 markdown 代码块标记（LLM 有时会返回 ```xml ... ```）
        xml_text = re.sub(r'```(?:xml)?\s*', '', xml_text).strip()
        
        # 尝试修复常见的 XML 问题
        xml_text = self._repair_xml(xml_text)
        
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"XML 解析失败，尝试修复: {e}")
            # 如果解析失败，使用样本数据
            logger.info("使用样本 Wiki 结构 XML 作为回退")
            try:
                root = ET.fromstring(SAMPLE_WIKI_STRUCTURE_XML)
            except ET.ParseError:
                # 极端回退：创建默认结构
                return self._create_default_structure()
        
        # 提取基本信息
        title = self._extract_tag_text(root, "title") or f"{self.repo} Documentation"
        description = self._extract_tag_text(root, "description") or ""
        
        structure = WikiStructure(
            id=f"wiki-{self.owner}-{self.repo}",
            title=title,
            description=description,
        )
        
        # 解析页面（无章节模式）
        pages_root = root.find("pages")
        if pages_root is not None:
            for page_el in pages_root.findall("page"):
                page = self._parse_page_element(page_el)
                structure.pages.append(page)
        
        # 解析章节（有章节模式）
        sections_root = root.find("sections")
        if sections_root is not None:
            for section_el in sections_root.findall("section"):
                section = self._parse_section_element(section_el, structure)
                structure.sections.append(section)
                structure.rootSections.append(section.id)
        
        # 如果既没有 pages 也没有 sections，尝试从根级别提取 page
        if not structure.pages and not structure.sections:
            for page_el in root.findall("page"):
                page = self._parse_page_element(page_el)
                structure.pages.append(page)
        
        logger.info(f"解析完成: {len(structure.pages)} 页面, {len(structure.sections)} 章节")
        return structure
    
    def _repair_xml(self, xml_text: str) -> str:
        """修复常见的 XML 格式问题"""
        # 移除 BOM
        xml_text = xml_text.lstrip('\ufeff')
        # 确保有根元素
        if not xml_text.strip().startswith("<"):
            xml_text = f"<wiki_structure>{xml_text}</wiki_structure>"
        return xml_text
    
    def _extract_tag_text(self, element: ET.Element, tag: str) -> Optional[str]:
        """安全提取子标签的文本内容"""
        sub = element.find(tag)
        if sub is not None and sub.text:
            return sub.text.strip()
        return None
    
    def _parse_page_element(self, page_el: ET.Element) -> WikiPage:
        """解析 XML <page> 元素为 WikiPage 对象"""
        page_id = self._extract_tag_text(page_el, "id") or f"page-{len(self.generated_pages)}"
        title = self._extract_tag_text(page_el, "title") or page_id
        importance = self._extract_tag_text(page_el, "importance") or "medium"
        
        # 解析文件路径
        file_paths = []
        fps = page_el.find("filePaths")
        if fps is not None:
            for path_el in fps.findall("path"):
                if path_el.text:
                    file_paths.append(path_el.text.strip())
        
        # 解析相关页面
        related_pages = []
        rps = page_el.find("relatedPages")
        if rps is not None:
            for id_el in rps.findall("id"):
                if id_el.text:
                    related_pages.append(id_el.text.strip())
        
        return WikiPage(
            id=page_id,
            title=title,
            filePaths=file_paths,
            importance=importance,
            relatedPages=related_pages,
        )
    
    def _parse_section_element(self, section_el: ET.Element, structure: WikiStructure) -> WikiSection:
        """解析 XML <section> 元素为 WikiSection 对象"""
        section_id = self._extract_tag_text(section_el, "id") or f"section-{len(structure.sections)}"
        title = self._extract_tag_text(section_el, "title") or section_id
        
        section = WikiSection(id=section_id, title=title)
        
        # 解析章节内的页面
        pages_el = section_el.find("pages")
        if pages_el is not None:
            for page_el in pages_el.findall("page"):
                page = self._parse_page_element(page_el)
                structure.pages.append(page)
                section.pages.append(page.id)
        
        # 递归解析子章节
        subsections_el = section_el.find("subsections")
        if subsections_el is not None:
            for sub_el in subsections_el.findall("section"):
                sub_section = self._parse_section_element(sub_el, structure)
                structure.sections.append(sub_section)
                section.subsections.append(sub_section.id)
        
        return section
    
    def _create_default_structure(self) -> WikiStructure:
        """创建默认的 Wiki 结构（极端回退）"""
        structure = WikiStructure(
            id=f"wiki-{self.owner}-{self.repo}",
            title=f"{self.repo} Documentation",
            description=f"Documentation for {self.repo}",
        )
        
        pages_data = [
            ("overview", "项目概述", ["README.md"], "high"),
            ("getting-started", "快速开始", ["README.md"], "high"),
            ("architecture", "架构设计", ["src/"], "high"),
            ("api-reference", "API 参考", ["api/"], "medium"),
            ("configuration", "配置说明", ["config/"], "medium"),
            ("development", "开发指南", ["src/", "tests/"], "low"),
        ]
        
        for pid, ptitle, paths, imp in pages_data:
            structure.pages.append(WikiPage(
                id=pid, title=ptitle, filePaths=paths, importance=imp
            ))
        
        return structure
    
    # ── 步骤 3: 生成所有页面 ──────────────────────────────────────────────
    
    async def _generate_all_pages(self) -> Dict[str, str]:
        """
        逐页生成 Wiki 页面内容。
        
        对应前端 page.tsx 中的 generatePageContent 调用逻辑（lines 1088-1154）：
          - MAX_CONCURRENT = 1（串行生成）
          - 使用 processQueue 模式逐个处理
          - 每生成一页就更新状态
        
        本实现串行调用 LLM 为每个页面生成内容。
        """
        logger.info("\n" + "=" * 60)
        logger.info("步骤 3: 生成所有页面 (_generate_all_pages)")
        logger.info("=" * 60)
        
        if not self.wiki_structure:
            await self.determine_wiki_structure()
        
        if not self.wiki_structure:
            logger.warning("wiki_structure 为 None，无法生成页面")
            return {}
        
        pages = self.wiki_structure.pages
        total = len(pages)
        logger.info(f"开始生成 {total} 个页面（串行模式）")
        
        for idx, page in enumerate(pages):
            logger.info(f"\n  [{idx + 1}/{total}] 生成页面: {page.title} ({page.id})")
            content = await self._generate_single_page(page)
            self.generated_pages[page.id] = content
            page.content = content
            logger.info(f"  ✓ 页面 '{page.title}' 生成完成 ({len(content)} 字符)")
        
        logger.info(f"\n✓ 所有 {total} 个页面生成完成")
        return self.generated_pages
    
    async def _generate_single_page(self, page: WikiPage) -> str:
        """
        生成单个 Wiki 页面内容。
        
        对应前端 page.tsx 中的 generatePageContent() 函数（lines 373-681）：
          1. 构建详细的页面 prompt（包含文件路径、语言、格式要求）
          2. 通过 WebSocket 发送到 ws://localhost:8001/ws/chat
          3. 接收流式响应并拼接
          4. 清理 markdown 分隔符
        
        本实现直接调用 call_llm_stream。
        """
        prompt = self._build_page_prompt(page)
        
        messages = [{"role": "user", "content": prompt}]
        
        full_content = ""
        try:
            full_content = await _call_llm_and_collect(self.provider, self.model, messages)
        except Exception as e:
            logger.warning(f"LLM 调用失败，使用样本内容: {e}")
            full_content = SAMPLE_GENERATED_PAGE_CONTENT
        
        # 清理 markdown 代码块分隔符（对应前端 line 645 的 cleanMarkdownDelimiters）
        full_content = self._clean_markdown_delimiters(full_content)
        
        return full_content
    
    def _build_page_prompt(self, page: WikiPage) -> str:
        """
        构建单个页面的生成 prompt。
        
        对应前端 page.tsx 中 generatePageContent() 里的 prompt 构建逻辑（lines 419-526）：
          - 包含页面标题、文件路径列表
          - 指定语言
          - 要求使用 Mermaid 图表
          - 要求使用表格
          - 要求代码引用
          - 要求引用链接
        """
        # 构建文件路径列表（带 URL）
        file_paths_str = ""
        for fp in page.filePaths:
            url = generate_file_url(fp, self.repo_url, self.repo_type)
            file_paths_str += f"  - {fp}\n    URL: {url}\n"
        
        # 构建相关页面列表
        related_str = ""
        if self.wiki_structure:
            for rp_id in page.relatedPages:
                rp_title = rp_id
                for p in self.wiki_structure.pages:
                    if p.id == rp_id:
                        rp_title = p.title
                        break
                related_str += f"  - [{rp_title}]({rp_id})\n"
        
        prompt = (
            f"You are a technical documentation writer. Generate a comprehensive wiki page for the following topic.\n\n"
            f"## Project\n"
            f"- Repository: {self.repo_url}\n"
            f"- Language: {self.language_name}\n"
            f"- Page ID: {page.id}\n"
            f"- Page Title: {page.title}\n"
            f"- Importance: {page.importance}\n\n"
            f"## Relevant Source Files\n"
            f"{file_paths_str or '  (No specific files assigned)'}\n"
            f"## Related Pages\n"
            f"{related_str or '  (No related pages)'}\n\n"
            f"## Requirements\n"
            f"1. Write the content in {self.language_name}\n"
            f"2. Include a Mermaid diagram where appropriate (flowchart, sequence diagram, or class diagram)\n"
            f"3. Use tables for structured data\n"
            f"4. Include code examples with proper syntax highlighting\n"
            f"5. Add citations and references to source files\n"
            f"6. Use proper markdown formatting\n"
        )
        
        return prompt
    
    def _clean_markdown_delimiters(self, content: str) -> str:
        """
        清理 markdown 代码块分隔符。
        
        对应前端 page.tsx line 645 的 cleanMarkdownDelimiters 函数。
        LLM 有时会在返回内容外包裹 ```markdown 或 ``` 代码块标记。
        """
        # 移除开头的 ```markdown 或 ``` 以及结尾的 ```
        content = re.sub(r'^```(?:markdown)?\s*\n', '', content)
        content = re.sub(r'\n```\s*$', '', content)
        return content.strip()
    
    # ── 步骤 4: 打印结果摘要 ──────────────────────────────────────────────
    
    def print_summary(self) -> None:
        """打印 Wiki 生成结果摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("Wiki 生成完成 — 结果摘要")
        logger.info("=" * 60)
        
        if not self.wiki_structure:
            logger.warning("未生成 Wiki 结构")
            return
        
        ws = self.wiki_structure
        logger.info(f"标题: {ws.title}")
        logger.info(f"描述: {ws.description}")
        logger.info(f"缓存键: {self.cache_key}")
        logger.info(f"")
        logger.info(f"页面总数: {len(ws.pages)}")
        logger.info(f"章节总数: {len(ws.sections)}")
        logger.info(f"")
        
        # 按重要性分组
        high_pages = [p for p in ws.pages if p.importance == "high"]
        medium_pages = [p for p in ws.pages if p.importance == "medium"]
        low_pages = [p for p in ws.pages if p.importance == "low"]
        
        logger.info(f"高优先级页面: {len(high_pages)}")
        for p in high_pages:
            status = "✓" if p.id in self.generated_pages else "✗"
            logger.info(f"  [{status}] {p.title} ({p.id})")
        
        logger.info(f"中优先级页面: {len(medium_pages)}")
        for p in medium_pages:
            status = "✓" if p.id in self.generated_pages else "✗"
            logger.info(f"  [{status}] {p.title} ({p.id})")
        
        logger.info(f"低优先级页面: {len(low_pages)}")
        for p in low_pages:
            status = "✓" if p.id in self.generated_pages else "✗"
            logger.info(f"  [{status}] {p.title} ({p.id})")
        
        logger.info(f"")
        logger.info(f"已生成内容页面: {len(self.generated_pages)}/{len(ws.pages)}")
        
        # 章节结构
        if ws.sections:
            logger.info(f"")
            logger.info("章节结构:")
            for section in ws.sections:
                logger.info(f"  📁 {section.title} ({section.id})")
                for pid in section.pages:
                    p = next((p for p in ws.pages if p.id == pid), None)
                    if p:
                        logger.info(f"    📄 {p.title}")
                for sub_id in section.subsections:
                    sub = next((s for s in ws.sections if s.id == sub_id), None)
                    if sub:
                        logger.info(f"  📂 {sub.title} ({sub.id})")
                        for pid in sub.pages:
                            p = next((p for p in ws.pages if p.id == pid), None)
                            if p:
                                logger.info(f"    📄 {p.title}")
        
        logger.info(f"")
        logger.info(f"语言: {self.language_name}")
        logger.info(f"提供者: {self.provider}/{self.model}")
        logger.info(f"仓库: {self.repo_url}")


# ══════════════════════════════════════════════════════════════════════════
# Part 4: 用户 Q&A 简单聊天流
# ══════════════════════════════════════════════════════════════════════════

class SimpleChatFlow:
    """
    用户 Q&A 简单聊天流 — 完整复现前端 Ask.tsx 中的简单聊天逻辑。
    
    流程:
      1. 构建 RAG 上下文（从 pgvector 检索相关文档）
      2. 组装 prompt（系统指令 + 对话历史 + RAG 上下文 + 用户问题）
      3. 调用 LLM 流式回答
      4. 返回完整回答
    
    对应前端 Ask.tsx 中的:
      - handleConfirmAsk() — 发送聊天请求
      - createChatWebSocket() — WebSocket 通信
    
    对应后端 api/simple_chat.py 中的:
      - build_simple_chat_prompt() — 构建 prompt
      - _handle_simple_chat() — 处理聊天
      - PgvectorRetriever — RAG 检索
    """
    
    def __init__(
        self,
        repo_url: str,
        provider: str = "google",
        model: str = "gemini-2.0-flash-exp",
        language: str = "zh",
        use_database: bool = True,
    ):
        self.repo_url = repo_url
        self.provider = provider
        self.model = model
        self.language = language
        self.use_database = use_database
        
        # 解析仓库信息
        repo_info = parse_repo_url(repo_url)
        self.owner = repo_info["owner"]
        self.repo = repo_info["repo"]
        self.repo_type = repo_info["repo_type"]
        
        # 加载配置
        self.configs = load_configs()
        self.lang_config = self.configs.get("lang", {})
        self.language_name = get_language_name(language, self.lang_config)
        
        # RAG 组件
        self.retriever: Optional[PgvectorRetriever] = None
        self.project_id: Optional[str] = None
        
        # 对话历史
        self.messages: List[Message] = []
        
        logger.info(f"初始化 SimpleChatFlow:")
        logger.info(f"  仓库: {repo_url}")
        logger.info(f"  提供者: {provider}/{model}")
        logger.info(f"  语言: {self.language_name}")
    
    def _init_retriever(self) -> Optional[PgvectorRetriever]:
        """
        初始化 RAG 检索器。
        
        对应后端 api/simple_chat.py 中 _handle_simple_chat() 的 retriever 初始化逻辑。
        使用 PgvectorRetriever 进行混合检索（向量 + 关键词）。
        """
        if not self.use_database:
            logger.info("跳过 RAG 检索器初始化（use_database=False）")
            return None
        
        try:
            # 查找项目 ID
            projects = ProjectRepository.list_all()
            for proj in projects:
                proj_url = proj.get("repo_url", "") or proj.get("url", "")
                if self.repo_url in proj_url or proj_url in self.repo_url:
                    self.project_id = proj.get("id") or proj.get("project_id")
                    break
                proj_name = proj.get("name", "")
                if self.repo.lower() in proj_name.lower():
                    self.project_id = proj.get("id") or proj.get("project_id")
                    break
            
            if not self.project_id:
                logger.warning(f"未找到项目: {self.repo_url}，跳过 RAG 检索")
                return None
            
            # 创建 PgvectorRetriever
            self.retriever = PgvectorRetriever(
                project_id=self.project_id,
                retrieval_type="hybrid",
                top_k=5,
            )
            logger.info(f"✓ RAG 检索器已初始化 (project_id={self.project_id})")
            return self.retriever
            
        except Exception as e:
            logger.warning(f"初始化 RAG 检索器失败: {e}")
            return None
    
    def _build_context(self, query: str) -> str:
        """
        构建 RAG 上下文文本。
        
        对应后端 api/simple_chat.py 中的 build_context_from_results() 函数。
        使用 PgvectorRetriever 检索相关文档块，然后格式化为上下文文本。
        """
        if not self.retriever:
            logger.info("无 RAG 检索器，跳过上下文构建")
            return ""
        
        try:
            # 执行检索 — PgvectorRetriever.search() 返回 (List[RetrievalResult], RetrievalStats)
            results, stats = self.retriever.search(query, top_k=5)
            
            if not results:
                logger.info("检索结果为空")
                return ""
            
            # 构建上下文文本
            context_parts = []
            for i, result in enumerate(results, 1):
                # RetrievalResult 是 dataclass，有 content, file_path, final_score 字段
                content = getattr(result, "content", "") or ""
                file_path = getattr(result, "file_path", "") or ""
                score = getattr(result, "final_score", 0.0) or getattr(result, "vector_score", 0.0)
                
                context_parts.append(
                    f"[{i}] 文件: {file_path}\n"
                    f"    相关度: {score:.4f}\n"
                    f"    内容: {content[:500]}..."
                )
            
            context = "\n\n".join(context_parts)
            logger.info(f"✓ RAG 上下文构建完成 ({len(context)} 字符, {len(results)} 个结果)")
            return context
            
        except Exception as e:
            logger.warning(f"RAG 检索失败: {e}")
            return ""
    
    def _build_prompt(self, query: str, context: str, history: Optional[List[Message]] = None) -> List[Dict[str, str]]:
        """
        构建聊天 prompt。
        
        对应后端 api/simple_chat.py 中的 build_simple_chat_prompt() 函数：
          1. 系统指令 (SIMPLE_CHAT_SYSTEM_PROMPT)
          2. 对话历史
          3. RAG 上下文
          4. 用户问题
        
        对应 api/prompts.py 中的 RAG_TEMPLATE 模板。
        """
        messages = []
        
        # 1. 系统指令
        system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT.format(
            language=self.language_name,
            repo_url=self.repo_url,
        )
        messages.append({"role": "system", "content": system_prompt})
        
        # 2. 对话历史
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        
        # 3. RAG 上下文 + 用户问题
        if context:
            user_prompt = RAG_TEMPLATE.format(
                system_prompt=system_prompt,
                conversation_history="",
                contexts=context,
                query=query,
                language=self.language_name,
            )
            messages.append({"role": "user", "content": user_prompt})
        else:
            messages.append({"role": "user", "content": query})
        
        return messages
    
    async def chat(self, query: str, history: Optional[List[Message]] = None) -> str:
        """
        执行一次聊天问答。
        
        对应前端 Ask.tsx 中的 handleConfirmAsk() 流程：
          1. 准备请求体（repo_url, messages, provider, model, language）
          2. 通过 WebSocket 发送
          3. 接收流式响应
        
        对应后端 api/simple_chat.py 中的 _handle_simple_chat() 流程：
          1. 初始化 RAG 检索器
          2. 检索相关文档
          3. 构建 prompt
          4. 调用 LLM 流式生成
        """
        logger.info("\n" + "=" * 60)
        logger.info("SimpleChatFlow.chat()")
        logger.info("=" * 60)
        logger.info(f"问题: {query}")
        
        # 步骤 1: 初始化 RAG 检索器
        if not self.retriever:
            self._init_retriever()
        
        # 步骤 2: 构建 RAG 上下文
        logger.info("步骤 1: RAG 检索...")
        context = self._build_context(query)
        
        # 步骤 3: 构建 prompt
        logger.info("步骤 2: 构建 prompt...")
        messages = self._build_prompt(query, context, history)
        logger.info(f"  messages 数: {len(messages)}")
        
        # 步骤 4: 调用 LLM
        logger.info(f"步骤 3: 调用 LLM ({self.provider}/{self.model})...")
        full_response = ""
        
        try:
            full_response = await _call_llm_and_collect(self.provider, self.model, messages)
            logger.info(f"✓ LLM 返回完成 ({len(full_response)} 字符)")
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            full_response = f"[错误] LLM 调用失败: {e}"
        
        # 记录对话历史
        self.messages.append(Message(role="user", content=query))
        self.messages.append(Message(role="assistant", content=full_response))
        
        return full_response
    
    def print_conversation(self) -> None:
        """打印对话历史"""
        logger.info("\n" + "=" * 60)
        logger.info("对话历史")
        logger.info("=" * 60)
        for i, msg in enumerate(self.messages):
            role_label = "👤 用户" if msg.role == "user" else "🤖 助手"
            logger.info(f"\n[{i + 1}] {role_label}:")
            # 只打印前 200 字符
            preview = msg.content[:200]
            if len(msg.content) > 200:
                preview += "..."
            logger.info(preview)


# ══════════════════════════════════════════════════════════════════════════
# Part 5: 深度研究流
# ══════════════════════════════════════════════════════════════════════════

class DeepResearchFlow:
    """
    深度研究流 — 完整复现前端 Ask.tsx 中的深度研究逻辑。
    
    流程:
      1. 发送初始研究问题（带 [DEEP RESEARCH] 标记）
      2. 自动继续研究（最多 5 轮迭代）
      3. 每轮检测研究是否完成
      4. 提取研究阶段（计划/更新/结论）
      5. 返回完整研究结果
    
    对应前端 Ask.tsx 中的:
      - handleConfirmAsk() — 发送初始研究请求
      - continueResearch() — 自动继续研究
      - checkIfResearchComplete() — 检测研究是否完成
      - extractResearchStage() — 提取研究阶段
    
    对应后端 api/simple_chat.py 中的:
      - build_deep_research_prompt() — 构建研究 prompt
      - _handle_deep_research() — 处理深度研究
    """
    
    # 研究完成标记 — 对应前端 Ask.tsx 中的 checkIfResearchComplete()
    COMPLETION_MARKERS = [
        "## Final Conclusion",
        "## Conclusion",
        "This concludes our research",
        "## 最终结论",
        "## 结论",
        "本研究至此结束",
    ]
    
    # 最大迭代次数 — 对应前端 Ask.tsx 中的 MAX_RESEARCH_ITERATIONS
    MAX_ITERATIONS = 5
    
    def __init__(
        self,
        repo_url: str,
        provider: str = "google",
        model: str = "gemini-2.0-flash-exp",
        language: str = "zh",
        use_database: bool = True,
    ):
        self.repo_url = repo_url
        self.provider = provider
        self.model = model
        self.language = language
        self.use_database = use_database
        
        # 解析仓库信息
        repo_info = parse_repo_url(repo_url)
        self.owner = repo_info["owner"]
        self.repo = repo_info["repo"]
        self.repo_type = repo_info["repo_type"]
        
        # 加载配置
        self.configs = load_configs()
        self.lang_config = self.configs.get("lang", {})
        self.language_name = get_language_name(language, self.lang_config)
        
        # RAG 组件
        self.retriever: Optional[PgvectorRetriever] = None
        self.project_id: Optional[str] = None
        
        # 研究状态
        self.messages: List[Message] = []
        self.research_stages: List[ResearchStage] = []
        self.current_iteration: int = 0
        self.is_complete: bool = False
        self.final_answer: str = ""
        
        logger.info(f"初始化 DeepResearchFlow:")
        logger.info(f"  仓库: {repo_url}")
        logger.info(f"  提供者: {provider}/{model}")
        logger.info(f"  语言: {self.language_name}")
        logger.info(f"  最大迭代: {self.MAX_ITERATIONS}")
    
    def _init_retriever(self) -> Optional[PgvectorRetriever]:
        """初始化 RAG 检索器（同 SimpleChatFlow）"""
        if not self.use_database:
            return None
        
        try:
            projects = ProjectRepository.list_all()
            for proj in projects:
                proj_url = proj.get("repo_url", "") or proj.get("url", "")
                if self.repo_url in proj_url or proj_url in self.repo_url:
                    self.project_id = proj.get("id") or proj.get("project_id")
                    break
                proj_name = proj.get("name", "")
                if self.repo.lower() in proj_name.lower():
                    self.project_id = proj.get("id") or proj.get("project_id")
                    break
            
            if not self.project_id:
                return None
            
            self.retriever = PgvectorRetriever(
                project_id=self.project_id,
                retrieval_type="hybrid",
                top_k=5,
            )
            return self.retriever
        except Exception as e:
            logger.warning(f"初始化 RAG 检索器失败: {e}")
            return None
    
    def _build_context(self, query: str) -> str:
        """构建 RAG 上下文（同 SimpleChatFlow）"""
        if not self.retriever:
            return ""
        
        try:
            # 执行检索 — PgvectorRetriever.search() 返回 (List[RetrievalResult], RetrievalStats)
            results, stats = self.retriever.search(query, top_k=5)
            if not results:
                return ""
            
            context_parts = []
            for i, result in enumerate(results, 1):
                content = getattr(result, "content", "") or ""
                file_path = getattr(result, "file_path", "") or ""
                score = getattr(result, "final_score", 0.0) or getattr(result, "vector_score", 0.0)
                
                context_parts.append(
                    f"[{i}] 文件: {file_path}\n"
                    f"    相关度: {score:.4f}\n"
                    f"    内容: {content[:500]}..."
                )
            
            return "\n\n".join(context_parts)
        except Exception as e:
            logger.warning(f"RAG 检索失败: {e}")
            return ""
    
    def _build_research_prompt(self, query: str, iteration: int, context: str) -> List[Dict[str, str]]:
        """
        构建深度研究 prompt。
        
        对应后端 api/simple_chat.py 中的 build_deep_research_prompt() 函数。
        根据迭代次数选择不同的 prompt 模板：
          - 第 1 轮: DEEP_RESEARCH_FIRST_ITERATION_PROMPT（研究计划）
          - 中间轮: DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT（研究更新）
          - 最终轮: DEEP_RESEARCH_FINAL_ITERATION_PROMPT（最终结论）
        
        对应 api/prompts.py 中的三个模板。
        """
        messages = []
        
        # 系统指令
        system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT.format(
            language=self.language_name,
            repo_url=self.repo_url,
        )
        messages.append({"role": "system", "content": system_prompt})
        
        # 选择 prompt 模板
        if iteration == 1:
            prompt_template = DEEP_RESEARCH_FIRST_ITERATION_PROMPT
        elif iteration >= self.MAX_ITERATIONS:
            prompt_template = DEEP_RESEARCH_FINAL_ITERATION_PROMPT
        else:
            prompt_template = DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT
        
        # 构建研究 prompt
        if context:
            research_prompt = prompt_template.format(
                query=query,
                contexts=context,
                iteration=iteration,
                language=self.language_name,
            )
        else:
            research_prompt = prompt_template.format(
                query=query,
                contexts="(无相关上下文)",
                iteration=iteration,
                language=self.language_name,
            )
        
        messages.append({"role": "user", "content": research_prompt})
        return messages
    
    def _check_if_complete(self, content: str) -> bool:
        """
        检测研究是否完成。
        
        对应前端 Ask.tsx 中的 checkIfResearchComplete() 函数（lines 176-209）。
        检查内容中是否包含完成标记。
        """
        for marker in self.COMPLETION_MARKERS:
            if marker in content:
                logger.info(f"  检测到完成标记: '{marker}'")
                return True
        return False
    
    def _extract_stage(self, content: str, iteration: int) -> Optional[ResearchStage]:
        """
        提取研究阶段。
        
        对应前端 Ask.tsx 中的 extractResearchStage() 函数（lines 212-253）。
        从 LLM 返回内容中提取 plan/update/conclusion 阶段。
        """
        # 检测阶段类型
        stage_type = "update"
        title = f"研究更新 #{iteration}"
        
        if iteration == 1:
            # 第一轮通常是研究计划
            if "## Research Plan" in content or "## 研究计划" in content:
                stage_type = "plan"
                title = "研究计划"
        elif iteration >= self.MAX_ITERATIONS or self._check_if_complete(content):
            stage_type = "conclusion"
            title = "最终结论"
        
        # 提取标题行后的内容作为阶段内容
        # 简单实现：使用整个内容
        return ResearchStage(
            title=title,
            content=content,
            iteration=iteration,
            type=stage_type,
        )
    
    async def research(self, query: str) -> str:
        """
        执行深度研究。
        
        对应前端 Ask.tsx 中的完整深度研究流程：
          1. handleConfirmAsk() — 发送初始请求（带 [DEEP RESEARCH] 标记）
          2. continueResearch() — 自动继续研究（最多 5 轮）
          3. checkIfResearchComplete() — 每轮检测是否完成
        
        对应后端 api/simple_chat.py 中的 _handle_deep_research() 流程。
        """
        logger.info("\n" + "=" * 60)
        logger.info("DeepResearchFlow.research()")
        logger.info("=" * 60)
        logger.info(f"研究问题: {query}")
        logger.info(f"最大迭代: {self.MAX_ITERATIONS}")
        
        # 初始化 RAG 检索器
        if not self.retriever:
            self._init_retriever()
        
        # 构建初始 RAG 上下文
        context = self._build_context(query)
        
        # 迭代研究循环
        for iteration in range(1, self.MAX_ITERATIONS + 1):
            self.current_iteration = iteration
            logger.info(f"\n{'─' * 50}")
            logger.info(f"迭代 {iteration}/{self.MAX_ITERATIONS}")
            logger.info(f"{'─' * 50}")
            
            # 构建研究 prompt
            messages = self._build_research_prompt(query, iteration, context)
            
            # 如果有对话历史，添加到 messages 中
            for msg in self.messages:
                messages.append({"role": msg.role, "content": msg.content})
            
            # 调用 LLM
            logger.info(f"调用 LLM ({self.provider}/{self.model})...")
            full_response = ""
            try:
                full_response = await _call_llm_and_collect(self.provider, self.model, messages)
                logger.info(f"✓ 响应 ({len(full_response)} 字符)")
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                full_response = f"[错误] LLM 调用失败: {e}"
            
            # 记录消息
            self.messages.append(Message(role="assistant", content=full_response))
            
            # 提取研究阶段
            stage = self._extract_stage(full_response, iteration)
            if stage:
                self.research_stages.append(stage)
                logger.info(f"  阶段: {stage.type} - {stage.title}")
            
            # 检测是否完成
            if self._check_if_complete(full_response):
                logger.info(f"✓ 研究在第 {iteration} 轮完成")
                self.is_complete = True
                self.final_answer = full_response
                break
            
            # 如果不是最后一轮，准备继续研究
            if iteration < self.MAX_ITERATIONS:
                # 对应前端 Ask.tsx 中 continueResearch() 的逻辑：
                # 添加 "[DEEP RESEARCH] Continue the research" 到消息历史
                continue_prompt = "[DEEP RESEARCH] Continue the research"
                self.messages.append(Message(role="user", content=continue_prompt))
                logger.info("  准备继续下一轮研究...")
        
        if not self.is_complete:
            logger.info(f"达到最大迭代次数 ({self.MAX_ITERATIONS})，研究结束")
            self.final_answer = self.messages[-1].content if self.messages else ""
        
        logger.info(f"\n✓ 深度研究完成")
        logger.info(f"总迭代: {self.current_iteration}")
        logger.info(f"阶段数: {len(self.research_stages)}")
        
        return self.final_answer
    
    def print_research_summary(self) -> None:
        """打印研究结果摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("深度研究 — 结果摘要")
        logger.info("=" * 60)
        
        logger.info(f"研究问题: {self.messages[0].content if self.messages else 'N/A'}")
        logger.info(f"总迭代: {self.current_iteration}/{self.MAX_ITERATIONS}")
        logger.info(f"是否完成: {'是' if self.is_complete else '否'}")
        logger.info(f"")
        
        logger.info("研究阶段:")
        for i, stage in enumerate(self.research_stages, 1):
            icon = {"plan": "📋", "update": "🔄", "conclusion": "✅"}.get(stage.type, "📝")
            logger.info(f"  {icon} [{i}] {stage.title} (迭代 {stage.iteration})")
            preview = stage.content[:150]
            if len(stage.content) > 150:
                preview += "..."
            logger.info(f"     {preview}")
        
        logger.info(f"")
        logger.info(f"最终回答长度: {len(self.final_answer)} 字符")
        logger.info(f"提供者: {self.provider}/{self.model}")
        logger.info(f"仓库: {self.repo_url}")


# ══════════════════════════════════════════════════════════════════════════
# Part 6: 主入口
# ══════════════════════════════════════════════════════════════════════════

async def run_wiki_mode(args: Any) -> None:
    """运行 Wiki 生成模式"""
    logger.info("=" * 70)
    logger.info("模式: Wiki 文档生成")
    logger.info("=" * 70)
    
    flow = WikiGenerationFlow(
        repo_url=args.repo_url,
        provider=args.provider,
        model=args.model,
        language=args.language,
        comprehensive=not args.concise,
        use_database=not args.no_db,
    )
    
    # 步骤 1: 获取仓库结构
    flow.fetch_repository_structure()
    
    # 步骤 2: 确定 Wiki 结构
    await flow.determine_wiki_structure()
    
    # 步骤 3: 生成所有页面
    await flow._generate_all_pages()
    
    # 打印摘要
    flow.print_summary()


async def run_chat_mode(args: Any) -> None:
    """运行 Q&A 聊天模式"""
    logger.info("=" * 70)
    logger.info("模式: 用户 Q&A 简单聊天")
    logger.info("=" * 70)
    
    flow = SimpleChatFlow(
        repo_url=args.repo_url,
        provider=args.provider,
        model=args.model,
        language=args.language,
        use_database=not args.no_db,
    )
    
    answer = await flow.chat(args.query)
    
    logger.info("\n" + "=" * 60)
    logger.info("回答:")
    logger.info("=" * 60)
    logger.info(answer)
    
    flow.print_conversation()


async def run_research_mode(args: Any) -> None:
    """运行深度研究模式"""
    logger.info("=" * 70)
    logger.info("模式: 深度研究")
    logger.info("=" * 70)
    
    flow = DeepResearchFlow(
        repo_url=args.repo_url,
        provider=args.provider,
        model=args.model,
        language=args.language,
        use_database=not args.no_db,
    )
    
    final_answer = await flow.research(args.query)
    
    logger.info("\n" + "=" * 60)
    logger.info("最终研究结果:")
    logger.info("=" * 60)
    logger.info(final_answer)
    
    flow.print_research_summary()


def parse_args() -> Any:
    """解析命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DeepWiki-open 业务逻辑流独立调试程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Wiki 生成
  python scripts/simulate/debug_flow.py --mode wiki --repo-url https://github.com/user/repo
  
  # Q&A 聊天
  python scripts/simulate/debug_flow.py --mode chat --repo-url https://github.com/user/repo --query "如何配置项目？"
  
  # 深度研究
  python scripts/simulate/debug_flow.py --mode research --repo-url https://github.com/user/repo --query "架构设计原理"
  
  # 使用样本数据（不连接数据库）
  python scripts/simulate/debug_flow.py --mode wiki --repo-url https://github.com/user/repo --no-db
        """,
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["wiki", "chat", "research"],
        default="wiki",
        help="运行模式: wiki (Wiki生成), chat (Q&A聊天), research (深度研究)",
    )
    
    parser.add_argument(
        "--repo-url", "-u",
        type=str,
        default="https://github.com/Xihaixin/MathModelAgent",
        help="仓库 URL",
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="这个项目的主要功能是什么？",
        help="查询问题（chat/research 模式）",
    )
    
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="dashscope",
        help="LLM 提供者 (google, openai, dashscope, ollama 等)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-plus",
        help="LLM 模型名称",
    )
    
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="zh",
        help="语言代码 (zh, en 等)",
    )
    
    parser.add_argument(
        "--concise", "-c",
        action="store_true",
        help="使用简洁模式（仅 wiki 模式）",
    )
    
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="不使用数据库，使用 fixtures 样本数据",
    )
    
    return parser.parse_args()


async def main() -> None:
    """主入口"""
    args = parse_args()
    
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  DeepWiki-open 业务逻辑流独立调试程序")
    logger.info("║  Debug Flow for DeepWiki-open Business Logic")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info(f"启动时间: {datetime.now().isoformat()}")
    logger.info(f"模式: {args.mode}")
    logger.info(f"仓库: {args.repo_url}")
    logger.info(f"提供者: {args.provider}/{args.model}")
    logger.info(f"语言: {args.language}")
    logger.info(f"使用数据库: {not args.no_db}")
    logger.info("")
    
    if args.mode == "wiki":
        await run_wiki_mode(args)
    elif args.mode == "chat":
        await run_chat_mode(args)
    elif args.mode == "research":
        await run_research_mode(args)
    
    logger.info("\n" + "=" * 70)
    logger.info("程序执行完毕")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
