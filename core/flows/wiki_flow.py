"""
wiki_flow.py — Wiki 文档生成流
===============================

完整复现前端 page.tsx 中的 Wiki 生成逻辑。

流程:
  1. fetch_repository_structure()  → 获取文件树和 README
  2. determine_wiki_structure()    → 调用 LLM 确定 Wiki 结构
  3. _generate_all_pages()         → 逐页生成 Wiki 页面内容
  4. _save_to_database()           → 持久化到 wiki_pages 表
  5. print_summary()               → 打印结果摘要

依赖:
  - core.flows.base — BaseFlow 公共基类
  - core.models — WikiPage, WikiSection, WikiStructure
  - api.prompts — Prompt 模板
  - rag_optimizer.db.repository — DocumentRepository, WikiPageRepository
  - core.ingestion.ingestor — DataIngestor（自动触发）
"""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from core.flows.base import (
    BaseFlow,
    call_llm_and_collect,
    generate_file_url,
    get_cache_key,
    parse_repo_url,
    load_configs,
)
from core.models import WikiPage, WikiSection, WikiStructure
from rag_optimizer.db.repository import DocumentRepository, WikiPageRepository
from rag_optimizer.integration.deepwiki_adapter import PgvectorRetriever

# 数据摄取（自动触发）
from core.ingestion.ingestor import DataIngestor

logger = logging.getLogger("core.flows.wiki")


class WikiGenerationFlow(BaseFlow):
    """
    Wiki 文档生成流 — 完整复现前端 page.tsx 中的 Wiki 生成逻辑。

    流程:
      1. fetch_repository_structure()  → 获取文件树和 README
      2. determine_wiki_structure()    → 调用 LLM 确定 Wiki 结构
      3. _generate_all_pages()         → 逐页生成 Wiki 页面内容
      4. _save_to_database()           → 持久化到 wiki_pages 表
      5. print_summary()               → 打印结果摘要

    对应前端 page.tsx 中的:
      - fetchRepositoryStructure()
      - determineWikiStructure()
      - generatePageContent()
    """

    def __init__(
        self,
        repo_url: str,
        provider: str = "dashscope",
        model: str = "qwen-plus",
        language: str = "zh",
        comprehensive: bool = True,
        use_database: bool = True,
        local_path: Optional[str] = None,
    ):
        # 调用 BaseFlow.__init__ 初始化公共属性
        super().__init__(
            repo_url=repo_url,
            provider=provider,
            model=model,
            language=language,
            use_database=use_database,
        )

        self.comprehensive = comprehensive
        self.local_path = local_path

        # Wiki 生成状态
        self.file_tree: Optional[str] = None
        self.readme: Optional[str] = None
        self.wiki_structure: Optional[WikiStructure] = None
        self.generated_pages: Dict[str, str] = {}  # page_id → content

        # 缓存键（对应前端 getCacheKey）
        self.cache_key = get_cache_key(
            self.owner, self.repo, self.repo_type, self.language, self.comprehensive
        )

        logger.info(f"初始化 WikiGenerationFlow:")
        logger.info(f"  仓库: {repo_url}")
        logger.info(f"  本地路径: {local_path}")
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
          1. use_database=True  → 从 PostgreSQL 数据库查询已有数据；
             如果数据库无数据，自动触发 DataIngestor 摄取管道
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

            # ── 数据库无数据，自动触发数据摄取管道 ──
            logger.warning("数据库中无此项目数据，自动触发数据摄取 (DataIngestor)...")
            logger.info("=" * 50)

            ingestor = DataIngestor(
                repo_url=self.repo_url,
                repo_type=self.repo_type,
                access_token=None,
                local_path=self.local_path,
            )
            project_id = ingestor.run()

            if not project_id:
                logger.error("❌ 数据摄取失败，无法继续 Wiki 生成")
                raise RuntimeError(
                    f"数据摄取失败 (repo_url={self.repo_url}, local_path={self.local_path})。"
                    f"请先确保数据摄取已完成。"
                )

            self.project_id = project_id
            logger.info(f"✓ 数据摄取完成 (project_id={project_id})")
            logger.info("=" * 50)

            # 重新从数据库获取仓库结构
            logger.info("重新从数据库获取仓库结构...")
            result = self._fetch_from_database()
            if result is not None:
                self.file_tree, self.readme = result
                logger.info(f"✓ 从数据库获取文件树 ({len(self.file_tree)} 字符)")
                logger.info(f"✓ 从数据库获取 README ({len(self.readme)} 字符)")
                return self.file_tree, self.readme
            else:
                logger.error("❌ 数据摄取后仍无法从数据库获取仓库结构")
                raise RuntimeError("数据摄取后数据库查询仍然失败")

        # 没有数据库回退，抛出异常
        raise RuntimeError(
            f"数据库中无此项目数据 (repo_url={self.repo_url})，"
            f"且 use_database=False 模式已不再支持样本数据回退。"
            f"请确保数据摄取已完成。"
        )

    def _fetch_from_database(self) -> Optional[Tuple[str, str]]:
        """
        从 PostgreSQL 数据库查询项目数据。

        对应后端 api/api.py 中的 /local_repo/structure 端点逻辑。
        使用 ProjectRepository 和 DocumentRepository 获取数据。
        """
        try:
            # 查找项目
            if not self.project_id:
                self._find_project_id()

            if not self.project_id:
                logger.warning(f"未找到匹配的项目: {self.repo_url}")
                return None

            logger.info(f"  项目 ID: {self.project_id}")

            # 获取文档列表
            documents = DocumentRepository.get_by_project(self.project_id)
            logger.info(f"  项目文档数: {len(documents)}")

            if not documents:
                logger.warning("数据库中没有文档记录")
                return None

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
            full_response = await call_llm_and_collect(self.provider, self.model, messages)
            logger.info(f"✓ LLM 返回响应 ({len(full_response)} 字符)")
        except Exception as e:
            logger.error(f"LLM 调用失败，无法生成 Wiki 结构: {e}")
            raise

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
            full_response = await call_llm_and_collect(self.provider, self.model, messages)
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
            # 直接回退到默认结构
            logger.info("使用默认 Wiki 结构作为回退")
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
            full_content = await call_llm_and_collect(self.provider, self.model, messages)
        except Exception as e:
            logger.error(f"LLM 调用失败，无法生成页面内容: {e}")
            raise

        # 清理 markdown 代码块分隔符（对应前端 line 645 的 cleanMarkdownDelimiters）
        full_content = self._clean_markdown_delimiters(full_content)

        return full_content

    def _init_retriever(self, top_k: int = 10) -> Optional[PgvectorRetriever]:
        """
        初始化 RAG 检索器。

        使用项目中已有的 PgvectorRetriever 进行混合检索。
        对应原始项目 websocket_wiki.py 中 request_rag.prepare_retriever() 的逻辑。
        """
        if not self.use_database:
            logger.info("  跳过 RAG 检索器初始化（use_database=False）")
            return None

        if not self.project_id:
            logger.warning("  project_id 为空，无法初始化检索器")
            return None

        try:
            retriever = PgvectorRetriever(
                project_id=self.project_id,
                retrieval_type="hybrid",
                top_k=top_k,
            )
            logger.info(f"  ✓ RAG 检索器已初始化 (project_id={self.project_id})")
            return retriever
        except Exception as e:
            logger.warning(f"  初始化 RAG 检索器失败: {e}", exc_info=True)
            return None

    def _fetch_file_contents(self, page: WikiPage) -> Dict[str, str]:
        """
        从数据库获取页面相关文件的完整内容。

        对应原始项目 websocket_wiki.py 中 get_file_content() 的逻辑（lines 401-409）：
          file_content = get_file_content(request.repo_url, request.filePath, ...)

        使用项目中已有的 DocumentRepository.get_by_project() 获取所有文档，
        然后按 file_path 匹配 page.filePaths 中的路径，提取文件内容。

        Returns:
            Dict[str, str] — file_path → content 的映射
        """
        if not self.use_database or not self.project_id:
            return {}

        try:
            documents = DocumentRepository.get_by_project(self.project_id)
            if not documents:
                logger.info("  数据库中没有文档记录")
                return {}

            # 建立 file_path → content 索引
            doc_map: Dict[str, str] = {}
            for doc in documents:
                fp = doc.get("file_path", "")
                content = doc.get("content", "")
                if fp and content:
                    doc_map[fp] = content

            # 匹配页面相关文件
            file_contents: Dict[str, str] = {}
            for fp in page.filePaths:
                # 精确匹配
                if fp in doc_map:
                    file_contents[fp] = doc_map[fp]
                    logger.info(f"  ✓ 获取文件内容: {fp} ({len(doc_map[fp])} 字符)")
                else:
                    # 模糊匹配（路径后缀或包含）
                    matched = False
                    for db_fp, content in doc_map.items():
                        if db_fp.endswith(fp) or fp in db_fp:
                            file_contents[db_fp] = content
                            logger.info(f"  ✓ 模糊匹配文件内容: {db_fp} ({len(content)} 字符)")
                            matched = True
                            break
                    if not matched:
                        logger.info(f"  - 未找到文件: {fp}")

            return file_contents

        except Exception as e:
            logger.warning(f"  获取文件内容失败: {e}")
            return {}

    def _build_page_prompt(self, page: WikiPage) -> str:
        """
        构建单个页面的生成 prompt。

        对应原始项目 websocket_wiki.py 中 handle_websocket_chat() 的 prompt 构建逻辑（lines 418-438）。

        原始项目的 prompt 结构：
          /no_think {system_prompt}
          <conversation_history>...</conversation_history>
          <currentFileContent path="src/rag.py">{完整文件代码}</currentFileContent>
          <START_OF_CONTEXT>
          ## File Path: src/rag.py
          {RAG 检索到的代码片段}
          <END_OF_CONTEXT>
          <query>{用户问题}</query>
          Assistant:

        本实现使用项目中已有的组件：
          1. DocumentRepository — 获取文件完整内容 → <currentFileContent>
          2. PgvectorRetriever — RAG 检索相关代码片段 → <START_OF_CONTEXT>
        """
        # ── 1. 获取文件完整内容（对应原始项目的 get_file_content） ──────────
        file_contents = self._fetch_file_contents(page)

        file_content_blocks = ""
        for fp, content in file_contents.items():
            file_content_blocks += (
                f"<currentFileContent path=\"{fp}\">\n"
                f"{content}\n"
                f"</currentFileContent>\n\n"
            )

        # ── 2. RAG 检索相关代码片段（对应原始项目的 request_rag(rag_query)） ──
        context_text = ""
        if self.use_database and self.project_id:
            try:
                retriever = self._init_retriever()
                if retriever:
                    # 使用页面标题和文件路径作为 RAG 查询
                    rag_query = f"Contexts related to {page.title}: {', '.join(page.filePaths)}"
                    logger.info(f"  RAG 查询: {rag_query}")

                    # 使用 PgvectorRetriever.search() 原生接口（返回 (List[RetrievalResult], RetrievalStats)）
                    results, stats = retriever.search(rag_query, top_k=10)

                    if results:
                        # 按 file_path 分组（对应原始项目 lines 216-234）
                        docs_by_file: Dict[str, List] = {}
                        for r in results:
                            file_path = getattr(r, "file_path", "unknown") or "unknown"
                            if file_path not in docs_by_file:
                                docs_by_file[file_path] = []
                            docs_by_file[file_path].append(r)

                        # 格式化为 <START_OF_CONTEXT> 块
                        context_parts = []
                        for file_path, docs in docs_by_file.items():
                            header = f"## File Path: {file_path}\n\n"
                            contents = []
                            for d in docs:
                                content = getattr(d, "content", "") or getattr(d, "text", "") or ""
                                if content:
                                    contents.append(content)
                            if contents:
                                context_parts.append(f"{header}{chr(10).join(contents)}")

                        if context_parts:
                            context_text = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
                            logger.info(f"  ✓ RAG 检索到 {len(results)} 个结果，来自 {len(docs_by_file)} 个文件")
                    else:
                        # 区分"检索执行成功但无结果"和"检索执行失败"
                        if stats and stats.total_results == 0 and stats.latency_ms > 0:
                            logger.info(f"  RAG 检索完成（{stats.latency_ms}ms），但未找到相关结果")
                        else:
                            logger.info("  RAG 检索未返回结果")
                else:
                    logger.warning("  RAG 检索器初始化失败（retriever is None），跳过 RAG")
            except Exception as e:
                logger.warning(f"  RAG 检索失败: {e}", exc_info=True)

        # ── 3. 构建文件路径列表（带 URL） ──────────────────────────────────
        file_paths_str = ""
        for fp in page.filePaths:
            url = generate_file_url(fp, self.repo_url, self.repo_type)
            file_paths_str += f"  - {fp}\n    URL: {url}\n"

        # ── 4. 构建相关页面列表 ────────────────────────────────────────────
        related_str = ""
        if self.wiki_structure:
            for rp_id in page.relatedPages:
                rp_title = rp_id
                for p in self.wiki_structure.pages:
                    if p.id == rp_id:
                        rp_title = p.title
                        break
                related_str += f"  - [{rp_title}]({rp_id})\n"

        # ── 5. 组装最终 prompt（匹配原始项目的结构 lines 418-438） ──────────
        system_prompt = (
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

        # 构建最终 prompt（匹配原始项目的结构）
        prompt = f"/no_think {system_prompt}\n\n"

        # 注入文件完整内容（对应原始项目的 <currentFileContent>）
        if file_content_blocks:
            prompt += file_content_blocks

        # 注入 RAG 上下文（对应原始项目的 <START_OF_CONTEXT>）
        CONTEXT_START = "<START_OF_CONTEXT>"
        CONTEXT_END = "<END_OF_CONTEXT>"
        if context_text.strip():
            prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
        else:
            prompt += "<note>Answering without retrieval augmentation.</note>\n\n"

        # 添加查询
        prompt += f"<query>\nGenerate the wiki page for: {page.title}\n</query>\n\nAssistant: "

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

    # ── 步骤 4: 保存到数据库 ──────────────────────────────────────────────

    def _save_to_database(self) -> int:
        """
        将生成的 Wiki 页面持久化到 wiki_pages 表。

        使用 WikiPageRepository.upsert() 写入每条页面记录，
        (project_id, page_slug, language) 唯一约束确保幂等性。

        Returns:
            int: 保存的页面数
        """
        if not self.use_database:
            logger.info("跳过数据库保存（use_database=False）")
            return 0

        if not self.project_id:
            logger.warning("project_id 为空，无法保存 Wiki 页面到数据库")
            return 0

        if not self.generated_pages:
            logger.warning("没有已生成的页面，跳过数据库保存")
            return 0

        if not self.wiki_structure or not self.wiki_structure.pages:
            logger.warning("wiki_structure 为空，跳过数据库保存")
            return 0

        saved_count = 0
        logger.info("\n" + "=" * 60)
        logger.info("步骤 4: 保存 Wiki 页面到数据库 (_save_to_database)")
        logger.info("=" * 60)

        for page in self.wiki_structure.pages:
            content_md = self.generated_pages.get(page.id)
            if not content_md:
                logger.warning(f"  跳过页面 '{page.title}'（无内容）")
                continue

            try:
                page_id = WikiPageRepository.upsert(
                    project_id=self.project_id,
                    page_slug=page.id,
                    title=page.title,
                    content_md=content_md,
                    language=self.language,
                    is_comprehensive=self.comprehensive,
                    provider=self.provider,
                    model=self.model,
                    source_chunks=None,  # 暂不记录来源分块
                )
                saved_count += 1
                logger.info(f"  ✓ 已保存: {page.title} ({page.id}) → id={page_id}")
            except Exception as e:
                logger.error(f"  ✗ 保存失败: {page.title} ({page.id}): {e}")

        logger.info(f"\n✓ 共保存 {saved_count}/{len(self.wiki_structure.pages)} 个页面到 wiki_pages 表")
        return saved_count

    # ── 步骤 5: 打印结果摘要 ──────────────────────────────────────────────

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

    # ── 主入口 ────────────────────────────────────────────────────────────

    async def run(self) -> Dict[str, Any]:
        """
        执行 Wiki 生成流程主入口。

        依次执行 5 个步骤：
          1. fetch_repository_structure()
          2. determine_wiki_structure()
          3. _generate_all_pages()
          4. _save_to_database()
          5. print_summary()

        Returns:
            Dict with keys: wiki_structure, generated_pages, saved_count
        """
        self.fetch_repository_structure()
        await self.determine_wiki_structure()
        await self._generate_all_pages()
        saved = self._save_to_database()
        self.print_summary()

        return {
            "wiki_structure": self.wiki_structure,
            "generated_pages": self.generated_pages,
            "saved_count": saved,
        }