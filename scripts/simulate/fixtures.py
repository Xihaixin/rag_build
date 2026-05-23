"""
统一样例负载 — 为后端模拟验证和独立调试程序提供可复用的数据

包含：
- Wiki 缓存样例（WikiStructureModel、WikiPage、WikiCacheRequest）
- 聊天消息样例（ChatCompletionRequest payload）
- WebSocket 深度研究样例
- 本地仓库路径样例
- 文件树和 README 样例
- XML Wiki 结构样例
- 生成页面内容样例
"""

from typing import Any, Dict, List, Optional

# ============================================================
# Wiki 缓存样例
# ============================================================

SAMPLE_WIKI_STRUCTURE: Dict[str, Any] = {
    "id": "sample-repo-wiki",
    "title": "Sample Repository Wiki",
    "description": "A sample wiki structure for testing the API",
    "pages": [
        {
            "id": "page-introduction",
            "title": "Introduction",
            "content": "This is the introduction page for the sample repository.\n\n## Overview\n\nThe sample repository demonstrates the wiki generation feature.",
            "filePaths": ["README.md"],
            "importance": "high",
            "relatedPages": ["page-installation", "page-usage"],
        },
        {
            "id": "page-installation",
            "title": "Installation",
            "content": "## Installation Guide\n\nTo install the project:\n\n```bash\npip install sample-project\n```\n\n### Requirements\n\n- Python 3.12+\n- PostgreSQL 15+ with pgvector extension",
            "filePaths": ["docs/install.md"],
            "importance": "high",
            "relatedPages": ["page-introduction"],
        },
        {
            "id": "page-usage",
            "title": "Usage",
            "content": "## Usage Examples\n\n### Basic Usage\n\n```python\nfrom sample import Client\n\nclient = Client()\nresult = client.process()\nprint(result)\n```\n\n### Advanced Configuration\n\nSee the configuration guide for more details.",
            "filePaths": ["docs/usage.md"],
            "importance": "medium",
            "relatedPages": ["page-installation"],
        },
        {
            "id": "page-api",
            "title": "API Reference",
            "content": "## API Reference\n\n### `Client.process()`\n\nProcesses the input data and returns results.\n\n**Parameters:**\n- `input_data` (str): The input data to process\n- `config` (dict, optional): Configuration options\n\n**Returns:**\n- `dict`: Processed results",
            "filePaths": ["src/client.py"],
            "importance": "medium",
            "relatedPages": ["page-usage"],
        },
    ],
    "sections": [
        {"id": "sec-getting-started", "title": "Getting Started", "pages": ["page-introduction", "page-installation"]},
        {"id": "sec-guides", "title": "Guides", "pages": ["page-usage"]},
        {"id": "sec-reference", "title": "Reference", "pages": ["page-api"]},
    ],
    "rootSections": ["sec-getting-started", "sec-guides", "sec-reference"],
}

SAMPLE_GENERATED_PAGES: Dict[str, Any] = {
    "page-introduction": SAMPLE_WIKI_STRUCTURE["pages"][0],
    "page-installation": SAMPLE_WIKI_STRUCTURE["pages"][1],
    "page-usage": SAMPLE_WIKI_STRUCTURE["pages"][2],
    "page-api": SAMPLE_WIKI_STRUCTURE["pages"][3],
}

SAMPLE_REPO_INFO: Dict[str, Any] = {
    "owner": "test-owner",
    "repo": "sample-repo",
    "type": "github",
    "token": None,
    "localPath": None,
    "repoUrl": "https://github.com/test-owner/sample-repo",
}

SAMPLE_WIKI_CACHE_REQUEST: Dict[str, Any] = {
    "repo": SAMPLE_REPO_INFO,
    "language": "en",
    "wiki_structure": SAMPLE_WIKI_STRUCTURE,
    "generated_pages": SAMPLE_GENERATED_PAGES,
    "provider": "dashscope",
    "model": "qwen-plus",
}

SAMPLE_WIKI_EXPORT_REQUEST: Dict[str, Any] = {
    "repo_url": "https://github.com/test-owner/sample-repo",
    "pages": SAMPLE_WIKI_STRUCTURE["pages"],
    "format": "markdown",
}

SAMPLE_WIKI_EXPORT_REQUEST_JSON: Dict[str, Any] = {
    "repo_url": "https://github.com/test-owner/sample-repo",
    "pages": SAMPLE_WIKI_STRUCTURE["pages"],
    "format": "json",
}


# ============================================================
# 聊天消息样例
# ============================================================

SAMPLE_CHAT_MESSAGES: List[Dict[str, str]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is this project about?"},
]

SAMPLE_CHAT_REQUEST_NO_REPO: Dict[str, Any] = {
    "messages": SAMPLE_CHAT_MESSAGES,
    "provider": "dashscope",
    "model": "qwen-plus",
    "stream": True,
    "language": "en",
}

SAMPLE_CHAT_REQUEST_WITH_REPO: Dict[str, Any] = {
    "messages": SAMPLE_CHAT_MESSAGES,
    "provider": "dashscope",
    "model": "qwen-plus",
    "stream": True,
    "repo_url": "https://github.com/test-owner/sample-repo",
    "repo_type": "github",
    "language": "en",
    "deep_research": False,
}


# ============================================================
# 深度研究样例（WebSocket）
# ============================================================

SAMPLE_DEEP_RESEARCH_REQUEST: Dict[str, Any] = {
    "repo_url": "https://github.com/test-owner/sample-repo",
    "type": "github",
    "token": None,
    "provider": "dashscope",
    "model": "qwen-plus",
    "language": "en",
    "query": "Explain the architecture of this project in detail.",
    "filePath": None,
    "deep_research": True,
    "research_iterations": 3,
    "excluded_dirs": None,
    "excluded_files": None,
    "included_dirs": None,
    "included_files": None,
}

SAMPLE_SIMPLE_CHAT_WS_REQUEST: Dict[str, Any] = {
    "repo_url": "https://github.com/test-owner/sample-repo",
    "type": "github",
    "token": None,
    "provider": "dashscope",
    "model": "qwen-plus",
    "language": "en",
    "query": "What is the main functionality of this project?",
    "filePath": None,
    "deep_research": False,
    "research_iterations": 3,
    "excluded_dirs": None,
    "excluded_files": None,
    "included_dirs": None,
    "included_files": None,
}


# ============================================================
# 本地仓库路径样例
# ============================================================

# 使用当前工作区自身作为测试仓库路径
SAMPLE_LOCAL_REPO_PATH: str = r"D:\ProgramFile2_OR\Python_Study_System\OpenStudy\rag_build"
SAMPLE_INVALID_PATH: str = r"D:\nonexistent\path"


# ============================================================
# 文件树和 README 样例（用于 Wiki 生成流程）
# ============================================================

SAMPLE_FILE_TREE: str = """README.md
pyproject.toml
src/main.py
src/client.py
src/utils/helpers.py
src/models/user.py
src/models/__init__.py
docs/install.md
docs/usage.md
docs/api.md
tests/test_main.py
tests/test_client.py
tests/conftest.py
config/settings.yaml
config/logging.yaml
scripts/setup.sh
scripts/deploy.sh"""

SAMPLE_README: str = """# Sample Repository

A sample project for demonstrating wiki generation.

## Features

- Feature 1: Description of feature 1
- Feature 2: Description of feature 2
- Feature 3: Description of feature 3

## Installation

```bash
pip install sample-project
```

## Usage

```python
from sample import Client
client = Client()
result = client.process()
```

## License

MIT License
"""


# ============================================================
# XML Wiki 结构样例（用于 LLM 返回模拟）
# ============================================================

SAMPLE_WIKI_STRUCTURE_XML: str = """<wiki_structure>
  <title>Sample Repository Wiki</title>
  <description>A comprehensive wiki for the sample repository</description>
  <sections>
    <section id="sec-overview">
      <title>Overview</title>
      <pages>
        <page_ref>page-introduction</page_ref>
        <page_ref>page-installation</page_ref>
      </pages>
    </section>
    <section id="sec-architecture">
      <title>System Architecture</title>
      <pages>
        <page_ref>page-architecture</page_ref>
        <page_ref>page-data-flow</page_ref>
      </pages>
    </section>
    <section id="sec-guides">
      <title>Guides</title>
      <pages>
        <page_ref>page-usage</page_ref>
        <page_ref>page-api</page_ref>
      </pages>
    </section>
  </sections>
  <pages>
    <page id="page-introduction">
      <title>Introduction</title>
      <description>Overview of the project and its purpose</description>
      <importance>high</importance>
      <relevant_files>
        <file_path>README.md</file_path>
        <file_path>pyproject.toml</file_path>
      </relevant_files>
      <related_pages>
        <related>page-installation</related>
      </related_pages>
      <parent_section>sec-overview</parent_section>
    </page>
    <page id="page-installation">
      <title>Installation Guide</title>
      <description>How to install and set up the project</description>
      <importance>high</importance>
      <relevant_files>
        <file_path>docs/install.md</file_path>
        <file_path>pyproject.toml</file_path>
        <file_path>scripts/setup.sh</file_path>
      </relevant_files>
      <related_pages>
        <related>page-introduction</related>
        <related>page-usage</related>
      </related_pages>
      <parent_section>sec-overview</parent_section>
    </page>
    <page id="page-architecture">
      <title>System Architecture</title>
      <description>Overall architecture and design patterns</description>
      <importance>high</importance>
      <relevant_files>
        <file_path>src/main.py</file_path>
        <file_path>src/client.py</file_path>
        <file_path>src/utils/helpers.py</file_path>
      </relevant_files>
      <related_pages>
        <related>page-data-flow</related>
      </related_pages>
      <parent_section>sec-architecture</parent_section>
    </page>
    <page id="page-data-flow">
      <title>Data Flow</title>
      <description>How data flows through the system</description>
      <importance>medium</importance>
      <relevant_files>
        <file_path>src/client.py</file_path>
        <file_path>src/models/user.py</file_path>
        <file_path>config/settings.yaml</file_path>
      </relevant_files>
      <related_pages>
        <related>page-architecture</related>
      </related_pages>
      <parent_section>sec-architecture</parent_section>
    </page>
    <page id="page-usage">
      <title>Usage Guide</title>
      <description>How to use the project</description>
      <importance>medium</importance>
      <relevant_files>
        <file_path>docs/usage.md</file_path>
        <file_path>src/client.py</file_path>
        <file_path>tests/test_client.py</file_path>
      </relevant_files>
      <related_pages>
        <related>page-api</related>
      </related_pages>
      <parent_section>sec-guides</parent_section>
    </page>
    <page id="page-api">
      <title>API Reference</title>
      <description>Complete API documentation</description>
      <importance>medium</importance>
      <relevant_files>
        <file_path>src/client.py</file_path>
        <file_path>src/models/user.py</file_path>
        <file_path>src/utils/helpers.py</file_path>
      </relevant_files>
      <related_pages>
        <related>page-usage</related>
      </related_pages>
      <parent_section>sec-guides</parent_section>
    </page>
  </pages>
</wiki_structure>"""


# ============================================================
# 生成页面内容样例（用于 LLM 调用失败时的 fallback）
# ============================================================

SAMPLE_GENERATED_PAGE_CONTENT: str = """<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](https://github.com/test-owner/sample-repo/blob/main/README.md)
- [src/main.py](https://github.com/test-owner/sample-repo/blob/main/src/main.py)
- [src/client.py](https://github.com/test-owner/sample-repo/blob/main/src/client.py)
- [src/utils/helpers.py](https://github.com/test-owner/sample-repo/blob/main/src/utils/helpers.py)
- [config/settings.yaml](https://github.com/test-owner/sample-repo/blob/main/config/settings.yaml)

</details>

# Page Title

## Introduction

This page provides a comprehensive overview of the component within the sample repository.

## Architecture

The component follows a modular architecture pattern:

```mermaid
graph TD
    A[Main Entry] --> B[Client Module]
    B --> C[Helper Utilities]
    B --> D[Data Models]
    C --> E[External Services]
```

## Key Components

| Component | Description | File |
|-----------|-------------|------|
| Client | Main client interface | src/client.py |
| Helpers | Utility functions | src/utils/helpers.py |
| Models | Data models | src/models/ |

## Usage Example

```python
from sample import Client

client = Client(config={"timeout": 30})
result = client.process()
print(result)
```

## Summary

This component is a core part of the system, providing essential functionality for data processing and external service integration.
"""


# ============================================================
# 辅助函数
# ============================================================


def get_sample_wiki_cache_request() -> Dict[str, Any]:
    """获取 Wiki 缓存请求的深拷贝"""
    import copy
    return copy.deepcopy(SAMPLE_WIKI_CACHE_REQUEST)


def get_sample_chat_request(repo_url: Optional[str] = None) -> Dict[str, Any]:
    """获取聊天请求，可选是否带 repo_url"""
    import copy
    if repo_url:
        req = copy.deepcopy(SAMPLE_CHAT_REQUEST_WITH_REPO)
        req["repo_url"] = repo_url
        return req
    return copy.deepcopy(SAMPLE_CHAT_REQUEST_NO_REPO)
