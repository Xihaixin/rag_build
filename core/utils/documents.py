"""
documents.py — 文档读取与 Token 计数工具
=========================================

提供从本地目录递归读取文档文件和 Token 计数功能。

依赖:
  - core.config — DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
"""

import logging
import os
from typing import Any, Dict, List, Optional

from core.config import DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES

logger = logging.getLogger("core.utils.documents")


# ══════════════════════════════════════════════════════════════════════════
# Token 计数
# ══════════════════════════════════════════════════════════════════════════


def count_tokens(text: str, embedder_type: Optional[str] = None, is_ollama_embedder: Optional[bool] = None) -> int:
    """
    计算文本的 token 数量。

    参数:
        text: 文本内容
        embedder_type: 嵌入器类型（未使用，保留接口兼容）
        is_ollama_embedder: 是否使用 Ollama 嵌入器（未使用，保留接口兼容）

    返回:
        int: token 数量
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # 回退：使用简单估算（约 4 字符/token）
        return len(text) // 4


# ══════════════════════════════════════════════════════════════════════════
# 文档读取
# ══════════════════════════════════════════════════════════════════════════


def read_all_documents(
    path: str,
    embedder_type: Optional[str] = None,
    is_ollama_embedder: Optional[bool] = None,
    excluded_dirs: Optional[List[str]] = None,
    excluded_files: Optional[List[str]] = None,
    included_dirs: Optional[List[str]] = None,
    included_files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    递归读取目录中的所有文档文件。

    参数:
        path: 目录路径
        embedder_type: 嵌入器类型（未使用，保留接口兼容）
        is_ollama_embedder: 是否使用 Ollama 嵌入器（未使用，保留接口兼容）
        excluded_dirs: 排除的目录列表
        excluded_files: 排除的文件列表
        included_dirs: 包含的目录列表
        included_files: 包含的文件列表

    返回:
        List[Dict]: 文档列表，每项包含 file_path, content, file_type
    """
    excluded_dirs = excluded_dirs or DEFAULT_EXCLUDED_DIRS
    excluded_files = excluded_files or DEFAULT_EXCLUDED_FILES

    documents: List[Dict[str, Any]] = []

    # 规范化排除目录
    normalized_excluded_dirs = []
    for d in excluded_dirs:
        d = d.strip("./").strip("/")
        if d:
            normalized_excluded_dirs.append(d)

    def should_process_file(
        file_path: str,
        use_inclusion: bool,
        included_dirs_list: List[str],
        included_files_list: List[str],
        excluded_dirs_list: List[str],
        excluded_files_list: List[str],
    ) -> bool:
        """判断文件是否应该被处理"""
        rel_path = os.path.relpath(file_path, path).replace("\\", "/")

        # 检查排除目录
        for excl_dir in excluded_dirs_list:
            if rel_path.startswith(excl_dir + "/") or rel_path == excl_dir:
                return False

        # 检查排除文件
        for excl_file in excluded_files_list:
            if excl_file.startswith("*."):
                # 通配符匹配
                ext = excl_file[1:]
                if rel_path.endswith(ext):
                    return False
            elif excl_file == os.path.basename(rel_path):
                return False

        # 包含模式
        if use_inclusion:
            in_included = False
            for inc_dir in included_dirs_list:
                if rel_path.startswith(inc_dir + "/") or rel_path == inc_dir:
                    in_included = True
                    break
            for inc_file in included_files_list:
                if inc_file == os.path.basename(rel_path):
                    in_included = True
                    break
            return in_included

        return True

    use_inclusion_mode = bool(included_dirs or included_files)
    included_dirs_list = included_dirs or []
    included_files_list = included_files or []

    # 支持的文件扩展名
    text_extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".hpp",
        ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".dart",
        ".md", ".mdx", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
        ".cfg", ".conf", ".xml", ".html", ".css", ".scss", ".less", ".sql",
        ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".dockerfile",
        ".gradle", ".sbt", ".clj", ".ex", ".exs", ".erl", ".hrl",
        ".lua", ".r", ".m", ".mm", ".pl", ".pm", ".t", ".pod",
        ".vue", ".svelte", ".astro", ".graphql", ".gql", ".proto",
        ".cmake", ".makefile", ".gnumakefile", ".dockerignore",
        ".env.example", ".env.sample",
    }

    for root, dirs, files in os.walk(path):
        # 过滤排除目录
        dirs[:] = [
            d for d in dirs
            if d not in normalized_excluded_dirs
            and not d.startswith(".")
        ]

        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, path).replace("\\", "/")

            # 检查是否应该处理
            if not should_process_file(
                file_path,
                use_inclusion_mode,
                included_dirs_list,
                included_files_list,
                normalized_excluded_dirs,
                excluded_files,
            ):
                continue

            # 检查文件扩展名
            ext = os.path.splitext(file)[1].lower()
            if ext not in text_extensions and file not in (
                "Dockerfile", "Makefile", "GNUmakefile",
                "docker-compose.yml", "docker-compose.yaml",
            ):
                continue

            # 读取文件内容
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if content.strip():
                    documents.append({
                        "file_path": rel_path,
                        "content": content,
                        "file_type": ext.lstrip(".") if ext else "text",
                    })
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")

    logger.info(f"Read {len(documents)} documents from {path}")
    return documents
