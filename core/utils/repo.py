"""
repo.py — 仓库下载与远程文件读取工具
=====================================

提供仓库克隆和远程文件内容获取功能。

依赖:
  - core.config — DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES（间接）
"""

import logging
import os
import subprocess
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("core.utils.repo")


# ══════════════════════════════════════════════════════════════════════════
# 仓库下载
# ══════════════════════════════════════════════════════════════════════════


def download_repo(
    repo_url: str,
    local_path: str,
    repo_type: Optional[str] = None,
    access_token: Optional[str] = None,
) -> str:
    """
    下载仓库到本地。

    参数:
        repo_url: 仓库 URL
        local_path: 本地路径
        repo_type: 仓库类型 (github, gitlab, bitbucket, gitee)
        access_token: 访问令牌

    返回:
        str: 本地路径
    """
    logger.info(f"Downloading repo: {repo_url} to {local_path}")

    # 如果本地路径已存在，跳过下载
    if os.path.exists(local_path) and os.listdir(local_path):
        logger.info(f"Local path already exists: {local_path}")
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        # 构建带认证的 URL
        if access_token:
            parsed = urlparse(repo_url)
            auth_url = f"{parsed.scheme}://{access_token}@{parsed.netloc}{parsed.path}"
        else:
            auth_url = repo_url

        # 执行 git clone
        result = subprocess.run(
            ["git", "clone", "--depth=1", auth_url, local_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 分钟超时
        )

        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr}")
            raise RuntimeError(f"Failed to clone repository: {result.stderr}")

        logger.info(f"Repository cloned successfully to {local_path}")
        return local_path

    except subprocess.TimeoutExpired:
        logger.error("Git clone timed out")
        raise RuntimeError("Repository clone timed out")
    except Exception as e:
        logger.error(f"Error downloading repo: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════
# 远程文件内容获取
# ══════════════════════════════════════════════════════════════════════════


def get_github_file_content(repo_url: str, file_path: str, access_token: Optional[str] = None) -> str:
    """通过 GitHub API 获取文件内容"""
    import json
    import urllib.request

    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip("/").split("/")

    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {repo_url}")

    owner, repo = path_parts[0], path_parts[1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path.lstrip('/')}"

    headers = {
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "DeepWiki-Open",
    }
    if access_token:
        headers["Authorization"] = f"token {access_token}"

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Error fetching GitHub file {file_path}: {e}")
        raise


def get_gitlab_file_content(repo_url: str, file_path: str, access_token: Optional[str] = None) -> str:
    """通过 GitLab API 获取文件内容"""
    import urllib.parse
    import urllib.request

    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip("/").split("/")

    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitLab URL: {repo_url}")

    project_path = urllib.parse.quote("/".join(path_parts), safe="")
    encoded_file_path = urllib.parse.quote(file_path.lstrip("/"), safe="")
    api_url = f"https://gitlab.com/api/v4/projects/{project_path}/repository/files/{encoded_file_path}/raw"

    headers = {"User-Agent": "DeepWiki-Open"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Error fetching GitLab file {file_path}: {e}")
        raise


def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: Optional[str] = None) -> str:
    """通过 Bitbucket API 获取文件内容"""
    import urllib.parse
    import urllib.request

    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip("/").split("/")

    if len(path_parts) < 2:
        raise ValueError(f"Invalid Bitbucket URL: {repo_url}")

    owner, repo = path_parts[0], path_parts[1]
    encoded_path = urllib.parse.quote(file_path.lstrip("/"), safe="")
    api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/src/master/{encoded_path}"

    headers = {"User-Agent": "DeepWiki-Open"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Error fetching Bitbucket file {file_path}: {e}")
        raise


def get_file_content(
    repo_url: str,
    file_path: str,
    repo_type: Optional[str] = None,
    access_token: Optional[str] = None,
) -> str:
    """
    从远程仓库获取文件内容。

    参数:
        repo_url: 仓库 URL
        file_path: 文件路径
        repo_type: 仓库类型 (github, gitlab, bitbucket)
        access_token: 访问令牌

    返回:
        str: 文件内容
    """
    if repo_type == "gitlab" or "gitlab.com" in repo_url:
        return get_gitlab_file_content(repo_url, file_path, access_token)
    elif repo_type == "bitbucket" or "bitbucket.org" in repo_url:
        return get_bitbucket_file_content(repo_url, file_path, access_token)
    else:
        return get_github_file_content(repo_url, file_path, access_token)
