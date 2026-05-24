"""
core.ingestion.ingestor — 数据摄取器：代码仓库 → PostgreSQL + pgvector
=====================================================================

本模块实现完整的预处理数据流，将代码仓库的数据存储到 PostgreSQL + pgvector 数据库中，
替代原始 deepwiki-open 项目的 .pkl 本地文件存储。

数据流：
  1. download()    — git clone 仓库到 DATA_SOURCE/repos/
  2. prepare()     — 创建/获取项目记录和摄取任务
  3. read_documents() — 读取仓库中的所有文档文件
  4. ingest()      — 分块 + 嵌入 + 存储 (IngestionPipeline.process_documents)
  5. finalize()    — 更新任务状态，返回统计信息

设计原则：
  - 复用 core/utils/repo.py 中的 download_repo()
  - 复用 core/utils/documents.py 中的 read_all_documents()
  - 复用 rag_optimizer/pipeline/ingestion.py 中的 IngestionPipeline
  - 复用 rag_optimizer/db/repository.py 中的各 Repository 类
  - 与 core/flows/ 中的各 Flow 类完全兼容，作为其前置步骤
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from core.config import DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
from core.utils.repo import download_repo
from core.utils.documents import read_all_documents
from rag_optimizer.db.repository import (
    ProjectRepository,
    DocumentRepository,
    ChunkRepository,
    EmbeddingRepository,
    IngestionJobRepository,
    PipelineLogRepository,
)
from rag_optimizer.pipeline.ingestion import IngestionPipeline

logger = logging.getLogger("core.ingestion.ingestor")


# ============================================================
# 数据根目录
# ============================================================

# 所有克隆的仓库统一存储在此目录下
DATA_SOURCE_ROOT = r"D:\ProgramFile2_OR\Python_Study_System\OpenStudy\DATA_SOURCE"
REPOS_DIR = os.path.join(DATA_SOURCE_ROOT, "repos")


def ensure_data_source_dirs():
    """确保 DATA_SOURCE 目录结构存在"""
    os.makedirs(REPOS_DIR, exist_ok=True)
    logger.info(f"数据根目录: {DATA_SOURCE_ROOT}")
    logger.info(f"仓库存储目录: {REPOS_DIR}")


# ============================================================
# 数据摄取器
# ============================================================

class DataIngestor:
    """
    数据摄取器 — 将代码仓库完整处理并存储到 PostgreSQL + pgvector。

    完整流程：
      1. download()    — git clone 仓库到 DATA_SOURCE/repos/
      2. prepare()     — 创建/获取项目记录和摄取任务
      3. read_documents() — 读取仓库中的所有文档文件
      4. ingest()      — 分块 → 嵌入 → 存储
      5. finalize()    — 更新任务状态，返回统计信息
    """

    def __init__(
        self,
        repo_url: str,
        repo_type: str = "github",
        access_token: Optional[str] = None,
        excluded_dirs: Optional[List[str]] = None,
        excluded_files: Optional[List[str]] = None,
        included_dirs: Optional[List[str]] = None,
        included_files: Optional[List[str]] = None,
        local_path: Optional[str] = None,
    ):
        """
        Args:
            repo_url: 仓库 URL
            repo_type: 仓库类型 (github, gitlab, bitbucket, gitee)
            access_token: 访问令牌
            excluded_dirs: 排除的目录列表
            excluded_files: 排除的文件列表
            included_dirs: 包含的目录列表
            included_files: 包含的文件列表
            local_path: 本地路径（如果已克隆，可指定）
        """
        self.repo_url = repo_url
        self.repo_type = repo_type
        self.access_token = access_token
        self.excluded_dirs = excluded_dirs or DEFAULT_EXCLUDED_DIRS
        self.excluded_files = excluded_files or DEFAULT_EXCLUDED_FILES
        self.included_dirs = included_dirs
        self.included_files = included_files
        self.local_path = local_path

        # 从 URL 提取仓库名
        self.repo_name = self._extract_repo_name(repo_url)

        # 确保 DATA_SOURCE 目录存在
        ensure_data_source_dirs()

        # 运行时状态
        self.project_id: Optional[str] = None
        self.job_id: Optional[str] = None
        self.repo_local_path: Optional[str] = None
        self.stats: Dict[str, Any] = {}

    def _extract_repo_name(self, repo_url: str) -> str:
        """从 URL 提取仓库名"""
        return repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    # ── 步骤 1: 下载仓库 ──────────────────────────────────────────────

    def download(self) -> str:
        """
        下载仓库到 DATA_SOURCE/repos/ 目录。

        Returns:
            str: 本地路径
        """
        if self.local_path and os.path.exists(self.local_path):
            logger.info(f"使用已有本地路径: {self.local_path}")
            self.repo_local_path = self.local_path
            return self.repo_local_path

        # 统一存储到 DATA_SOURCE/repos/{repo_name}
        default_path = os.path.join(REPOS_DIR, self.repo_name)

        if os.path.exists(default_path) and os.listdir(default_path):
            logger.info(f"仓库已存在: {default_path}")
            self.repo_local_path = default_path
            return self.repo_local_path

        logger.info(f"下载仓库到: {default_path}")
        logger.info(f"仓库 URL: {self.repo_url}")
        self.repo_local_path = download_repo(
            repo_url=self.repo_url,
            local_path=default_path,
            repo_type=self.repo_type,
            access_token=self.access_token,
        )
        logger.info(f"仓库下载完成: {self.repo_local_path}")
        return self.repo_local_path

    # ── 步骤 2: 准备数据库记录 ────────────────────────────────────────

    def prepare(self) -> str:
        """
        创建/获取项目记录和摄取任务。

        Returns:
            str: project_id
        """
        # 1. 创建或获取项目
        project = ProjectRepository.get_or_create(
            name=self.repo_name,
            repo_url=self.repo_url,
            owner=None,
            repo_type=self.repo_type,
            local_path=self.repo_local_path,
        )
        self.project_id = str(project["id"])
        logger.info(f"项目记录: {self.repo_name} (id={self.project_id})")

        # 2. 创建摄取任务
        job = IngestionJobRepository.create(
            project_id=self.project_id,
            trigger_type="manual",
        )
        self.job_id = str(job["id"])
        logger.info(f"摄取任务: {self.job_id}")

        # 3. 更新任务状态为 cloning
        IngestionJobRepository.update_status(
            self.job_id, "cloning", stage="download",
        )

        return self.project_id

    # ── 步骤 3: 读取文档 ──────────────────────────────────────────────

    def read_documents(self) -> List[Dict[str, Any]]:
        """
        读取仓库中的所有文档文件。

        Returns:
            List[Dict]: 文档列表
        """
        if not self.repo_local_path:
            raise RuntimeError("请先调用 download() 下载仓库")

        logger.info(f"读取文档: {self.repo_local_path}")

        # 更新任务状态
        if self.job_id:
            IngestionJobRepository.update_status(
                self.job_id, "parsing", stage="reading",
            )

        documents = read_all_documents(
            path=self.repo_local_path,
            excluded_dirs=self.excluded_dirs,
            excluded_files=self.excluded_files,
            included_dirs=self.included_dirs,
            included_files=self.included_files,
        )

        logger.info(f"读取到 {len(documents)} 个文档")
        return documents

    # ── 步骤 4: 执行摄取（分块 + 嵌入 + 存储） ────────────────────────

    def ingest(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        执行数据摄取：分块 → 嵌入 → 存储到 PostgreSQL。

        Args:
            documents: 文档列表

        Returns:
            Dict: 处理统计信息
        """
        if not self.project_id:
            raise RuntimeError("请先调用 prepare() 创建项目记录")

        if not documents:
            logger.warning("没有文档需要处理")
            return {"processed": 0, "skipped": 0, "errors": 0,
                    "total_chunks": 0, "total_embeddings": 0}

        logger.info(f"开始数据摄取: {len(documents)} 个文档")

        # 更新任务状态
        if self.job_id:
            IngestionJobRepository.update_status(
                self.job_id, "in_progress", stage="processing",
                progress=0, processed=0, total=len(documents),
            )

        # 记录管道开始
        PipelineLogRepository.log(
            project_id=self.project_id, job_id=self.job_id,
            step_name="ingestion", status="started",
            input_count=len(documents),
            parameters={"total_documents": len(documents)},
        )

        start_time = time.time()

        # 使用 IngestionPipeline 处理所有文档
        pipeline = IngestionPipeline(
            project_id=self.project_id,
            job_id=self.job_id,
        )
        stats = pipeline.process_documents(documents)

        # 记录管道完成
        elapsed = int((time.time() - start_time) * 1000)
        PipelineLogRepository.log(
            project_id=self.project_id, job_id=self.job_id,
            step_name="ingestion", status="completed",
            input_count=len(documents),
            output_count=stats["total_chunks"],
            duration_ms=elapsed,
            parameters=stats,
        )

        logger.info(
            f"数据摄取完成: "
            f"{stats['processed']} 处理, {stats['skipped']} 跳过, "
            f"{stats['errors']} 错误, "
            f"{stats['total_chunks']} 分块, {stats['total_embeddings']} 嵌入向量 "
            f"耗时 {elapsed / 1000:.1f}s"
        )

        self.stats = stats
        return stats

    # ── 步骤 5: 完成 ──────────────────────────────────────────────────

    def finalize(self, success: bool = True, error: Optional[str] = None):
        """
        完成摄取任务。

        Args:
            success: 是否成功
            error: 错误信息
        """
        if self.job_id:
            status = "completed" if success else "failed"
            IngestionJobRepository.update_status(
                self.job_id, status,
                stage="completed" if success else "failed",
                progress=1.0 if success else None,
                error=error,
            )

        if success:
            logger.info("=" * 60)
            logger.info(f"✅ 数据摄取成功完成!")
            logger.info(f"   项目: {self.repo_name} (id={self.project_id})")
            logger.info(f"   文档: {self.stats.get('processed', 0)} 处理")
            logger.info(f"   分块: {self.stats.get('total_chunks', 0)}")
            logger.info(f"   嵌入: {self.stats.get('total_embeddings', 0)}")
            logger.info(f"   本地路径: {self.repo_local_path}")
            logger.info("=" * 60)
        else:
            logger.error(f"❌ 数据摄取失败: {error}")

    # ── 全流程执行 ────────────────────────────────────────────────────

    def run(self) -> Optional[str]:
        """
        执行完整的数据摄取流程。

        Returns:
            Optional[str]: 成功返回 project_id，失败返回 None
        """
        try:
            # 步骤 1: 下载仓库
            self.download()

            # 步骤 2: 准备数据库记录
            self.prepare()

            # 步骤 3: 读取文档
            documents = self.read_documents()

            if not documents:
                logger.warning("没有找到可处理的文档，跳过摄取")
                self.finalize(True)
                return self.project_id

            # 步骤 4: 执行摄取
            self.ingest(documents)

            # 步骤 5: 完成
            self.finalize(True)
            return self.project_id

        except Exception as e:
            logger.error(f"数据摄取过程出错: {e}", exc_info=True)
            self.finalize(False, str(e))
            return None


# ============================================================
# 便捷函数
# ============================================================

def run_ingestion(
    repo_url: str,
    repo_type: str = "github",
    access_token: Optional[str] = None,
    excluded_dirs: Optional[List[str]] = None,
    excluded_files: Optional[List[str]] = None,
    included_dirs: Optional[List[str]] = None,
    included_files: Optional[List[str]] = None,
    local_path: Optional[str] = None,
) -> Optional[str]:
    """
    便捷函数 — 执行完整的数据摄取流程。

    Args:
        repo_url: 仓库 URL
        repo_type: 仓库类型
        access_token: 访问令牌
        excluded_dirs: 排除的目录
        excluded_files: 排除的文件
        included_dirs: 包含的目录
        included_files: 包含的文件
        local_path: 本地路径

    Returns:
        Optional[str]: project_id
    """
    ingestor = DataIngestor(
        repo_url=repo_url,
        repo_type=repo_type,
        access_token=access_token,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
        included_dirs=included_dirs,
        included_files=included_files,
        local_path=local_path,
    )
    return ingestor.run()


def check_project_exists(repo_url: str) -> Optional[str]:
    """
    检查仓库是否已在数据库中有数据。

    Args:
        repo_url: 仓库 URL

    Returns:
        Optional[str]: 如果存在返回 project_id，否则返回 None
    """
    try:
        projects = ProjectRepository.list_all()
        for proj in projects:
            proj_url = proj.get("repo_url", "") or ""
            if repo_url in proj_url or proj_url in repo_url:
                pid = str(proj["id"])
                # 检查是否有文档数据
                docs = DocumentRepository.get_by_project(pid)
                if docs:
                    logger.info(f"项目已存在且有数据: {proj.get('name')} (id={pid}, docs={len(docs)})")
                    return pid
                else:
                    logger.info(f"项目已存在但无文档数据: {proj.get('name')} (id={pid})")
                    return pid
        return None
    except Exception as e:
        logger.warning(f"检查项目是否存在时出错: {e}")
        return None
