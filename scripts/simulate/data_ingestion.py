"""
data_ingestion.py — 向后兼容入口（委托给 core.ingestion.ingestor）
=====================================================================

本文件是重构后的向后兼容入口，所有核心逻辑已迁移到:
  core/ingestion/ingestor.py

保留此文件以确保现有导入和命令行调用不受影响。
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# ── 项目路径 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── 日志配置 ──────────────────────────────────────────────────────────────
from config.logging_config import setup_logging
setup_logging()
logger = logging.getLogger("data_ingestion")

from dotenv import load_dotenv
load_dotenv()

# ── 从 core.ingestion.ingestor 导入所有核心功能 ──────────────────────────
from core.ingestion.ingestor import (
    DataIngestor,
    run_ingestion,
    check_project_exists,
    DATA_SOURCE_ROOT,
    REPOS_DIR,
    ensure_data_source_dirs,
)


# ============================================================
# 命令行入口（保留在原文件）
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="数据摄取管道 — 将代码仓库存储到 PostgreSQL + pgvector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python -m scripts.simulate.data_ingestion --repo-url https://github.com/user/repo

  # 指定仓库类型和访问令牌
  python -m scripts.simulate.data_ingestion --repo-url https://github.com/user/repo --repo-type github --token ghp_xxx

  # 使用本地已克隆的仓库
  python -m scripts.simulate.data_ingestion --repo-url https://github.com/user/repo --local-path D:\\DATA_SOURCE\\repos\\my-repo

  # 仅检查项目是否已存在
  python -m scripts.simulate.data_ingestion --repo-url https://github.com/user/repo --check-only
        """,
    )

    parser.add_argument(
        "--repo-url", "-u",
        type=str,
        required=True,
        help="仓库 URL",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="github",
        choices=["github", "gitlab", "bitbucket", "gitee"],
        help="仓库类型",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="访问令牌",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="本地仓库路径（如果已克隆）",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查项目是否已存在，不执行摄取",
    )
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  数据摄取管道 — 代码仓库 → PostgreSQL + pgvector")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info(f"仓库: {args.repo_url}")
    logger.info(f"类型: {args.repo_type}")
    logger.info(f"数据根目录: {DATA_SOURCE_ROOT}")

    # 检查是否已存在
    existing = check_project_exists(args.repo_url)
    if existing:
        logger.info(f"项目已在数据库中 (id={existing})")
        if args.check_only:
            logger.info("仅检查模式，退出")
            return
        logger.info("将重新摄取数据（覆盖已有文档）")

    if args.check_only:
        if existing:
            logger.info(f"✅ 项目已存在: {existing}")
        else:
            logger.info("❌ 项目不存在")
        return

    # 执行摄取
    project_id = run_ingestion(
        repo_url=args.repo_url,
        repo_type=args.repo_type,
        access_token=args.token,
        local_path=args.local_path,
    )

    if project_id:
        logger.info(f"\n✅ 数据摄取成功! project_id={project_id}")
        logger.info(f"现在可以运行 debug_flow.py 使用此项目数据:")
        logger.info(f"  python scripts/simulate/debug_flow.py --mode wiki --repo-url {args.repo_url}")
    else:
        logger.error("\n❌ 数据摄取失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
