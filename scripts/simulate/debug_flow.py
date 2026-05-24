"""
debug_flow.py — 向后兼容入口（委托给 core 模块）
=====================================================

本文件是重构后的向后兼容入口，所有核心逻辑已迁移到:
  core/flows/base.py          — BaseFlow 基类 + 工具函数
  core/flows/wiki_flow.py     — WikiGenerationFlow
  core/flows/chat_flow.py     — SimpleChatFlow
  core/flows/research_flow.py — DeepResearchFlow
  core/models/__init__.py     — 数据模型
  core/ingestion/ingestor.py  — DataIngestor
  core/cli.py                 — 统一命令行入口

保留此文件以确保现有导入和命令行调用不受影响。
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Any, Optional

# ── 项目路径 ──────────────────────────────────────────────────────────────
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── 日志配置 ──────────────────────────────────────────────────────────────
from config.logging_config import setup_logging
setup_logging()
logger = logging.getLogger("debug_flow")

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════
# 从 core 模块重新导出所有公共 API
# ══════════════════════════════════════════════════════════════════════════

# ── 工具函数 ──────────────────────────────────────────────────────────────
from core.flows.base import (
    parse_repo_url,
    load_configs,
    get_cache_key,
    generate_file_url,
    parse_sse_chunk as _parse_sse_chunk,
    call_llm_and_collect as _call_llm_and_collect,
    BaseFlow,
)

# ── 数据模型 ──────────────────────────────────────────────────────────────
from core.models import (
    WikiPage,
    WikiSection,
    WikiStructure,
    Message,
    ResearchStage,
)

# ── 业务流 ────────────────────────────────────────────────────────────────
from core.flows.wiki_flow import WikiGenerationFlow
from core.flows.chat_flow import SimpleChatFlow
from core.flows.research_flow import DeepResearchFlow

# ── 数据摄取 ──────────────────────────────────────────────────────────────
from core.ingestion.ingestor import (
    DataIngestor,
    run_ingestion,
    check_project_exists,
)


# ══════════════════════════════════════════════════════════════════════════
# 模式运行函数（委托给 core.cli）
# ══════════════════════════════════════════════════════════════════════════

async def run_ingest_mode(args: Any) -> None:
    """运行数据摄取模式（委托给 core.cli）"""
    from core.cli import run_ingest_mode as _run
    await _run(args)


async def run_wiki_mode(args: Any) -> None:
    """运行 Wiki 生成模式（委托给 core.cli）"""
    from core.cli import run_wiki_mode as _run
    await _run(args)


async def run_chat_mode(args: Any) -> None:
    """运行 Q&A 聊天模式（委托给 core.cli）"""
    from core.cli import run_chat_mode as _run
    await _run(args)


async def run_research_mode(args: Any) -> None:
    """运行深度研究模式（委托给 core.cli）"""
    from core.cli import run_research_mode as _run
    await _run(args)


# ══════════════════════════════════════════════════════════════════════════
# 参数解析 & 主入口（保留在原文件）
# ══════════════════════════════════════════════════════════════════════════

def parse_args() -> Any:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DeepWiki-open 业务逻辑流独立调试程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 数据摄取（前置步骤：将仓库数据存入数据库）
  python scripts/simulate/debug_flow.py --mode ingest --repo-url https://github.com/user/repo

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
        choices=["ingest", "wiki", "chat", "research"],
        default="wiki",
        help="运行模式: ingest (数据摄取), wiki (Wiki生成), chat (Q&A聊天), research (深度研究)",
    )

    parser.add_argument(
        "--repo-url", "-u",
        type=str,
        default="https://github.com/Xihaixin/MathModelAgent",
        help="仓库 URL",
    )

    parser.add_argument(
        "--repo-type",
        type=str,
        default=None,
        choices=["github", "gitlab", "bitbucket", "gitee"],
        help="仓库类型（仅 ingest 模式）",
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

    # ── 数据摄取相关参数 ──────────────────────────────────────────────
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Git 访问令牌（仅 ingest 模式）",
    )

    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="本地仓库路径（仅 ingest 模式，如果已克隆）",
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

    if args.mode == "ingest":
        await run_ingest_mode(args)
    elif args.mode == "wiki":
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
