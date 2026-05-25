"""
core.cli — 统一命令行入口
============================

整合所有业务流和数据管道的 CLI 入口点。

支持模式：
  - ingest   数据摄取（代码仓库 → PostgreSQL + pgvector）
  - wiki     Wiki 文档生成
  - chat     Q&A 简单聊天
  - research 深度研究

用法：
  python -m core.cli --mode wiki --repo-url https://github.com/user/repo
  python -m core.cli --mode ingest --repo-url https://github.com/user/repo
  python -m core.cli --mode chat --repo-url https://github.com/user/repo --query "问题"
  python -m core.cli --mode research --repo-url https://github.com/user/repo --query "研究主题"
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Any, Optional

from core.config.logging_config import setup_logging
setup_logging()
logger = logging.getLogger("core.cli")

from dotenv import load_dotenv
load_dotenv()

import os

from core.ingestion.ingestor import DataIngestor, check_project_exists
from core.flows.wiki_flow import WikiGenerationFlow
from core.flows.chat_flow import SimpleChatFlow
from core.flows.research_flow import DeepResearchFlow


def _resolve_repo_url(repo_url: Optional[str], local_path: Optional[str]) -> str:
    """
    解析仓库 URL：若未显式提供 repo_url，则从 local_path 推导。

    推导规则：
      1. 如果 repo_url 已提供，直接返回
      2. 如果 local_path 已提供，使用 local_path 的目录名作为 repo 名，
         构造一个本地友好的 URL（file:/// 格式）
      3. 兜底返回空字符串（后续流程会报错提示用户）
    """
    if repo_url:
        return repo_url
    if local_path:
        dirname = os.path.basename(os.path.normpath(local_path.rstrip("/\\")))
        # 使用 file:// 协议表示本地项目
        abs_path = os.path.abspath(local_path)
        return f"file:///{abs_path.replace(os.sep, '/')}"
    return ""


# ============================================================
# 模式运行函数
# ============================================================

async def run_ingest_mode(args: Any) -> None:
    """运行数据摄取模式"""
    logger.info("=" * 70)
    logger.info("模式: 数据摄取 (代码仓库 → PostgreSQL + pgvector)")
    logger.info("=" * 70)

    repo_url = _resolve_repo_url(args.repo_url, args.local_path)

    # 检查项目是否已存在
    existing = check_project_exists(repo_url)
    if existing:
        logger.info(f"项目已在数据库中 (id={existing})，将重新摄取")

    # 执行数据摄取
    ingestor = DataIngestor(
        repo_url=repo_url,
        repo_type=args.repo_type or "github",
        access_token=args.token,
        local_path=args.local_path,
    )
    project_id = ingestor.run()

    if project_id:
        logger.info(f"\n✅ 数据摄取成功! project_id={project_id}")
        logger.info(f"现在可以运行其他模式使用此项目数据:")
        logger.info(f"  python -m core.cli --mode wiki --repo-url {repo_url}")
        logger.info(f"  python -m core.cli --mode chat --repo-url {repo_url} --query '...'")
        logger.info(f"  python -m core.cli --mode research --repo-url {repo_url} --query '...'")
    else:
        logger.error("\n❌ 数据摄取失败")


async def run_wiki_mode(args: Any) -> None:
    """运行 Wiki 生成模式"""
    logger.info("=" * 70)
    logger.info("模式: Wiki 文档生成")
    logger.info("=" * 70)

    repo_url = _resolve_repo_url(args.repo_url, args.local_path)

    # ── 初始化 Wiki 生成流 ────────────────────────────────────────────────
    # WikiGenerationFlow.fetch_repository_structure() 内部会自动处理：
    #   1. use_database=True  → 尝试从数据库获取数据
    #   2. 数据库无数据       → 自动触发 DataIngestor 摄取管道
    #   3. 摄取完成           → 重新从数据库获取
    flow = WikiGenerationFlow(
        repo_url=repo_url,
        provider=args.provider,
        model=args.model,
        language=args.language,
        comprehensive=not args.concise,
        use_database=not args.no_db,
        local_path=args.local_path,
    )

    # 执行完整流程（run() 内部包含所有步骤）
    result = await flow.run()

    saved_count = result.get("saved_count", 0)
    if saved_count > 0:
        logger.info(f"✅ Wiki 页面已保存到数据库: {saved_count} 页")


async def run_chat_mode(args: Any) -> None:
    """运行 Q&A 聊天模式"""
    logger.info("=" * 70)
    logger.info("模式: 用户 Q&A 简单聊天")
    logger.info("=" * 70)

    repo_url = _resolve_repo_url(args.repo_url, args.local_path)

    flow = SimpleChatFlow(
        repo_url=repo_url,
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

    repo_url = _resolve_repo_url(args.repo_url, args.local_path)

    flow = DeepResearchFlow(
        repo_url=repo_url,
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


# ============================================================
# 参数解析
# ============================================================

def parse_args(argv: Optional[list] = None) -> Any:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DeepWiki-open 核心业务逻辑流 — 统一命令行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 数据摄取（前置步骤：将仓库数据存入数据库）
  python -m core.cli --mode ingest --repo-url https://github.com/user/repo

  # Wiki 生成
  python -m core.cli --mode wiki --repo-url https://github.com/user/repo

  # Q&A 聊天
  python -m core.cli --mode chat --repo-url https://github.com/user/repo --query "如何配置项目？"

  # 深度研究
  python -m core.cli --mode research --repo-url https://github.com/user/repo --query "架构设计原理"

  # 使用样本数据（不连接数据库）
  python -m core.cli --mode wiki --repo-url https://github.com/user/repo --no-db
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
        default=None,
        help="仓库 URL（未指定时，若提供 --local-path 则自动从本地路径推导）",
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
        help="本地仓库路径（ingest/wiki 模式通用）",
    )

    return parser.parse_args(argv)


# ============================================================
# 主入口
# ============================================================

async def main(argv: Optional[list] = None) -> None:
    """主入口"""
    args = parse_args(argv)

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  DeepWiki-open 核心业务逻辑流")
    logger.info("║  Core Business Logic Flows")
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
    import asyncio
    asyncio.run(main())
