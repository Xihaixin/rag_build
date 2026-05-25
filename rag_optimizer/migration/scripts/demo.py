"""
rag_optimizer 端到端演示脚本

演示完整的 RAG 优化流程：
1. 数据库初始化（建表）
2. 数据迁移（从 .pkl 导入 PostgreSQL）
3. 混合检索演示
4. RAG 问答演示
5. 性能基准测试

用法：
    python -m rag_optimizer.scripts.demo --pkl-path ./gitingest.pkl
    python -m rag_optimizer.scripts.demo --pkl-path ./gitingest.pkl --skip-migration
    python -m rag_optimizer.scripts.demo --pkl-path ./gitingest.pkl --retrieval-only
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


# ============================================================
# 工具函数
# ============================================================

def print_header(title: str):
    """打印分区标题"""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_result(label: str, value: str = "", status: str = "OK"):
    """打印带状态的结果"""
    icon = "[OK]" if status == "OK" else "[FAIL]"
    print(f"  {icon} {label}: {value}")


# ============================================================
# 演示步骤
# ============================================================

def step_check_environment() -> bool:
    """步骤 1: 环境检查"""
    print_header("步骤 1: 环境检查")

    # Python 版本
    print_result("Python 版本", sys.version.split()[0])

    # 检查 .env 文件
    env_path = Path(".env")
    if env_path.exists():
        print_result(".env 文件", "已找到")
    else:
        print_result(".env 文件", "未找到 (将使用默认配置)", "FAIL")

    # 检查 PostgreSQL 连接
    try:
        from rag_optimizer.config.settings import settings  # noqa: F401
        from rag_optimizer.db.connection import sync_conn

        conn = sync_conn
        result = conn.execute("SELECT version();")
        pg_version = result[0]["version"] if result else "unknown"
        print_result("PostgreSQL 连接", pg_version[:60])
    except Exception as e:
        print_result("PostgreSQL 连接", f"失败: {e}", "FAIL")
        return False

    # 检查 pgvector 扩展
    try:
        from rag_optimizer.db.connection import sync_conn
        result = sync_conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
        )
        if result:
            print_result("pgvector 扩展", f"版本 {result[0]['extversion']}")
        else:
            print_result("pgvector 扩展", "未安装", "FAIL")
    except Exception:
        print_result("pgvector 扩展", "检查失败", "FAIL")

    # 检查 Redis 连接
    try:
        from rag_optimizer.cache.redis_client import redis_client
        if redis_client.ping():
            print_result("Redis 连接", "正常")
        else:
            print_result("Redis 连接", "无法 ping", "FAIL")
    except Exception as e:
        print_result("Redis 连接", f"失败: {e}", "FAIL")

    return True


def step_init_schema() -> bool:
    """步骤 2: 初始化数据库 schema"""
    print_header("步骤 2: 初始化数据库 Schema")

    try:
        from rag_optimizer.db.connection import sync_conn

        sql_path = Path(__file__).resolve().parent / "001_create_schema.sql"
        if not sql_path.exists():
            print_result("Schema 文件", f"未找到: {sql_path}", "FAIL")
            return False

        sql = sql_path.read_text(encoding="utf-8")

        # 按语句分割执行
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        executed = 0
        for stmt in statements:
            try:
                sync_conn.execute(stmt + ";")
                executed += 1
            except Exception as e:
                logger.warning(f"语句执行跳过: {e}")

        print_result("Schema 初始化", f"执行了 {executed} 条语句")
        return True

    except Exception as e:
        print_result("Schema 初始化", f"失败: {e}", "FAIL")
        return False


def step_migrate_data(pkl_path: str) -> Optional[str]:
    """步骤 3: 数据迁移"""
    print_header("步骤 3: 数据迁移 (pkl -> PostgreSQL)")

    if not os.path.exists(pkl_path):
        print_result("PKL 文件", f"未找到: {pkl_path}", "FAIL")
        return None

    try:
        from rag_optimizer.migration.pkl_to_pg import migrate_pkl_to_postgresql, verify_migration

        # 执行迁移
        project_id = migrate_pkl_to_postgresql(pkl_path)
        print_result("数据迁移", f"项目 ID: {project_id}")

        # 验证
        stats = verify_migration(project_id)
        if stats:
            print_result("文档数", str(stats.get("documents", 0)))
            print_result("分块数", str(stats.get("chunks", 0)))
            print_result("嵌入向量数", str(stats.get("embeddings", 0)))
            print_result("代码符号数", str(stats.get("code_symbols", 0)))

        return project_id

    except Exception as e:
        print_result("数据迁移", f"失败: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return None


def step_hybrid_retrieval(project_id: str) -> bool:
    """步骤 4: 混合检索演示"""
    print_header("步骤 4: 混合检索演示")

    try:
        from rag_optimizer.integration.deepwiki_adapter import PgvectorRetriever

        retriever = PgvectorRetriever(project_id=project_id)

        # 测试查询
        test_queries = [
            "如何配置数据库连接",
            "API 接口文档",
            "项目初始化流程",
        ]

        for query in test_queries:
            print(f"\n  --- 查询: '{query}' ---")

            # 向量检索
            t0 = time.time()
            vec_results = retriever.vector_search(query, top_k=3)
            t1 = time.time()
            print_result(
                "向量检索",
                f"{len(vec_results)} 结果, {(t1-t0)*1000:.1f}ms"
            )

            # 关键词检索
            kw_results = retriever.keyword_search(query, top_k=3)
            t2 = time.time()
            print_result(
                "关键词检索",
                f"{len(kw_results)} 结果, {(t2-t1)*1000:.1f}ms"
            )

            # 混合检索 (RRF)
            hybrid_results = retriever.rrf_search(query, top_k=3)
            t3 = time.time()
            print_result(
                "混合检索 (RRF)",
                f"{len(hybrid_results)} 结果, {(t3-t2)*1000:.1f}ms"
            )

            # 显示 top-1 结果
            if hybrid_results:
                top = hybrid_results[0]
                print(f"    Top-1: [{top.file_path}] (得分: {top.final_score:.4f})")
                print(f"    {top.content[:120]}...")

        return True

    except Exception as e:
        print_result("混合检索", f"失败: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return False


def step_rag_qa(project_id: str) -> bool:
    """步骤 5: RAG 问答演示"""
    print_header("步骤 5: RAG 问答演示")

    try:
        from core.rag_engine import RAGEngine

        engine = RAGEngine(project_id=project_id)

        test_questions = [
            "这个项目的主要功能是什么？",
            "如何配置 API 密钥？",
        ]

        for question in test_questions:
            print(f"\n  --- 问题: '{question}' ---")

            t0 = time.time()
            ctx = engine.answer(
                query=question,
                retrieval_type="hybrid",
                top_k=5,
                use_semantic_cache=False,
            )
            elapsed = time.time() - t0

            print_result("检索结果数", str(len(ctx.results)))
            print_result("回答长度", f"{len(ctx.answer)} 字符")
            print_result("总耗时", f"{ctx.latency_ms}ms ({elapsed:.2f}s)")

            # 显示回答摘要
            if ctx.answer:
                preview = ctx.answer[:300].replace("\n", " ")
                print(f"  回答预览: {preview}...")

        return True

    except Exception as e:
        print_result("RAG 问答", f"失败: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return False


def step_performance_benchmark(project_id: str) -> bool:
    """步骤 6: 性能基准测试"""
    print_header("步骤 6: 性能基准测试")

    try:
        from rag_optimizer.integration.deepwiki_adapter import PgvectorRetriever
        import random

        retriever = PgvectorRetriever(project_id=project_id)

        # 生成随机查询
        def random_query() -> str:
            words = ["database", "config", "api", "function", "class",
                     "import", "setup", "deploy", "test", "error"]
            return " ".join(random.choices(words, k=3))

        queries = [random_query() for _ in range(10)]

        print(f"  运行 10 次查询基准测试...")

        # 向量检索基准
        vec_times = []
        for q in queries:
            t0 = time.time()
            retriever.vector_search(q, top_k=5)
            vec_times.append((time.time() - t0) * 1000)

        avg_vec = sum(vec_times) / len(vec_times)
        print_result("向量检索 (平均)", f"{avg_vec:.1f}ms")

        # 关键词检索基准
        kw_times = []
        for q in queries:
            t0 = time.time()
            retriever.keyword_search(q, top_k=5)
            kw_times.append((time.time() - t0) * 1000)

        avg_kw = sum(kw_times) / len(kw_times)
        print_result("关键词检索 (平均)", f"{avg_kw:.1f}ms")

        # 混合检索基准
        hybrid_times = []
        for q in queries:
            t0 = time.time()
            retriever.hybrid_search(q, top_k=5)
            hybrid_times.append((time.time() - t0) * 1000)

        avg_hybrid = sum(hybrid_times) / len(hybrid_times)
        print_result("混合检索 (平均)", f"{avg_hybrid:.1f}ms")

        print()
        print_result("性能总结",
            f"向量 {avg_vec:.0f}ms | 关键词 {avg_kw:.0f}ms | 混合 {avg_hybrid:.0f}ms")

        return True

    except Exception as e:
        print_result("性能基准", f"失败: {e}", "FAIL")
        return False


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="rag_optimizer 端到端演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整演示
  python -m rag_optimizer.scripts.demo --pkl-path ./gitingest.pkl

  # 仅检索演示（跳过迁移）
  python -m rag_optimizer.scripts.demo --pkl-path ./gitingest.pkl --skip-migration

  # 仅检索测试
  python -m rag_optimizer.scripts.demo --pkl-path ./gitingest.pkl --retrieval-only
        """,
    )
    parser.add_argument(
        "--pkl-path",
        default="./gitingest.pkl",
        help="PKL 数据文件路径 (默认: ./gitingest.pkl)",
    )
    parser.add_argument(
        "--skip-migration",
        action="store_true",
        help="跳过数据迁移步骤",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="仅执行检索演示",
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="指定项目 ID (跳过迁移时使用)",
    )

    args = parser.parse_args()

    print()
    print("  [1m[37m╔══════════════════════════════════════════════════╗[0m")
    print("  [1m[37m║       rag_optimizer - RAG 增强检索演示            ║[0m")
    print("  [1m[37m║   基于 PostgreSQL + pgvector 的透明化 RAG         ║[0m")
    print("  [1m[37m╚══════════════════════════════════════════════════╝[0m")
    print()

    # 步骤 1: 环境检查
    if not step_check_environment():
        logger.error("环境检查失败，请修复后重试")
        sys.exit(1)

    # 步骤 2: Schema 初始化
    if not step_init_schema():
        logger.warning("Schema 初始化异常，继续执行...")

    # 步骤 3: 数据迁移
    project_id: Optional[str] = args.project_id
    if args.skip_migration:
        if not project_id:
            # 尝试从数据库获取最新项目
            try:
                from rag_optimizer.db.connection import sync_conn
                result = sync_conn.execute(
                    "SELECT id FROM projects ORDER BY created_at DESC LIMIT 1"
                )
                if result:
                    project_id = str(result[0]["id"])
                    print_result("使用已有项目", project_id)
                else:
                    logger.error("数据库中没有项目，请先执行迁移")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"获取项目失败: {e}")
                sys.exit(1)
    else:
        pkl_path = args.pkl_path
        if not os.path.exists(pkl_path):
            logger.error(f"PKL 文件不存在: {pkl_path}")
            sys.exit(1)
        project_id = step_migrate_data(pkl_path)
        if not project_id:
            logger.error("数据迁移失败")
            sys.exit(1)

    if args.retrieval_only:
        # 仅执行检索演示
        step_hybrid_retrieval(project_id)
        step_performance_benchmark(project_id)
    else:
        # 完整演示
        step_hybrid_retrieval(project_id)
        step_rag_qa(project_id)
        step_performance_benchmark(project_id)

    # 总结
    print_header("演示完成")
    print(f"  项目 ID: {project_id}")
    print(f"  数据存储: PostgreSQL + pgvector")
    print(f"  检索方式: 向量 + 关键词 + 混合 (RRF)")
    print(f"  缓存层: Redis (Embedding + 语义)")
    print()
    print("  后续步骤:")
    print("  1. 配置 DASHSCOPE_API_KEY 环境变量以启用 LLM 生成")
    print("  2. 运行 python -m rag_optimizer.migration.pkl_to_pg --help")
    print("  3. 集成到 deepwiki-open (见阶段5)")
    print()


if __name__ == "__main__":
    main()
