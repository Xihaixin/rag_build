"""
rag_optimizer 环境设置脚本

一键安装依赖、初始化数据库、创建 .env 配置。

用法：
    python -m rag_optimizer.scripts.setup          # 完整设置
    python -m rag_optimizer.scripts.setup --db-only # 仅初始化数据库
    python -m rag_optimizer.scripts.setup --check   # 仅检查环境
    python -m rag_optimizer.scripts.setup --clean   # 清理数据库
"""

import argparse
import os
import subprocess
import sys
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"

def ok(msg: str):
    print(f"  {Colors.GREEN}[OK]{Colors.END} {msg}")

def warn(msg: str):
    print(f"  {Colors.YELLOW}[WARN]{Colors.END} {msg}")

def fail(msg: str):
    print(f"  {Colors.RED}[FAIL]{Colors.END} {msg}")

def info(msg: str):
    print(f"  {Colors.CYAN}[INFO]{Colors.END} {msg}")

def check_python():
    """检查 Python 版本"""
    v = sys.version_info
    if v.major >= 3 and v.minor >= 10:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    fail(f"Python {v.major}.{v.minor}.{v.micro} (需要 >= 3.10)")
    return False

def check_postgresql():
    """检查 PostgreSQL 连接"""
    try:
        import psycopg2
    except ImportError:
        warn("psycopg2 未安装 (将自动安装)")
        return None

    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", "5432"),
            dbname=os.getenv("PGDATABASE", "rag_optimizer"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"),
        )
        cur = conn.cursor()
        cur.execute("SELECT version();")
        row = cur.fetchone()
        if row:
            version = row[0]
            ok(f"PostgreSQL: {version[:50]}...")
        else:
            fail("无法获取 PostgreSQL 版本信息")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        fail(f"PostgreSQL 连接失败: {e}")
        return False

def check_pgvector():
    """检查 pgvector 扩展"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", "5432"),
            dbname=os.getenv("PGDATABASE", "rag_optimizer"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"),
        )
        cur = conn.cursor()
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        row = cur.fetchone()
        if row:
            ok(f"pgvector 扩展: v{row[0]}")
        else:
            fail("pgvector 扩展未安装 (需要 CREATE EXTENSION vector;)")
        cur.close()
        conn.close()
        return bool(row)
    except Exception as e:
        fail(f"pgvector 检查失败: {e}")
        return False

def check_redis():
    """检查 Redis 连接"""
    try:
        import redis
    except ImportError:
        warn("redis 未安装 (将自动安装)")
        return None

    try:
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            socket_connect_timeout=3,
        )
        r.ping()
        ok("Redis 连接正常")
        r.close()
        return True
    except Exception as e:
        warn(f"Redis 连接失败: {e} (缓存功能将降级)")
        return False

def check_dashscope_key():
    """检查 DashScope API Key"""
    key = os.getenv("DASHSCOPE_API_KEY")
    if key:
        ok("DASHSCOPE_API_KEY 已配置")
        return True
    warn("DASHSCOPE_API_KEY 未配置 (LLM 生成将使用 mock)")
    return False

def create_env_file():
    """创建 .env 配置文件"""
    env_path = Path(".env")
    if env_path.exists():
        warn(".env 文件已存在，跳过创建")
        return

    content = """# rag_optimizer 配置
# PostgreSQL
PGHOST=localhost
PGPORT=5432
PGDATABASE=rag_optimizer
PGUSER=postgres
PGPASSWORD=postgres

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# DashScope
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
"""
    env_path.write_text(content, encoding="utf-8")
    ok("已创建 .env 配置文件")

def init_database():
    """初始化数据库 schema"""
    info("初始化数据库 Schema...")

    sql_path = Path(__file__).resolve().parent / "001_create_schema.sql"
    if not sql_path.exists():
        fail(f"Schema 文件未找到: {sql_path}")
        return False

    try:
        import psycopg2
    except ImportError:
        fail("psycopg2 未安装，无法初始化数据库")
        return False

    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", "5432"),
            dbname=os.getenv("PGDATABASE", "rag_optimizer"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"),
        )
        cur = conn.cursor()

        sql = sql_path.read_text(encoding="utf-8")
        statements = [s.strip() for s in sql.split(";") if s.strip()]

        executed = 0
        skipped = 0
        for stmt in statements:
            try:
                cur.execute(stmt + ";")
                conn.commit()
                executed += 1
            except Exception as e:
                conn.rollback()
                skipped += 1
                if "already exists" not in str(e).lower():
                    warn(f"  语句跳过: {str(e)[:80]}")

        cur.close()
        conn.close()

        ok(f"Schema 初始化完成: {executed} 条执行, {skipped} 条跳过")
        return True

    except Exception as e:
        fail(f"数据库初始化失败: {e}")
        return False


def clean_database():
    """
    彻底清理 public schema 下的所有对象（表、类型、扩展数据等）
    通过删除并重建 public schema 实现，比逐个删除更可靠
    """
    info("清理数据库...")

    try:
        import psycopg2
    except ImportError:
        fail("psycopg2 未安装，无法清理数据库")
        return False

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", "5432"),
            dbname=os.getenv("PGDATABASE", "rag_optimizer"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"),
        )
        conn.autocommit = True  # 创建/删除 schema 需要自动提交
        cur = conn.cursor()

        # 1. 检查 public schema 是否存在
        cur.execute("SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'public');")
        if not cur.fetchone()[0]:
            warn("public schema 不存在，跳过清理")
            return True

        # 2. 删除 public schema（ CASCADE 会删除其中所有对象：表、类型、函数、扩展数据等）
        # 注意：这不会删除 extension 本身，只会删除 extension 创建的数据对象
        cur.execute("DROP SCHEMA public CASCADE;")
        
        # 3. 重新创建 public schema
        cur.execute("CREATE SCHEMA public;")
        
        # 4. 恢复默认权限（可选，但推荐，确保 postgres 用户有所有权）
        cur.execute("GRANT ALL ON SCHEMA public TO postgres;")
        cur.execute("GRANT ALL ON SCHEMA public TO public;")

        ok("数据库清理完成（public schema 已重置）")
        return True

    except Exception as e:
        fail(f"数据库清理失败: {e}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(
        description="rag_optimizer 环境设置",
    )
    parser.add_argument("--check", action="store_true", help="仅检查环境")
    parser.add_argument("--db-only", action="store_true", help="仅初始化数据库")
    parser.add_argument("--clean", action="store_true", help="清理数据库")

    args = parser.parse_args()

    print()
    print(f"  {Colors.BOLD}╔══════════════════════════════════════╗{Colors.END}")
    print(f"  {Colors.BOLD}║   rag_optimizer 环境设置工具        ║{Colors.END}")
    print(f"  {Colors.BOLD}║   基于 PostgreSQL + pgvector 的 RAG ║{Colors.END}")
    print(f"  {Colors.BOLD}╚══════════════════════════════════════╝{Colors.END}")
    print()

    if args.check:
        print(f"  {Colors.BOLD}--- 环境检查 ---{Colors.END}")
        check_python()
        check_postgresql()
        check_pgvector()
        check_redis()
        check_dashscope_key()
        print()
        return

    if args.db_only:
        print(f"  {Colors.BOLD}--- 数据库初始化 ---{Colors.END}")
        init_database()
        print()
        return

    if args.clean:
        print(f"  {Colors.BOLD}--- 数据库清理 ---{Colors.END}")
        clean_database()
        print()
        return

    print(f"  {Colors.BOLD}--- 步骤 1: 环境检查 ---{Colors.END}")
    check_python()

    print()
    print(f"  {Colors.BOLD}--- 步骤 2: 创建配置文件 ---{Colors.END}")
    create_env_file()

    print()
    print(f"  {Colors.BOLD}--- 步骤 3: 数据库初始化 ---{Colors.END}")
    init_database()

    print()
    print(f"  {Colors.BOLD}--- 步骤 4: 验证 ---{Colors.END}")
    check_postgresql()
    check_pgvector()
    check_redis()
    check_dashscope_key()

    print()
    info("设置完成！后续步骤:")
    info("  1. 编辑 .env 文件配置 DASHSCOPE_API_KEY")
    info("  2. 运行 python -m rag_optimizer.scripts.demo --pkl-path ./gitingest.pkl")
    info("  3. 查看文档: plans/毕业设计-RAG系统优化方案_v2.md")
    print()

if __name__ == "__main__":
    main()
