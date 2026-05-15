"""初始化 rag_optimizer 数据库 Schema"""
import psycopg2
from pathlib import Path

# 连接到 rag_optimizer 数据库
conn = psycopg2.connect(host='localhost', port=5432, dbname='rag_optimizer', user='postgres', password='200310')
conn.autocommit = True
cur = conn.cursor()

# 读取 SQL 文件
sql_path = Path(__file__).resolve().parent / "rag_optimizer" / "scripts" / "001_create_schema.sql"
sql = sql_path.read_text(encoding="utf-8")

# 按分号分割语句（跳过注释行和空行）
statements = []
for line in sql.split("\n"):
    line = line.strip()
    if line.startswith("--") or line.startswith("CREATE DATABASE"):
        continue
    statements.append(line)

full_sql = "\n".join(statements)

# 按分号分割执行
parts = full_sql.split(";")
executed = 0
skipped = 0
errors = []

for part in parts:
    stmt = part.strip()
    if not stmt:
        continue
    try:
        cur.execute(stmt + ";")
        executed += 1
        print(f"  [OK] 执行成功: {stmt[:60]}...")
    except Exception as e:
        err_msg = str(e)
        if "already exists" in err_msg.lower():
            skipped += 1
            print(f"  [SKIP] 已存在: {stmt[:60]}...")
        else:
            errors.append(f"{stmt[:80]} -> {err_msg}")
            print(f"  [ERR] {err_msg[:80]}")

cur.close()
conn.close()

print(f"\n完成: {executed} 条执行, {skipped} 条跳过")
if errors:
    print(f"错误: {len(errors)} 个")
    for e in errors[:5]:
        print(f"  {e}")
