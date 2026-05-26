"""
创建 wiki_caches 表（如果不存在）

从 settings 加载数据库配置（支持 .env 文件），
然后执行 CREATE TABLE IF NOT EXISTS。
"""

import sys
import os

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

from rag_optimizer.config.settings import settings
from rag_optimizer.db.connection import sync_conn

# 创建 wiki_caches 表
create_table_sql = """
CREATE TABLE IF NOT EXISTS wiki_caches (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    language        VARCHAR(10) DEFAULT 'zh',
    structure_json  JSONB DEFAULT '{}',
    repo_owner      VARCHAR(255),
    repo_name       VARCHAR(255),
    repo_type       VARCHAR(50),
    repo_url        TEXT,
    provider        VARCHAR(100),
    model           VARCHAR(100),
    source_chunks   JSONB DEFAULT '[]',
    version         INT DEFAULT 1,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (project_id, language)
);
"""

create_index_sql = """
CREATE INDEX IF NOT EXISTS idx_wiki_caches_project ON wiki_caches(project_id);
"""

print(f"Connecting to PostgreSQL: {settings.postgresql.host}:{settings.postgresql.port}/{settings.postgresql.database}")

try:
    sync_conn.execute(create_table_sql)
    print("✓ wiki_caches table created (or already exists)")

    sync_conn.execute(create_index_sql)
    print("✓ Index idx_wiki_caches_project created (or already exists)")

    # 验证
    rows = sync_conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name='wiki_caches'"
    )
    if rows:
        print(f"✓ Verification: wiki_caches table exists")
    else:
        print("✗ Verification: wiki_caches table NOT found")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
