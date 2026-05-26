-- ============================================================
-- 毕业设计：透明化 RAG 系统优化
-- 数据库 Schema 创建脚本 v2.0
-- 数据库：PostgreSQL 16 + pgvector 0.7+
-- ============================================================

-- 创建数据库（需要超级用户权限执行）
-- Windows 环境使用 template0 和系统默认排序规则：
--   CREATE DATABASE rag_optimizer WITH ENCODING 'UTF8'
--     LC_COLLATE='Chinese (Simplified)_China.936'
--     LC_CTYPE='Chinese (Simplified)_China.936'
--     TEMPLATE template0;
-- Linux/macOS 环境：
--   CREATE DATABASE rag_optimizer WITH ENCODING 'UTF8'
--     LC_COLLATE='zh_CN.UTF-8' LC_CTYPE='zh_CN.UTF-8';

-- ============================================================
-- 扩展安装
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;  -- 查询性能分析

-- ============================================================
-- 类型定义
-- ============================================================
-- 单独定义类型，避免 DO $$ 块问题
CREATE TYPE ingestion_status AS ENUM (
    'pending', 'cloning', 'parsing', 'chunking',
    'embedding', 'indexing', 'completed', 'failed'
);

CREATE TYPE symbol_type AS ENUM (
    'class', 'function', 'method', 'interface',
    'variable', 'module', 'decorator', 'enum'
);

-- ============================================================
-- 1. 嵌入模型注册表
-- ============================================================
CREATE TABLE IF NOT EXISTS embedding_models (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100) UNIQUE NOT NULL,
    provider        VARCHAR(50) NOT NULL,
    dimensions      INT NOT NULL,
    description     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE embedding_models IS '嵌入模型注册表，统一管理所有支持的嵌入模型';
COMMENT ON COLUMN embedding_models.name IS '模型名称，如 text-embedding-v4, text-embedding-3-small';
COMMENT ON COLUMN embedding_models.provider IS '提供商，如 dashscope, openai';
COMMENT ON COLUMN embedding_models.dimensions IS '向量维度，如 256, 1536, 1024';

-- ============================================================
-- 2. 项目/仓库表
-- ============================================================
CREATE TABLE IF NOT EXISTS projects (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    repo_url        TEXT,
    owner           VARCHAR(255),
    repo_type       VARCHAR(50) DEFAULT 'gitee',
    local_path      TEXT,
    last_commit     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    metadata        JSONB DEFAULT '{}',
    UNIQUE(repo_url)
);

CREATE INDEX IF NOT EXISTS idx_projects_owner_name ON projects(owner, name);

COMMENT ON TABLE projects IS '项目/仓库表，管理多个代码仓库';
COMMENT ON COLUMN projects.last_commit IS '最近一次处理的 commit hash，用于增量更新检测';

-- ============================================================
-- 3. 原始文档表
-- ============================================================
CREATE TABLE IF NOT EXISTS raw_documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_path       TEXT NOT NULL,
    file_type       VARCHAR(20),
    content         TEXT NOT NULL,
    token_count     INT,
    is_code         BOOLEAN DEFAULT TRUE,
    is_deleted      BOOLEAN DEFAULT FALSE,
    content_sha256  VARCHAR(64),
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT unique_file_per_project UNIQUE (project_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_raw_documents_project ON raw_documents(project_id);
CREATE INDEX IF NOT EXISTS idx_raw_documents_file_type ON raw_documents(file_type);
CREATE INDEX IF NOT EXISTS idx_raw_documents_content_hash ON raw_documents(content_sha256);

COMMENT ON TABLE raw_documents IS '原始文档表，记录从仓库读取的每个文件';
COMMENT ON COLUMN raw_documents.is_deleted IS '逻辑删除标记，保留历史数据';
COMMENT ON COLUMN raw_documents.content_sha256 IS '内容 SHA-256 哈希，用于幂等去重';

-- ============================================================
-- 4. 文档版本表
-- ============================================================
CREATE TABLE IF NOT EXISTS document_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES raw_documents(id) ON DELETE CASCADE,
    git_commit_hash TEXT,
    content_hash    VARCHAR(64),
    content         TEXT NOT NULL,
    token_count     INT,
    change_type     VARCHAR(20) DEFAULT 'added',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_versions_document ON document_versions(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_versions_commit ON document_versions(git_commit_hash);

COMMENT ON TABLE document_versions IS '文档版本表，支持 Git 版本追踪和回滚';
COMMENT ON COLUMN document_versions.change_type IS '变更类型：added, modified, deleted';

-- ============================================================
-- 5. 文档分块表
-- ============================================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES raw_documents(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    chunk_size      INT,
    chunk_overlap   INT,
    split_by        VARCHAR(50),
    token_count     INT,
    start_offset    INT,
    end_offset      INT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);

COMMENT ON TABLE document_chunks IS '文档分块表，记录分块策略和每个块的内容';
COMMENT ON COLUMN document_chunks.start_offset IS '在原文中的字符偏移，用于溯源引用';

-- ============================================================
-- 6. 向量嵌入表（256 维，text-embedding-v4）
-- ============================================================
CREATE TABLE IF NOT EXISTS chunk_embeddings_dim256 (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model_id        UUID NOT NULL REFERENCES embedding_models(id),

    embedding       VECTOR(256) NOT NULL,

    -- 冗余字段：减少 JOIN，提升检索性能
    content         TEXT,
    file_path       TEXT,
    chunk_index     INT,

    -- 全文检索向量（自动生成）
    fts_text        TSVECTOR GENERATED ALWAYS AS
                    (to_tsvector('simple', COALESCE(content, ''))) STORED,

    created_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (chunk_id, model_id)
);

-- HNSW 向量索引（召回率与速度的最佳平衡）
CREATE INDEX IF NOT EXISTS idx_embeddings_dim256_hnsw ON chunk_embeddings_dim256
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- 全文检索 GIN 索引
CREATE INDEX IF NOT EXISTS idx_embeddings_dim256_fts ON chunk_embeddings_dim256
    USING gin(fts_text);

-- 项目 ID 索引
CREATE INDEX IF NOT EXISTS idx_embeddings_dim256_project ON chunk_embeddings_dim256(project_id);

-- 文件路径索引
CREATE INDEX IF NOT EXISTS idx_embeddings_dim256_file_path ON chunk_embeddings_dim256(file_path);

COMMENT ON TABLE chunk_embeddings_dim256 IS '向量嵌入表（256维），适用于 text-embedding-v4 模型';
COMMENT ON COLUMN chunk_embeddings_dim256.fts_text IS '自动生成的全文检索向量，用于混合检索';

-- ============================================================
-- 7. 代码符号表
-- ============================================================
CREATE TABLE IF NOT EXISTS code_symbols (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES raw_documents(id) ON DELETE CASCADE,
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    symbol_type     VARCHAR(50) NOT NULL,
    name            VARCHAR(255) NOT NULL,
    signature       TEXT,
    visibility      VARCHAR(20),
    start_line      INT,
    end_line        INT,
    parent_symbol_id UUID REFERENCES code_symbols(id),

    docstring       TEXT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_code_symbols_project ON code_symbols(project_id);
CREATE INDEX IF NOT EXISTS idx_code_symbols_document ON code_symbols(document_id);
CREATE INDEX IF NOT EXISTS idx_code_symbols_type ON code_symbols(symbol_type);
CREATE INDEX IF NOT EXISTS idx_code_symbols_name ON code_symbols(name);

COMMENT ON TABLE code_symbols IS '代码符号表，支持符号级精确检索';
COMMENT ON COLUMN code_symbols.symbol_type IS '符号类型：class, function, method, interface, variable';
COMMENT ON COLUMN code_symbols.parent_symbol_id IS '父符号 ID，支持嵌套结构（如类中的方法）';

-- ============================================================
-- 8. 摄取任务表
-- ============================================================
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID NOT NULL REFERENCES projects(id),

    trigger_type    TEXT NOT NULL DEFAULT 'manual',
    status          ingestion_status NOT NULL DEFAULT 'pending',

    current_stage   TEXT,
    progress        FLOAT DEFAULT 0,

    total_files     INT DEFAULT 0,
    processed_files INT DEFAULT 0,

    error_message   TEXT,
    error_detail    JSONB,

    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_project ON ingestion_jobs(project_id);

COMMENT ON TABLE ingestion_jobs IS '摄取任务表，异步处理核心，支持长任务状态追踪';
COMMENT ON COLUMN ingestion_jobs.trigger_type IS '触发方式：manual, webhook, scheduled';
COMMENT ON COLUMN ingestion_jobs.progress IS '处理进度 0.0 ~ 1.0';

-- ============================================================
-- 9. 检索记录表
-- ============================================================
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID REFERENCES projects(id),
    query_text      TEXT NOT NULL,
    query_embedding VECTOR(256),

    top_k           INT DEFAULT 5,
    retrieval_type  VARCHAR(50) DEFAULT 'vector_only',
    hybrid_weight   FLOAT DEFAULT 0.7,

    latency_ms      INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE retrieval_logs IS '检索记录表，记录每次检索的详细信息';
COMMENT ON COLUMN retrieval_logs.retrieval_type IS '检索策略：vector_only, hybrid, keyword_only';
COMMENT ON COLUMN retrieval_logs.hybrid_weight IS '语义搜索权重（0~1），仅 hybrid 模式有效';

-- ============================================================
-- 10. 检索结果明细表
-- ============================================================
CREATE TABLE IF NOT EXISTS retrieval_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    retrieval_id    UUID NOT NULL REFERENCES retrieval_logs(id) ON DELETE CASCADE,
    chunk_id        UUID REFERENCES document_chunks(id),
    rank            INT,
    vector_score    FLOAT,
    keyword_score   FLOAT,
    final_score     FLOAT,
    metadata        JSONB DEFAULT '{}',

    UNIQUE (retrieval_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_results_log ON retrieval_results(retrieval_id);

COMMENT ON TABLE retrieval_results IS '检索结果明细表，记录每次检索返回的每个结果及其得分';

-- ============================================================
-- 11. 问答记录表
-- ============================================================
CREATE TABLE IF NOT EXISTS qa_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    retrieval_id    UUID REFERENCES retrieval_logs(id),
    project_id      UUID REFERENCES projects(id),
    query_text      TEXT NOT NULL,
    response_text   TEXT,
    model_name      VARCHAR(100),
    prompt_tokens   INT,
    completion_tokens INT,
    total_tokens    INT,
    latency_ms      INT,
    user_rating     INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE qa_logs IS '问答记录表，记录 LLM 的最终回答及性能指标';
COMMENT ON COLUMN qa_logs.user_rating IS '用户评分（1-5），用于评估回答质量';

-- ============================================================
-- 12. 管道日志表
-- ============================================================
CREATE TABLE IF NOT EXISTS pipeline_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID REFERENCES projects(id),
    job_id          UUID REFERENCES ingestion_jobs(id),

    step_name       VARCHAR(100),
    status          VARCHAR(20),
    input_count     INT,
    output_count    INT,
    duration_ms     INT,
    error_message   TEXT,
    parameters      JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_logs_project ON pipeline_logs(project_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_step ON pipeline_logs(step_name);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_job ON pipeline_logs(job_id);

COMMENT ON TABLE pipeline_logs IS '管道日志表，记录 RAG 管道的每个执行步骤';

-- ============================================================
-- 13. Wiki 页面表
-- ============================================================
CREATE TABLE IF NOT EXISTS wiki_pages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    page_slug       TEXT NOT NULL,
    title           TEXT NOT NULL,
    content_md      TEXT,

    language        TEXT DEFAULT 'zh',
    is_comprehensive BOOLEAN DEFAULT TRUE,

    provider        TEXT,
    model           TEXT,

    source_chunks   JSONB,
    version         INT DEFAULT 1,

    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (project_id, page_slug, language)
);

CREATE INDEX IF NOT EXISTS idx_wiki_pages_project ON wiki_pages(project_id);

COMMENT ON TABLE wiki_pages IS 'Wiki 页面表，替代 JSONB blob，支持局部更新和版本管理';
COMMENT ON COLUMN wiki_pages.source_chunks IS '引用的文档块 ID 列表，用于溯源';

-- ============================================================
-- 14. 对话历史表
-- ============================================================
CREATE TABLE IF NOT EXISTS conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID REFERENCES projects(id) ON DELETE CASCADE,
    provider        TEXT,
    model           TEXT,
    language        TEXT DEFAULT 'zh',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_index      INT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(conversation_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_conversations_project ON conversations(project_id);
CREATE INDEX IF NOT EXISTS idx_conv_turns_conversation ON conversation_turns(conversation_id);

COMMENT ON TABLE conversations IS '对话历史表，持久化用户与 LLM 的对话记录';

-- ============================================================
-- 15. Wiki 缓存表（存储 Wiki 结构 + 元数据）
-- ============================================================
CREATE TABLE IF NOT EXISTS wiki_caches (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    language        TEXT NOT NULL DEFAULT 'zh',

    -- Wiki 结构（JSONB，包含 sections、page hierarchy 等）
    structure_json  JSONB NOT NULL DEFAULT '{}',

    -- 仓库信息
    repo_owner      TEXT,
    repo_name       TEXT,
    repo_type       TEXT DEFAULT 'github',
    repo_url        TEXT,

    -- LLM 配置
    provider        TEXT,
    model           TEXT,

    -- 时间戳
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (project_id, language)
);

CREATE INDEX IF NOT EXISTS idx_wiki_caches_project ON wiki_caches(project_id);

COMMENT ON TABLE wiki_caches IS 'Wiki 缓存表，存储 Wiki 结构（sections、page hierarchy）和元数据';
COMMENT ON COLUMN wiki_caches.structure_json IS 'Wiki 结构 JSON，包含 sections、pages 列表、rootSections 等';
COMMENT ON COLUMN wiki_caches.repo_url IS '仓库 URL，用于前端展示和链接';

-- ============================================================
-- 插入默认嵌入模型数据
-- ============================================================
INSERT INTO embedding_models (name, provider, dimensions, description) VALUES
    ('text-embedding-v4', 'dashscope', 256, '阿里云百炼文本嵌入 v4，256 维'),
    ('text-embedding-3-small', 'openai', 1536, 'OpenAI text-embedding-3-small，1536 维'),
    ('text-embedding-3-large', 'openai', 3072, 'OpenAI text-embedding-3-large，3072 维')
ON CONFLICT (name) DO NOTHING;

-- ============================================================
-- 验证安装
-- ============================================================
-- 检查扩展
SELECT name, default_version, installed_version
FROM pg_available_extensions
WHERE name IN ('vector', 'pgcrypto');

-- 检查表数量
SELECT COUNT(*) AS table_count FROM information_schema.tables
WHERE table_schema = 'public';
