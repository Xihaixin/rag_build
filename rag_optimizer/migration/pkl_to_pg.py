"""
数据迁移脚本：将现有的 .pkl 文件数据迁移到 PostgreSQL

该脚本读取通过 load_pkl.py 成功加载的 gitingest.pkl 数据，
将其中的文档、分块和向量数据写入 PostgreSQL 数据库。

使用方法：
    python -m rag_optimizer.migration.pkl_to_pg --pkl-path ./gitingest.pkl

前置条件：
    1. PostgreSQL 已运行且 schema 已创建
    2. 依赖已安装：pip install psycopg2-binary
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_optimizer.config.settings import settings
from rag_optimizer.db.connection import sync_conn
from rag_optimizer.db.repository import (
    ProjectRepository,
    DocumentRepository,
    ChunkRepository,
    EmbeddingRepository,
    IngestionJobRepository,
    PipelineLogRepository,
    compute_sha256,
    vector_to_str,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pkl_to_pg")


# ============================================================
# SafeUnpickler — 安全的 pickle 加载器
# ============================================================

class SafeUnpickler(pickle.Unpickler):
    """安全的 Pickle 加载器，缺失模块时返回占位对象"""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError) as e:
            logger.warning(f"Missing module/class: {module}.{name} -> {e}, using placeholder")

            class Placeholder:
                def __init__(self, *args, **kwargs):
                    pass

                def __call__(self, *args, **kwargs):
                    return Placeholder()

                def __getattr__(self, name):
                    return Placeholder()

                def __repr__(self):
                    return f"<Missing {module}.{name}>"

            return Placeholder


# ============================================================
# 数据提取
# ============================================================

def load_pkl_data(pkl_path_str: str) -> Optional[Any]:
    """加载 .pkl 文件"""
    pkl_path = Path(pkl_path_str)
    if not pkl_path.exists():
        logger.error(f"PKL file not found: {pkl_path}")
        return None

    logger.info(f"Loading pickle file: {pkl_path} ({pkl_path.stat().st_size / 1024:.1f} KB)")

    try:
        # 方法1: 使用 SafeUnpickler
        with open(pkl_path, "rb") as f:
            data = SafeUnpickler(f).load()
        logger.info(f"Loaded with SafeUnpickler: {type(data).__name__}")
        return data
    except Exception as e:
        logger.warning(f"SafeUnpickler failed: {e}")

        try:
            # 方法2: 替换 LocalDB.__setstate__
            from adalflow_localdb import LocalDB

            original_setstate = LocalDB.__setstate__
            LocalDB.__setstate__ = lambda self, state: self.__dict__.update(state)

            try:
                with open(pkl_path, "rb") as f:
                    data = SafeUnpickler(f).load()
                logger.info(f"Loaded with __setstate__ patch: {type(data).__name__}")
                return data
            finally:
                LocalDB.__setstate__ = original_setstate
        except Exception as e2:
            logger.error(f"All loading methods failed: {e2}")
            return None


def extract_documents_from_db(db_obj) -> List[Dict[str, Any]]:
    """从 LocalDB 对象中提取文档数据"""
    documents = []

    # 尝试获取原始文档
    try:
        items = db_obj.get_items() if hasattr(db_obj, 'get_items') else db_obj.items
        for item in items:
            doc = {
                "file_path": getattr(item, 'file_path', getattr(item, 'meta_data', {}).get('file_path', 'unknown')),
                "content": getattr(item, 'text', getattr(item, 'content', str(item))),
                "token_count": getattr(item, 'token_count', 0),
            }
            # 提取元数据
            meta = getattr(item, 'meta_data', {}) or {}
            if isinstance(meta, dict):
                doc["file_path"] = meta.get("file_path", doc["file_path"])
                doc["file_type"] = meta.get("file_type", Path(doc["file_path"]).suffix.lstrip("."))
                doc["is_code"] = meta.get("is_code", doc["file_type"] in ("py", "js", "ts", "java", "cpp", "go", "rs"))
            else:
                doc["file_type"] = Path(doc["file_path"]).suffix.lstrip(".")
                doc["is_code"] = doc["file_type"] in ("py", "js", "ts", "java", "cpp", "go", "rs")

            documents.append(doc)
    except Exception as e:
        logger.warning(f"Error extracting items: {e}")

    logger.info(f"Extracted {len(documents)} raw documents")
    return documents


def extract_transformed_data(db_obj) -> List[Dict[str, Any]]:
    """从 LocalDB 对象中提取转换后的数据（分块 + 向量）"""
    chunks = []

    try:
        # 尝试获取 transformed data
        transformed = None
        if hasattr(db_obj, 'get_transformed_data'):
            transformed = db_obj.get_transformed_data(key="split_and_embed")
        elif hasattr(db_obj, '_transformed_data'):
            transformed = db_obj._transformed_data.get("split_and_embed")

        if transformed:
            for doc in transformed:
                chunk = {
                    "content": getattr(doc, 'text', getattr(doc, 'content', '')),
                    "file_path": getattr(doc, 'file_path', 'unknown'),
                    "chunk_index": getattr(doc, 'chunk_index', 0),
                    "token_count": getattr(doc, 'token_count', 0),
                    "vector": getattr(doc, 'vector', None),
                }
                # 尝试从 meta_data 获取更多信息
                meta = getattr(doc, 'meta_data', {}) or {}
                if isinstance(meta, dict):
                    chunk["file_path"] = meta.get("file_path", chunk["file_path"])
                    chunk["chunk_index"] = meta.get("chunk_index", chunk["chunk_index"])

                chunks.append(chunk)
    except Exception as e:
        logger.warning(f"Error extracting transformed data: {e}")

    logger.info(f"Extracted {len(chunks)} transformed chunks")
    return chunks


# ============================================================
# 迁移主逻辑
# ============================================================

def migrate_pkl_to_postgresql(pkl_path: str, project_name: Optional[str] = None,
                               repo_url: Optional[str] = None):
    """将 .pkl 文件数据迁移到 PostgreSQL"""
    start_time = time.time()
    logger.info(f"=" * 60)
    logger.info(f"Starting migration: {pkl_path}")
    logger.info(f"=" * 60)

    # 1. 加载 pkl 数据
    db_obj = load_pkl_data(pkl_path)
    if db_obj is None:
        logger.error("Failed to load pkl data. Aborting.")
        return False

    # 2. 提取数据
    raw_docs = extract_documents_from_db(db_obj)
    transformed_chunks = extract_transformed_data(db_obj)

    if not raw_docs and not transformed_chunks:
        logger.error("No data extracted from pkl file. Aborting.")
        return False

    # 3. 创建项目
    if not project_name:
        project_name = Path(pkl_path).stem  # 使用文件名作为项目名
    if not repo_url:
        repo_url = f"local://{project_name}"

    project = ProjectRepository.get_or_create(
        name=project_name,
        repo_url=repo_url,
        repo_type="local",
        local_path=str(Path(pkl_path).parent),
    )
    project_id = project["id"]
    logger.info(f"Project: {project_name} (id={project_id})")

    # 4. 创建摄取任务
    job = IngestionJobRepository.create(project_id, trigger_type="migration")
    job_id = job["id"]
    IngestionJobRepository.update_status(job_id, "started", stage="parsing")

    # 5. 获取嵌入模型 ID
    model_name = settings.embedding.default_model
    model_id = EmbeddingRepository.get_model_id(model_name)
    if not model_id:
        logger.warning(f"Embedding model '{model_name}' not found in DB. "
                       f"Please run the schema script first.")
        # 尝试插入默认模型
        from rag_optimizer.db.connection import sync_conn as conn
        conn.execute(
            """INSERT INTO embedding_models (name, provider, dimensions)
               VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING""",
            (model_name, settings.embedding.default_provider, settings.embedding.default_dimensions)
        )
        model_id = EmbeddingRepository.get_model_id(model_name)

    logger.info(f"Using embedding model: {model_name} (id={model_id})")

    # 6. 迁移原始文档
    logger.info(f"Migrating {len(raw_docs)} raw documents...")
    doc_count = 0
    for i, doc in enumerate(raw_docs):
        try:
            file_path = doc.get("file_path", f"unknown_{i}")
            content = doc.get("content", "")
            file_type = doc.get("file_type", Path(file_path).suffix.lstrip("."))
            is_code = doc.get("is_code", file_type in ("py", "js", "ts", "java", "cpp", "go", "rs"))
            token_count = doc.get("token_count", 0)

            doc_id, changed = DocumentRepository.upsert(
                project_id=project_id,
                file_path=file_path,
                content=content,
                file_type=file_type,
                is_code=is_code,
                token_count=token_count,
            )
            if changed:
                doc_count += 1
        except Exception as e:
            logger.warning(f"Error migrating document {i}: {e}")

    logger.info(f"Migrated {doc_count} documents (new/changed)")

    # 7. 迁移分块和向量
    logger.info(f"Migrating {len(transformed_chunks)} chunks with embeddings...")
    chunk_count = 0
    embed_count = 0

    for i, chunk in enumerate(transformed_chunks):
        try:
            content = chunk.get("content", "")
            file_path = chunk.get("file_path", "unknown")
            chunk_index = chunk.get("chunk_index", i)
            vector = chunk.get("vector", None)

            if not content:
                continue

            # 查找或创建文档
            doc_id, _ = DocumentRepository.upsert(
                project_id=project_id,
                file_path=file_path,
                content=content,
                token_count=chunk.get("token_count", 0),
            )

            # 插入分块
            chunk_data = [{
                "chunk_index": chunk_index,
                "content": content,
                "chunk_size": settings.chunk.default_chunk_size,
                "chunk_overlap": settings.chunk.default_chunk_overlap,
                "split_by": settings.chunk.default_split_by,
                "token_count": chunk.get("token_count", 0),
                "start_offset": None,
                "end_offset": None,
                "metadata": {"source": "pkl_migration", "original_index": i},
            }]
            ChunkRepository.batch_insert(doc_id, chunk_data)
            chunk_count += 1

            # 获取 chunk_id
            chunks_in_db = ChunkRepository.get_by_document(doc_id)
            if chunks_in_db:
                db_chunk = chunks_in_db[0]
                chunk_db_id = db_chunk["id"]

                # 插入向量
                if vector is not None and model_id:
                    # 处理向量格式
                    if hasattr(vector, 'tolist'):
                        vector_list = vector.tolist()
                    elif isinstance(vector, (list, tuple)):
                        vector_list = list(vector)
                    else:
                        logger.warning(f"Unknown vector type: {type(vector)}")
                        continue

                    EmbeddingRepository.insert(
                        chunk_id=chunk_db_id,
                        project_id=project_id,
                        model_id=model_id,
                        embedding=vector_list,
                        content=content,
                        file_path=file_path,
                        chunk_index=chunk_index,
                    )
                    embed_count += 1

        except Exception as e:
            logger.warning(f"Error migrating chunk {i}: {e}")

    logger.info(f"Migrated {chunk_count} chunks, {embed_count} embeddings")

    # 8. 更新任务状态
    IngestionJobRepository.update_status(
        job_id, "completed", stage="completed",
        progress=1.0, processed=chunk_count, total=chunk_count
    )

    # 9. 记录管道日志
    elapsed = int((time.time() - start_time) * 1000)
    PipelineLogRepository.log(
        project_id=project_id, job_id=job_id,
        step_name="pkl_migration", status="completed",
        input_count=len(raw_docs), output_count=chunk_count,
        duration_ms=elapsed,
        parameters={
            "pkl_path": pkl_path,
            "raw_docs": len(raw_docs),
            "chunks": len(transformed_chunks),
            "embeddings": embed_count,
        }
    )

    # 10. 统计
    logger.info(f"=" * 60)
    logger.info(f"Migration completed!")
    logger.info(f"  Project:     {project_name} (id={project_id})")
    logger.info(f"  Documents:   {doc_count}")
    logger.info(f"  Chunks:      {chunk_count}")
    logger.info(f"  Embeddings:  {embed_count}")
    logger.info(f"  Duration:    {elapsed / 1000:.2f}s")
    logger.info(f"=" * 60)

    return True


# ============================================================
# 验证迁移结果
# ============================================================

def verify_migration(project_id: str):
    """验证迁移结果"""
    logger.info("Verifying migration...")

    # 检查文档
    docs = sync_conn.execute(
        "SELECT COUNT(*) as count FROM raw_documents WHERE project_id = %s",
        (project_id,)
    )
    doc_count = docs[0]["count"] if docs else 0

    # 检查分块
    chunks = sync_conn.execute(
        """SELECT COUNT(*) as count FROM document_chunks dc
           JOIN raw_documents rd ON dc.document_id = rd.id
           WHERE rd.project_id = %s""",
        (project_id,)
    )
    chunk_count = chunks[0]["count"] if chunks else 0

    # 检查嵌入
    embeddings = sync_conn.execute(
        """SELECT COUNT(*) as count FROM chunk_embeddings_dim256
           WHERE project_id = %s""",
        (project_id,)
    )
    embed_count = embeddings[0]["count"] if embeddings else 0

    logger.info(f"Verification results:")
    logger.info(f"  Documents:  {doc_count}")
    logger.info(f"  Chunks:     {chunk_count}")
    logger.info(f"  Embeddings: {embed_count}")

    return {
        "documents": doc_count,
        "chunks": chunk_count,
        "embeddings": embed_count,
    }


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Migrate .pkl file data to PostgreSQL + pgvector"
    )
    parser.add_argument(
        "--pkl-path", type=str, required=True,
        help="Path to the .pkl file (e.g., ./gitingest.pkl)"
    )
    parser.add_argument(
        "--project-name", type=str, default=None,
        help="Project name (default: pkl filename without extension)"
    )
    parser.add_argument(
        "--repo-url", type=str, default=None,
        help="Repository URL (default: local://project_name)"
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Verify migration results after completion"
    )

    args = parser.parse_args()

    # 执行迁移
    success = migrate_pkl_to_postgresql(
        pkl_path=args.pkl_path,
        project_name=args.project_name,
        repo_url=args.repo_url,
    )

    if success and args.verify:
        # 获取项目 ID
        project = sync_conn.execute(
            "SELECT id FROM projects WHERE name = %s",
            (args.project_name or Path(args.pkl_path).stem,)
        )
        if project:
            verify_migration(project[0]["id"])

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
