"""
数据迁移脚本：将现有的 .pkl 文件数据迁移到 PostgreSQL
 
该脚本读取通过 load_pkl.py 成功加载的 gitingest.pkl 数据，
将其中的文档、分块和向量数据写入 PostgreSQL 数据库。
 
修复项：
    1. 验证前显式 commit，避免未提交事务导致验证结果全为 0
    2. 建立 doc_id_map，分块阶段不再重新 upsert 文档（避免覆盖完整内容）
    3. 按 chunk_index 精确匹配 chunk，而非 [0] 硬编码
 
使用方法：
    python -m rag_optimizer.migration.pkl_to_pg --pkl-path ./gitingest.pkl
 
前置条件：
    1. PostgreSQL 已运行且 schema 已创建
    2. 依赖已安装：pip install psycopg2-binary
"""
 
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
            from adalflow.core import LocalDB
 
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
    """从 LocalDB 对象中提取原始文档数据（完整文件内容）"""
    documents = []
 
    try:
        items = db_obj.items if hasattr(db_obj, 'items') else db_obj.get_items()
        for item in items:
            meta = getattr(item, 'meta_data', {}) or {}
            text = getattr(item, 'text', '')
 
            doc = {
                "file_path": meta.get("file_path", "unknown"),
                "content": text,  # ★ 完整文件文本
                "file_type": meta.get("type", Path(meta.get("file_path", "x.")).suffix.lstrip(".")),
                "is_code": meta.get("is_code", False),
                "is_implementation": meta.get("is_implementation", False),
                "token_count": meta.get("token_count", 0),
            }
            documents.append(doc)
    except Exception as e:
        logger.warning(f"Error extracting items: {e}")
 
    logger.info(f"Extracted {len(documents)} raw documents")
    return documents
 
 
def extract_transformed_data(db_obj) -> List[Dict[str, Any]]:
    """从 LocalDB 对象中提取转换后的数据（分块 + 向量）"""
    chunks = []
 
    try:
        # 遍历所有 transformer key，不硬编码
        transformed = None
        if hasattr(db_obj, 'transformed_items') and db_obj.transformed_items:
            # 取第一个 key 的数据（通常就是 split_and_embed）
            for key in db_obj.transformed_items:
                transformed = db_obj.transformed_items[key]
                logger.info(f"Found transformed data under key: '{key}'")
                break
        elif hasattr(db_obj, 'get_transformed_data'):
            transformed = db_obj.get_transformed_data(key="split_and_embed")
 
        if transformed:
            for idx, doc in enumerate(transformed):
                meta = getattr(doc, 'meta_data', {}) or {}
                text = getattr(doc, 'text', '')
                vector = getattr(doc, 'vector', None)
 
                chunk = {
                    "content": text,        # ★ 分块文本
                    "file_path": meta.get("file_path", "unknown"),
                    "chunk_index": idx,     # ★ 用遍历索引作为 chunk_index
                    "token_count": meta.get("token_count", 0),
                    "is_code": meta.get("is_code", False),
                    "vector": vector,
                }
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
        project_name = Path(pkl_path).stem
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
    IngestionJobRepository.update_status(job_id, "cloning", stage="parsing")
 
    # 5. 获取嵌入模型 ID
    model_name = settings.embedding.default_model
    model_id = EmbeddingRepository.get_model_id(model_name)
    if not model_id:
        logger.warning(f"Embedding model '{model_name}' not found in DB. Inserting default...")
        sync_conn.execute(
            """INSERT INTO embedding_models (name, provider, dimensions)
               VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING""",
            (model_name, settings.embedding.default_provider, settings.embedding.default_dimensions)
        )
        model_id = EmbeddingRepository.get_model_id(model_name)
 
    logger.info(f"Using embedding model: {model_name} (id={model_id})")
 
    # ====================================================================
    # 6. 迁移原始文档（完整文件内容）
    #    ★ 修复：建立 file_path -> doc_id 映射，供第 7 步使用
    # ====================================================================
    logger.info(f"Migrating {len(raw_docs)} raw documents...")
    doc_id_map = {}  # ★ file_path -> doc_id
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
            doc_id_map[file_path] = doc_id  # ★ 记录映射
            if changed:
                doc_count += 1
        except Exception as e:
            logger.warning(f"Error migrating document {i}: {e}")
 
    logger.info(f"Migrated {doc_count} documents (new/changed)")
    logger.info(f"doc_id_map contains {len(doc_id_map)} file paths")
 
    # ====================================================================
    # 7. 迁移分块和向量
    #    ★ 修复1：通过 doc_id_map 查找 doc_id，不再重新 upsert
    #    ★ 修复2：按 chunk_index 精确匹配，而非 [0] 硬编码
    # ====================================================================
    logger.info(f"Migrating {len(transformed_chunks)} chunks with embeddings...")
    chunk_count = 0
    embed_count = 0
 
    # ★ 按 file_path 分组，便于后续按 chunk_index 匹配
    chunks_by_file: Dict[str, List[Dict]] = {}
    for chunk in transformed_chunks:
        fp = chunk.get("file_path", "unknown")
        if fp not in chunks_by_file:
            chunks_by_file[fp] = []
        chunks_by_file[fp].append(chunk)
 
    for file_path, file_chunks in chunks_by_file.items():
        # ★ 从映射中获取 doc_id，不重新 upsert
        doc_id = doc_id_map.get(file_path)
        if not doc_id:
            # 如果分块对应的文件不在原始文档中，才创建新文档
            first_chunk = file_chunks[0]
            doc_id, _ = DocumentRepository.upsert(
                project_id=project_id,
                file_path=file_path,
                content=first_chunk.get("content", ""),
                file_type=Path(file_path).suffix.lstrip("."),
                is_code=first_chunk.get("is_code", False),
                token_count=first_chunk.get("token_count", 0),
            )
            doc_id_map[file_path] = doc_id
            logger.warning(f"Created missing document for chunks: {file_path}")
 
        # 批量插入该文件的所有分块
        chunk_data_list = []
        for chunk in file_chunks:
            chunk_data_list.append({
                "chunk_index": chunk["chunk_index"],
                "content": chunk.get("content", ""),
                "chunk_size": settings.chunk.default_chunk_size,
                "chunk_overlap": settings.chunk.default_chunk_overlap,
                "split_by": settings.chunk.default_split_by,
                "token_count": chunk.get("token_count", 0),
                "start_offset": None,
                "end_offset": None,
                "metadata": {"source": "pkl_migration"},
            })
 
        try:
            ChunkRepository.batch_insert(doc_id, chunk_data_list)
            chunk_count += len(file_chunks)
        except Exception as e:
            logger.error(f"Error batch inserting chunks for {file_path}: {e}")
            continue
 
        # ★ 批量查询刚插入的 chunks，按 chunk_index 精确匹配写入 embedding
        try:
            chunks_in_db = ChunkRepository.get_by_document(doc_id)
        except Exception as e:
            logger.error(f"Error fetching chunks for doc {doc_id}: {e}")
            continue
 
        # 建立 chunk_index -> db_chunk_id 映射
        db_chunk_map = {}
        for c in chunks_in_db:
            ci = c.get("chunk_index")
            if ci is not None:
                db_chunk_map[ci] = c["id"]
 
        # 为每个分块写入 embedding
        for chunk in file_chunks:
            chunk_index = chunk["chunk_index"]
            vector = chunk.get("vector")
            content = chunk.get("content", "")
 
            chunk_db_id = db_chunk_map.get(chunk_index)
            if not chunk_db_id:
                logger.warning(f"chunk_index {chunk_index} not found in DB for doc {doc_id}, skipping embedding")
                continue
 
            if vector is not None and model_id:
                # 处理向量格式
                if hasattr(vector, 'tolist'):
                    vector_list = vector.tolist()
                elif isinstance(vector, (list, tuple)):
                    vector_list = list(vector)
                else:
                    logger.warning(f"Unknown vector type at chunk {chunk_index}: {type(vector)}")
                    continue
 
                try:
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
                    logger.warning(f"Error inserting embedding for chunk {chunk_index}: {e}")
 
    logger.info(f"Migrated {chunk_count} chunks, {embed_count} embeddings")
 
    # ====================================================================
    # 8. ★ 修复：显式提交事务，验证才能看到数据
    # ====================================================================
    sync_conn.commit()
    logger.info("Transaction committed")
 
    # 9. 更新任务状态
    IngestionJobRepository.update_status(
        job_id, "completed", stage="completed",
        progress=1.0, processed=chunk_count, total=chunk_count
    )
 
    # 10. 记录管道日志
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
 
    # ★ 再次 commit，确保日志和状态也落盘
    sync_conn.commit()
 
    # 11. 统计
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
 
    # ★ 额外验证：检查文档内容长度是否合理（不是被截断的分块片段）
    if doc_count > 0:
        content_check = sync_conn.execute(
            """SELECT file_path, LENGTH(content) as content_len
               FROM raw_documents WHERE project_id = %s
               ORDER BY content_len DESC LIMIT 5""",
            (project_id,)
        )
        logger.info(f"  Document content lengths (top 5):")
        for row in content_check:
            logger.info(f"    {row['file_path']}: {row['content_len']} chars")
 
    # 检查分块
    chunks = sync_conn.execute(
        """SELECT COUNT(*) as count FROM document_chunks dc
           JOIN raw_documents rd ON dc.document_id = rd.id
           WHERE rd.project_id = %s""",
        (project_id,)
    )
    chunk_count = chunks[0]["count"] if chunks else 0
 
    # ★ 额外验证：检查每个文档的分块数是否合理
    if chunk_count > 0:
        chunk_dist = sync_conn.execute(
            """SELECT rd.file_path, COUNT(dc.id) as chunk_count
               FROM raw_documents rd
               LEFT JOIN document_chunks dc ON dc.document_id = rd.id
               WHERE rd.project_id = %s
               GROUP BY rd.file_path
               ORDER BY chunk_count DESC LIMIT 10""",
            (project_id,)
        )
        logger.info(f"  Chunks per document (top 10):")
        for row in chunk_dist:
            logger.info(f"    {row['file_path']}: {row['chunk_count']} chunks")
 
    # 检查嵌入
    embeddings = sync_conn.execute(
        """SELECT COUNT(*) as count FROM chunk_embeddings_dim256
           WHERE project_id = %s""",
        (project_id,)
    )
    embed_count = embeddings[0]["count"] if embeddings else 0
 
    # ★ 额外验证：embedding 数量应等于 chunk 数量
    if embed_count != chunk_count:
        logger.warning(f"  ⚠ Embedding count ({embed_count}) != Chunk count ({chunk_count})")
 
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
# 调试版本：直接在代码里填写参数
# ============================================================
def main():
    # ====================== 【手动填写参数】 ======================
    pkl_path = r"C:\Users\lenovo\AppData\Roaming\adalflow\databases\StarScout.pkl"
    project_name = "gitingest"
    repo_url = "https://github.com/Xihaixin/StarCount"
    verify = True
    # ==============================================================
 
    success = migrate_pkl_to_postgresql(
        pkl_path=pkl_path,
        project_name=project_name,
        repo_url=repo_url,
    )
 
    if success and verify:
        project = sync_conn.execute(
            "SELECT id FROM projects WHERE name = %s",
            (project_name or Path(pkl_path).stem,)
        )
        if project:
            verify_migration(project[0]["id"])
 
    return 0 if success else 1
 
 
if __name__ == "__main__":
    sys.exit(main())