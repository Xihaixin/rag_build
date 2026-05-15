"""
数据迁移脚本 v3 — 将 .pkl 文件数据完整迁移到 PostgreSQL

核心改进（对比 v1/v2）：
  1. 加载层：使用 patch_restore_value() + PklDumpUnpickler（来自 dump_pkl_data_03.py），
     正确跳过 Component._restore_value 的类实例恢复，所有 pkl 文件均能加载
  2. 数据提取：从 LocalDB.items 提取完整文件内容（raw_documents），
     从 LocalDB.transformed_items 提取分块+向量
  3. 写入层：建立 doc_id_map，分块阶段不再重新 upsert 文档，
     避免分块片段覆盖完整文件内容（v1 的核心 bug）

使用方法：
    python -m rag_optimizer.migration.pkl_to_pg_v3

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
)

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pkl_to_pg_v3")


# ============================================================
# 补丁：让 Component._restore_value 跳过组件恢复
# ============================================================
# adalflow 的 Component 使用 to_dict() / from_dict() 进行序列化，
# 反序列化时 _restore_value 会通过 EntityMapping 查找类并调用
# from_dict，这绕过了 Unpickler.find_class 的拦截。
#
# 解决方案：替换 _restore_value，让它只做简单的 dict/list 递归，
# 跳过 "type"+"data" 的类实例恢复逻辑。

_ORIGINAL_RESTORE_VALUE = None


def patch_restore_value():
    """替换 Component._restore_value 为安全版本"""
    global _ORIGINAL_RESTORE_VALUE
    try:
        from adalflow.core.component import Component

        _ORIGINAL_RESTORE_VALUE = Component._restore_value

        def safe_restore_value(value):
            """安全的 _restore_value：跳过类实例恢复，只处理 dict/list"""
            if isinstance(value, dict):
                if "_pickle_data" in value:
                    return pickle.loads(bytes.fromhex(value["_pickle_data"]))
                if "_ordered_dict" in value and value["_ordered_dict"]:
                    from collections import OrderedDict
                    return OrderedDict(
                        (safe_restore_value(k), safe_restore_value(v))
                        for k, v in value["data"]
                    )
                # 跳过 "type"+"data" 的类实例恢复，直接返回原始 dict
                return {k: safe_restore_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [safe_restore_value(v) for v in value]
            return value

        Component._restore_value = staticmethod(safe_restore_value)
        return True
    except ImportError:
        return False


def unpatch_restore_value():
    """恢复原始 Component._restore_value"""
    global _ORIGINAL_RESTORE_VALUE
    if _ORIGINAL_RESTORE_VALUE is not None:
        try:
            from adalflow.core.component import Component

            Component._restore_value = _ORIGINAL_RESTORE_VALUE
        except ImportError:
            pass


# ============================================================
# 自定义 Unpickler：拦截缺失模块
# ============================================================

class PklMigrateUnpickler(pickle.Unpickler):
    """
    安全的 Pickle 加载器。
    第一层防御：在 pickle 层面拦截缺失模块。
    第二层防御：patch_restore_value() 拦截 Component._restore_value。
    """

    # 需要正常还原的类（adalflow 核心类）
    REAL_CLASSES = {}

    @classmethod
    def _ensure_real_classes_loaded(cls):
        if cls.REAL_CLASSES:
            return
        try:
            from adalflow.core.db import LocalDB
            cls.REAL_CLASSES[('adalflow.core.db', 'LocalDB')] = LocalDB
        except ImportError:
            pass
        try:
            from adalflow.core.types import Document
            cls.REAL_CLASSES[('adalflow.core.types', 'Document')] = Document
        except ImportError:
            pass

    def find_class(self, module, name):
        self._ensure_real_classes_loaded()

        # 需要正常还原的类
        key = (module, name)
        if key in self.REAL_CLASSES:
            return self.REAL_CLASSES[key]

        # 缺失的非 adalflow 模块 → Dummy
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return self._make_dummy(module, name)

    @staticmethod
    def _make_dummy(module, name):
        class Dummy:
            def __init__(self, *args, **kwargs):
                pass

            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)

            def __getattr__(self, attr):
                return None

            def __call__(self, *args, **kwargs):
                return Dummy()

            def __repr__(self):
                return f"<Dummy {module}.{name}>"

        Dummy.__module__ = module
        Dummy.__qualname__ = name
        return Dummy


# ============================================================
# 安全的 pkl 加载器
# ============================================================

def safe_load_pkl(pkl_path: str):
    """
    安全加载 pkl 文件，双层防御：
      1. Component._restore_value 补丁：跳过 "type"+"data" 的类实例恢复
      2. 自定义 Unpickler：处理缺失的非 adalflow 模块
    """
    patched = patch_restore_value()
    try:
        with open(pkl_path, "rb") as f:
            db = PklMigrateUnpickler(f).load()
        return db
    finally:
        if patched:
            unpatch_restore_value()


# ============================================================
# 数据提取
# ============================================================

def extract_documents_from_db(db_obj) -> List[Dict[str, Any]]:
    """
    从 LocalDB 对象中提取原始文档数据（完整文件内容）。

    数据来源：LocalDB.items
    每个 item 包含：
      - text: 完整文件文本
      - meta_data: 元数据（file_path, type, is_code, is_implementation, token_count 等）
    """
    documents = []

    try:
        items = db_obj.items if hasattr(db_obj, 'items') else db_obj.get_items()
        for item in items:
            meta = getattr(item, 'meta_data', {}) or {}
            text = getattr(item, 'text', '')

            if not isinstance(meta, dict):
                meta = {}

            file_path = meta.get("file_path", "unknown")
            doc = {
                "file_path": file_path,
                "content": text,                          # ★ 完整文件文本
                "file_type": meta.get("type", Path(file_path).suffix.lstrip(".")),
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
    """
    从 LocalDB 对象中提取转换后的数据（分块 + 向量）。

    数据来源：LocalDB.transformed_items（dict，key 为 transformer 名称）
    每个 item 包含：
      - text: 分块文本
      - meta_data: 元数据（file_path, chunk_index, token_count 等）
      - vector: 嵌入向量（numpy array 或 list）
    """
    chunks = []

    try:
        # 遍历所有 transformer key
        transformed = None
        if hasattr(db_obj, 'transformed_items') and db_obj.transformed_items:
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

                if not isinstance(meta, dict):
                    meta = {}

                file_path = meta.get("file_path", "unknown")
                chunk = {
                    "content": text,                      # ★ 分块文本
                    "file_path": file_path,
                    "chunk_index": meta.get("chunk_index", idx),  # ★ 优先用 meta 中的索引
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

    # 1. 加载 pkl 数据（使用双层防御补丁）
    db_obj = safe_load_pkl(pkl_path)
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
    #    ★ 关键：建立 file_path -> doc_id 映射，供第 7 步使用
    #      避免分块阶段重新 upsert 导致完整内容被覆盖（v1 的核心 bug）
    # ====================================================================
    logger.info(f"Migrating {len(raw_docs)} raw documents...")
    doc_id_map = {}  # file_path -> doc_id
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

    # 按 file_path 分组，便于后续按 chunk_index 匹配
    chunks_by_file: Dict[str, List[Dict]] = {}
    for chunk in transformed_chunks:
        fp = chunk.get("file_path", "unknown")
        if fp not in chunks_by_file:
            chunks_by_file[fp] = []
        chunks_by_file[fp].append(chunk)

    for file_path, file_chunks in chunks_by_file.items():
        # ★ 从映射中获取 doc_id，不重新 upsert（避免覆盖完整内容）
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
                "metadata": {"source": "pkl_migration_v3"},
            })

        try:
            ChunkRepository.batch_insert(doc_id, chunk_data_list)
            chunk_count += len(file_chunks)
        except Exception as e:
            logger.error(f"Error batch inserting chunks for {file_path}: {e}")
            continue

        # 批量查询刚插入的 chunks，按 chunk_index 精确匹配写入 embedding
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
                # 处理向量格式（numpy array -> list）
                if hasattr(vector, 'tolist'):
                    vector_list = vector.tolist()
                elif isinstance(vector, (list, tuple)):
                    vector_list = list(vector)
                else:
                    logger.warning(f"Unknown vector type at chunk {chunk_index}: {type(vector)}")
                    continue

                # 跳过空向量（patch_restore_value 导致 numpy array 无法恢复）
                if not vector_list or (isinstance(vector_list, list) and len(vector_list) == 0):
                    logger.warning(f"Empty vector at chunk {chunk_index}, skipping embedding")
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
    # 8. 显式提交事务，验证才能看到数据
    # ====================================================================
    sync_conn._conn.commit()
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
        step_name="pkl_migration_v3", status="completed",
        input_count=len(raw_docs), output_count=chunk_count,
        duration_ms=elapsed,
        parameters={
            "pkl_path": pkl_path,
            "raw_docs": len(raw_docs),
            "chunks": len(transformed_chunks),
            "embeddings": embed_count,
        }
    )

    # 再次 commit，确保日志和状态也落盘
    sync_conn._conn.commit()

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


def main():
    """
    调试版本：直接在代码里填写参数运行。

    使用方法：
        1. 修改下方 pkl_path 为你的 .pkl 文件路径
        2. 运行：python -m rag_optimizer.migration.pkl_to_pg_v3
    """
    # ====================== 【手动填写参数】 ======================
    pkl_path = r"C:\Users\lenovo\AppData\Roaming\adalflow\databases\MathModelAgent.pkl"
    project_name = "MathModelAgent"
    repo_url = "https://github.com/Xihaixin/MathModelAgent"  # 不知道就填 None
    verify = True    # 迁移后自动校验
    # ==============================================================

    # 执行迁移
    success = migrate_pkl_to_postgresql(
        pkl_path=pkl_path,
        project_name=project_name,
        repo_url=repo_url,
    )

    if success and verify:
        # 获取项目 ID
        project = sync_conn.execute(
            "SELECT id FROM projects WHERE name = %s",
            (project_name or Path(pkl_path).stem,)
        )
        if project:
            verify_migration(project[0]["id"])

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
