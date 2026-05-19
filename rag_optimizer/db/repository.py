"""
数据访问层 — Repository 模式

封装对数据库的 CRUD 操作，提供高层 API 供业务逻辑调用。
支持同步和异步两种模式。
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rag_optimizer.config.settings import settings
from rag_optimizer.db.connection import sync_conn, async_pool

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def compute_sha256(content: str) -> str:
    """计算内容的 SHA-256 哈希"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def vector_to_str(vector: List[float]) -> str:
    """将向量列表转为 PostgreSQL vector 文本格式 '[x,y,z]'"""
    return "[" + ",".join(str(v) for v in vector) + "]"


# ============================================================
# 项目 Repository
# ============================================================

class ProjectRepository:
    """项目数据访问"""

    @staticmethod
    def get_or_create(name: str, repo_url: Optional[str] = None,
                      owner: Optional[str] = None, repo_type: str = "gitee",
                      local_path: Optional[str] = None) -> dict:
        """获取或创建项目"""
        if repo_url:
            existing = sync_conn.execute(
                "SELECT * FROM projects WHERE repo_url = %s", (repo_url,)
            )
            if existing:
                logger.info(f"Found existing project: {repo_url}")
                return dict(existing[0])

        result = sync_conn.execute(
            """INSERT INTO projects (name, repo_url, owner, repo_type, local_path)
               VALUES (%s, %s, %s, %s, %s)
               RETURNING *""",
            (name, repo_url, owner, repo_type, local_path)
        )
        logger.info(f"Created project: {name}")
        return dict(result[0])

    @staticmethod
    def get_by_id(project_id: str) -> Optional[dict]:
        """根据 ID 获取项目"""
        result = sync_conn.execute(
            "SELECT * FROM projects WHERE id = %s", (project_id,)
        )
        return dict(result[0]) if result else None

    @staticmethod
    def list_all() -> List[dict]:
        """列出所有项目"""
        result = sync_conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        )
        return [dict(r) for r in result] if result else []

    @staticmethod
    def update_last_commit(project_id: str, commit_hash: str):
        """更新最近 commit"""
        sync_conn.execute(
            "UPDATE projects SET last_commit = %s, updated_at = NOW() WHERE id = %s",
            (commit_hash, project_id)
        )


# ============================================================
# 文档 Repository
# ============================================================

class DocumentRepository:
    """文档数据访问"""

    @staticmethod
    def upsert(project_id: str, file_path: str, content: str,
               file_type: Optional[str] = None, is_code: bool = True,
               token_count: Optional[int] = None) -> Tuple[str, bool]:
        """
        插入或更新文档。
        返回 (document_id, is_changed) — is_changed 表示内容是否有变更。
        """
        content_hash = compute_sha256(content)

        # 检查是否已存在
        existing = sync_conn.execute(
            """SELECT id, content_sha256 FROM raw_documents
               WHERE project_id = %s AND file_path = %s""",
            (project_id, file_path)
        )

        if existing:
            doc = existing[0]
            if doc["content_sha256"] == content_hash:
                # 内容未变更
                return str(doc["id"]), False

            # 内容已变更：更新文档
            sync_conn.execute(
                """UPDATE raw_documents SET content = %s, content_sha256 = %s,
                   token_count = %s, is_deleted = FALSE, file_type = %s
                   WHERE id = %s""",
                (content, content_hash, token_count, file_type, doc["id"])
            )
            # 记录版本
            sync_conn.execute(
                """INSERT INTO document_versions (document_id, content_hash, content, token_count, change_type)
                   VALUES (%s, %s, %s, %s, 'modified')""",
                (doc["id"], content_hash, content, token_count)
            )
            return str(doc["id"]), True

        # 新建文档
        result = sync_conn.execute(
            """INSERT INTO raw_documents (project_id, file_path, file_type, content,
                                          token_count, is_code, content_sha256)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (project_id, file_path, file_type, content, token_count, is_code, content_hash)
        )
        doc_id = str(result[0]["id"])

        # 记录版本
        sync_conn.execute(
            """INSERT INTO document_versions (document_id, content_hash, content, token_count, change_type)
               VALUES (%s, %s, %s, %s, 'added')""",
            (doc_id, content_hash, content, token_count)
        )
        return doc_id, True

    @staticmethod
    def soft_delete(project_id: str, file_path: str):
        """逻辑删除文档"""
        sync_conn.execute(
            "UPDATE raw_documents SET is_deleted = TRUE WHERE project_id = %s AND file_path = %s",
            (project_id, file_path)
        )

    @staticmethod
    def get_by_project(project_id: str, include_deleted: bool = False) -> List[dict]:
        """获取项目的所有文档"""
        query = "SELECT * FROM raw_documents WHERE project_id = %s"
        if not include_deleted:
            query += " AND is_deleted = FALSE"
        query += " ORDER BY file_path"
        result = sync_conn.execute(query, (project_id,))
        return [dict(r) for r in result] if result else []


# ============================================================
# 分块 Repository
# ============================================================

class ChunkRepository:
    """文档分块数据访问"""

    @staticmethod
    def batch_insert(document_id: str, chunks: List[Dict[str, Any]]):
        """批量插入分块"""
        # 先删除旧分块
        sync_conn.execute(
            "DELETE FROM document_chunks WHERE document_id = %s", (document_id,)
        )

        for chunk in chunks:
            sync_conn.execute(
                """INSERT INTO document_chunks
                   (document_id, chunk_index, content, chunk_size, chunk_overlap,
                    split_by, token_count, start_offset, end_offset, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)""",
                (
                    document_id,
                    chunk["chunk_index"],
                    chunk["content"],
                    chunk.get("chunk_size"),
                    chunk.get("chunk_overlap"),
                    chunk.get("split_by"),
                    chunk.get("token_count"),
                    chunk.get("start_offset"),
                    chunk.get("end_offset"),
                    json.dumps(chunk.get("metadata", {})),
                )
            )

    @staticmethod
    def get_by_document(document_id: str) -> List[dict]:
        """获取文档的所有分块"""
        result = sync_conn.execute(
            "SELECT * FROM document_chunks WHERE document_id = %s ORDER BY chunk_index",
            (document_id,)
        )
        return [dict(r) for r in result] if result else []


# ============================================================
# 嵌入 Repository
# ============================================================

class EmbeddingRepository:
    """向量嵌入数据访问"""

    @staticmethod
    def insert(chunk_id: str, project_id: str, model_id: str,
               embedding: List[float], content: Optional[str] = None,
               file_path: Optional[str] = None, chunk_index: Optional[int] = None):
        """插入向量嵌入"""
        embedding_str = vector_to_str(embedding)
        sync_conn.execute(
            """INSERT INTO chunk_embeddings_dim256
               (chunk_id, project_id, model_id, embedding, content, file_path, chunk_index)
               VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
               ON CONFLICT (chunk_id, model_id) DO UPDATE SET
               embedding = %s::vector, content = %s, file_path = %s, chunk_index = %s""",
            (chunk_id, project_id, model_id, embedding_str,
             content, file_path, chunk_index,
             embedding_str, content, file_path, chunk_index)
        )

    @staticmethod
    def get_model_id(model_name: str) -> Optional[str]:
        """获取嵌入模型的 ID"""
        result = sync_conn.execute(
            "SELECT id FROM embedding_models WHERE name = %s", (model_name,)
        )
        return str(result[0]["id"]) if result else None


# ============================================================
# 摄取任务 Repository
# ============================================================

class IngestionJobRepository:
    """摄取任务数据访问"""

    @staticmethod
    def create(project_id: str, trigger_type: str = "manual") -> dict:
        """创建摄取任务"""
        result = sync_conn.execute(
            """INSERT INTO ingestion_jobs (project_id, trigger_type, status)
               VALUES (%s, %s, 'pending')
               RETURNING *""",
            (project_id, trigger_type)
        )
        return dict(result[0])

    @staticmethod
    def update_status(job_id: str, status: str, stage: Optional[str] = None,
                      progress: Optional[float] = None,
                      processed: Optional[int] = None,
                      total: Optional[int] = None,
                      error: Optional[str] = None):
        """更新任务状态"""
        updates = ["status = %s"]
        params = [status]

        if stage is not None:
            updates.append("current_stage = %s")
            params.append(stage)
        if progress is not None:
            updates.append("progress = %s")
            params.append(progress)
        if processed is not None:
            updates.append("processed_files = %s")
            params.append(processed)
        if total is not None:
            updates.append("total_files = %s")
            params.append(total)
        if error is not None:
            updates.append("error_message = %s")
            params.append(error)

        if status in ("cloning", "parsing"):
            updates.append("started_at = NOW()")
        elif status in ("completed", "failed"):
            updates.append("completed_at = NOW()")

        params.append(job_id)
        sync_conn.execute(
            f"UPDATE ingestion_jobs SET {', '.join(updates)} WHERE id = %s",
            tuple(params)
        )

    @staticmethod
    def get_pending_jobs() -> List[dict]:
        """获取待处理的任务"""
        result = sync_conn.execute(
            """SELECT * FROM ingestion_jobs
               WHERE status = 'pending'
               ORDER BY created_at ASC
               LIMIT 10"""
        )
        return [dict(r) for r in result] if result else []


# ============================================================
# 检索 Repository
# ============================================================

class RetrievalRepository:
    """检索记录数据访问"""

    @staticmethod
    def log_retrieval(project_id: Optional[str], query_text: str,
                      query_embedding: Optional[List[float]],
                      top_k: int, retrieval_type: str,
                      hybrid_weight: float, latency_ms: int) -> str:
        """记录检索"""
        embedding_str = vector_to_str(query_embedding) if query_embedding else None
        result = sync_conn.execute(
            """INSERT INTO retrieval_logs
               (project_id, query_text, query_embedding, top_k, retrieval_type, hybrid_weight, latency_ms)
               VALUES (%s, %s, %s::vector, %s, %s, %s, %s)
               RETURNING id""",
            (project_id, query_text, embedding_str, top_k, retrieval_type, hybrid_weight, latency_ms)
        )
        return str(result[0]["id"])

    @staticmethod
    def log_results(retrieval_id: str, results: List[Dict[str, Any]]):
        """批量记录检索结果"""
        for r in results:
            sync_conn.execute(
                """INSERT INTO retrieval_results
                   (retrieval_id, chunk_id, rank, vector_score, keyword_score, final_score, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)""",
                (retrieval_id, r.get("chunk_id"), r.get("rank"),
                 r.get("vector_score"), r.get("keyword_score"),
                 r.get("final_score"), json.dumps(r.get("metadata", {})))
            )


# ============================================================
# 管道日志 Repository
# ============================================================

class PipelineLogRepository:
    """管道日志数据访问"""

    @staticmethod
    def log(project_id: Optional[str], job_id: Optional[str],
            step_name: str, status: str,
            input_count: Optional[int] = None,
            output_count: Optional[int] = None,
            duration_ms: Optional[int] = None,
            error_message: Optional[str] = None,
            parameters: Optional[dict] = None):
        """记录管道步骤"""
        sync_conn.execute(
            """INSERT INTO pipeline_logs
               (project_id, job_id, step_name, status, input_count, output_count,
                duration_ms, error_message, parameters)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)""",
            (project_id, job_id, step_name, status,
             input_count, output_count, duration_ms,
             error_message, json.dumps(parameters) if parameters else None)
        )
