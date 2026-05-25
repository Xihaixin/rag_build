"""
数据摄取管道 — 从 Git 仓库到 PostgreSQL 的完整处理流程

流程：
1. 读取文档（复用 read_all_documents）
2. 文本分块（TextSplitter）
3. 向量嵌入（DashScope Embedding API）
4. 写入 PostgreSQL（chunk_embeddings_dim256）
5. 记录管道日志
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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

logger = logging.getLogger(__name__)


# ============================================================
# 文本分块器
# ============================================================

class TextSplitter:
    """文本分块器（兼容 adalflow TextSplitter 接口）"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None,
                 split_by: str = "word"):
        self.chunk_size = chunk_size or settings.chunk.default_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk.default_chunk_overlap
        self.split_by = split_by

    def split_text(self, text: str, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        将文本切分为块

        Args:
            text: 输入文本
            file_type: 文件类型（用于代码场景的特殊分块）

        Returns:
            分块列表，每块包含 content, chunk_index, token_count, start_offset, end_offset
        """
        # 代码文件使用行级分块
        if file_type in ("py", "js", "ts", "java", "cpp", "go", "rs", "c", "h"):
            return self._split_code(text)
        return self._split_text_generic(text)

    def _split_code(self, text: str) -> List[Dict[str, Any]]:
        """代码文件分块：按行分块，保留完整行"""
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_size = 0
        start_offset = 0
        chunk_index = 0

        for i, line in enumerate(lines):
            line_len = len(line) + 1  # +1 for newline
            if current_size + line_len > self.chunk_size and current_chunk:
                # 保存当前块
                chunk_text = "\n".join(current_chunk)
                chunks.append({
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_text) // 4,  # 粗略估计
                    "start_offset": start_offset,
                    "end_offset": start_offset + len(chunk_text),
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "split_by": "code_line",
                })
                chunk_index += 1

                # 重叠部分
                overlap_lines = []
                overlap_size = 0
                for cl in reversed(current_chunk):
                    if overlap_size + len(cl) + 1 > self.chunk_overlap:
                        break
                    overlap_lines.insert(0, cl)
                    overlap_size += len(cl) + 1

                current_chunk = overlap_lines + [line]
                current_size = overlap_size + line_len
                start_offset = max(0, start_offset + len("\n".join(current_chunk[:-1])) + 1 - overlap_size)
            else:
                current_chunk.append(line)
                current_size += line_len

        # 最后一个块
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                "content": chunk_text,
                "chunk_index": chunk_index,
                "token_count": len(chunk_text) // 4,
                "start_offset": start_offset,
                "end_offset": start_offset + len(chunk_text),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "split_by": "code_line",
            })

        return chunks

    def _split_text_generic(self, text: str) -> List[Dict[str, Any]]:
        """通用文本分块"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        char_offset = 0
        chunk_index = 0

        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_size + word_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_text) // 4,
                    "start_offset": char_offset - len(chunk_text),
                    "end_offset": char_offset,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "split_by": "word",
                })
                chunk_index += 1

                # 重叠
                overlap_words = []
                overlap_size = 0
                for cw in reversed(current_chunk):
                    if overlap_size + len(cw) + 1 > self.chunk_overlap:
                        break
                    overlap_words.insert(0, cw)
                    overlap_size += len(cw) + 1

                current_chunk = overlap_words + [word]
                current_size = overlap_size + word_len
            else:
                current_chunk.append(word)
                current_size += word_len

            char_offset += word_len

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "content": chunk_text,
                "chunk_index": chunk_index,
                "token_count": len(chunk_text) // 4,
                "start_offset": char_offset - len(chunk_text),
                "end_offset": char_offset,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "split_by": "word",
            })

        return chunks


# ============================================================
# 嵌入器
# ============================================================

class Embedder:
    """向量嵌入器（封装 DashScope Embedding API）"""

    # DashScope text-embedding-v4 的 batch size 上限为 10
    DASHSCOPE_MAX_BATCH_SIZE = 10

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None,
                 dimensions: Optional[int] = None):
        self.model_name = model_name or settings.embedding.default_model
        self.api_key = api_key or settings.embedding.dashscope_api_key
        self.dimensions = dimensions or settings.embedding.default_dimensions
        # DashScope text-embedding-v4 的 batch size 上限为 10
        self.batch_size = min(
            settings.embedding.batch_size,
            self.DASHSCOPE_MAX_BATCH_SIZE,
        )

        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY not set. Embedding will use mock vectors.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文本

        Args:
            texts: 文本列表

        Returns:
            向量列表（统一为 self.dimensions 维）
        """
        if not self.api_key:
            # Mock 模式：返回随机向量（用于测试）
            import random
            logger.warning(f"Using mock embeddings for {len(texts)} texts")
            return [[random.uniform(-1, 1) for _ in range(self.dimensions)] for _ in texts]

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=settings.embedding.dashscope_base_url,
            )

            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                response = client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    dimensions=self.dimensions,  # 明确指定维度，确保与数据库表一致
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Embedded batch {i // self.batch_size + 1}: {len(batch)} texts")

            return all_embeddings

        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            raise

    def embed_one(self, text: str) -> List[float]:
        """嵌入单条文本"""
        return self.embed([text])[0]


# ============================================================
# 摄取管道
# ============================================================

class IngestionPipeline:
    """
    数据摄取管道

    完整流程：读取文档 → 分块 → 嵌入 → 写入数据库
    """

    def __init__(self, project_id: str, job_id: Optional[str] = None):
        self.project_id = project_id
        self.job_id = job_id
        self.splitter = TextSplitter()
        self.embedder = Embedder()

        # 获取模型 ID
        result = sync_conn.execute(
            "SELECT id FROM embedding_models WHERE name = %s",
            (settings.embedding.default_model,)
        )
        self.model_id = str(result[0]["id"]) if result else None
        if not self.model_id:
            raise ValueError(f"Embedding model '{settings.embedding.default_model}' not found")

    def process_document(self, file_path: str, content: str,
                         file_type: Optional[str] = None,
                         is_code: bool = True,
                         force_reprocess: bool = False) -> Tuple[int, int]:
        """
        处理单个文档：分块 + 嵌入 + 写入

        Args:
            file_path: 文件路径
            content: 文件内容
            file_type: 文件类型
            is_code: 是否为代码文件
            force_reprocess: 是否强制重新处理（即使内容未变更也重新分块和嵌入）

        Returns:
            (chunk_count, embed_count)
        """
        # 1. 写入原始文档
        doc_id, changed = DocumentRepository.upsert(
            project_id=self.project_id,
            file_path=file_path,
            content=content,
            file_type=file_type,
            is_code=is_code,
            token_count=len(content) // 4,
        )

        # 2. 判断是否需要重新分块
        need_rechunk = changed or force_reprocess

        if not need_rechunk:
            # 内容未变更：检查是否已有向量嵌入
            existing_chunks = ChunkRepository.get_by_document(doc_id)
            if existing_chunks:
                # 检查第一个 chunk 是否有对应的向量
                first_chunk_id = existing_chunks[0]["id"]
                has_embedding = sync_conn.execute(
                    """SELECT 1 FROM chunk_embeddings_dim256
                       WHERE chunk_id = %s LIMIT 1""",
                    (first_chunk_id,)
                )
                if has_embedding:
                    return 0, 0  # 内容未变更且已有向量，跳过

                # 分块已存在但缺少向量 → 复用现有分块，只重新生成向量
                logger.info(f"分块已存在但缺少向量，仅重新生成嵌入: {file_path}")
                db_chunks = existing_chunks
                chunk_id_map = {c["chunk_index"]: c["id"] for c in db_chunks}
                texts = [c["content"] for c in db_chunks]
                return self._embed_and_store(
                    file_path, texts, chunk_id_map,
                    len(db_chunks)
                )

            # 内容未变更但连分块都没有（异常情况），需要重新分块
            logger.info(f"内容未变更但缺少分块，重新处理: {file_path}")
            need_rechunk = True

        # 3. 文本分块（内容变更 或 强制重新处理 或 缺少分块）
        chunks = self.splitter.split_text(content, file_type=file_type)
        ChunkRepository.batch_insert(doc_id, chunks)
        logger.debug(f"Split {file_path}: {len(chunks)} chunks")

        # 4. 获取 chunk IDs
        db_chunks = ChunkRepository.get_by_document(doc_id)
        chunk_id_map = {c["chunk_index"]: c["id"] for c in db_chunks}

        # 5. 批量嵌入并存储
        texts = [c["content"] for c in chunks]
        return self._embed_and_store(
            file_path, texts, chunk_id_map, len(chunks)
        )

    def _embed_and_store(self, file_path: str, texts: List[str],
                         chunk_id_map: Dict[int, str],
                         chunk_count: int) -> Tuple[int, int]:
        """
        批量嵌入文本并存储向量到数据库。

        Args:
            file_path: 文件路径（仅用于日志）
            texts: 待嵌入的文本列表
            chunk_id_map: chunk_index → chunk_id 的映射
            chunk_count: 分块总数（用于返回值）

        Returns:
            (chunk_count, embed_count)
        """
        try:
            embeddings = self.embedder.embed(texts)
        except Exception as e:
            logger.error(f"Embedding failed for {file_path}: {e}")
            return chunk_count, 0

        embed_count = 0
        for chunk_index, embedding in zip(chunk_id_map.keys(), embeddings):
            chunk_db_id = chunk_id_map.get(chunk_index)
            if chunk_db_id:
                EmbeddingRepository.insert(
                    chunk_id=chunk_db_id,
                    project_id=self.project_id,
                    model_id=self.model_id,
                    embedding=embedding,
                    content=texts[chunk_index],
                    file_path=file_path,
                    chunk_index=chunk_index,
                )
                embed_count += 1

        return chunk_count, embed_count

    def process_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        批量处理文档

        Args:
            documents: 文档列表，每项包含 file_path, content, file_type, is_code

        Returns:
            统计信息
        """
        total_chunks = 0
        total_embeds = 0
        processed = 0
        skipped = 0
        errors = 0

        for i, doc in enumerate(documents):
            try:
                chunks, embeds = self.process_document(
                    file_path=doc["file_path"],
                    content=doc["content"],
                    file_type=doc.get("file_type"),
                    is_code=doc.get("is_code", True),
                )
                if chunks > 0:
                    processed += 1
                    total_chunks += chunks
                    total_embeds += embeds
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Error processing {doc.get('file_path', 'unknown')}: {e}")
                errors += 1

            # 更新任务进度
            if self.job_id and (i + 1) % 10 == 0:
                IngestionJobRepository.update_status(
                    self.job_id, "chunking",
                    stage="processing",
                    progress=(i + 1) / len(documents) if documents else 0,
                    processed=processed,
                    total=len(documents),
                )

        return {
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "total_chunks": total_chunks,
            "total_embeddings": total_embeds,
        }


# ============================================================
# 便捷函数
# ============================================================

def run_ingestion(project_id: str, documents: List[Dict[str, Any]],
                  job_id: Optional[str] = None) -> Dict[str, int]:
    """
    运行数据摄取管道

    Args:
        project_id: 项目 ID
        documents: 文档列表
        job_id: 摄取任务 ID（可选）

    Returns:
        处理统计
    """
    start_time = time.time()

    # 记录管道开始
    PipelineLogRepository.log(
        project_id=project_id, job_id=job_id,
        step_name="ingestion", status="started",
        input_count=len(documents),
        parameters={"total_documents": len(documents)},
    )

    # 执行摄取
    pipeline = IngestionPipeline(project_id=project_id, job_id=job_id)
    stats = pipeline.process_documents(documents)

    # 记录管道完成
    elapsed = int((time.time() - start_time) * 1000)
    PipelineLogRepository.log(
        project_id=project_id, job_id=job_id,
        step_name="ingestion", status="completed",
        input_count=len(documents),
        output_count=stats["total_chunks"],
        duration_ms=elapsed,
        parameters=stats,
    )

    logger.info(
        f"Ingestion completed: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {stats['errors']} errors, "
        f"{stats['total_chunks']} chunks, {stats['total_embeddings']} embeddings "
        f"in {elapsed / 1000:.1f}s"
    )

    return stats
