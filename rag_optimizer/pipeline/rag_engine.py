"""
RAG 引擎 — 核心问答入口

整合检索、缓存、LLM 生成，提供完整的 RAG 问答能力。
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag_optimizer.cache.embedding_cache import embedding_cache
from rag_optimizer.cache.semantic_cache import semantic_cache
from rag_optimizer.cache.repo_lock import repo_lock
from rag_optimizer.config.settings import settings
from rag_optimizer.db.connection import sync_conn
from rag_optimizer.db.repository import RetrievalRepository
from rag_optimizer.pipeline.ingestion import Embedder
from rag_optimizer.retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievalResult,
    RetrievalStats,
)

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class RAGContext:
    """RAG 上下文"""
    query: str
    results: List[RetrievalResult] = field(default_factory=list)
    stats: Optional[RetrievalStats] = None
    answer: str = ""
    latency_ms: int = 0


# ============================================================
# RAG 引擎
# ============================================================

class RAGEngine:
    """
    RAG 引擎

    整合检索 + 缓存 + LLM 生成，提供完整的 RAG 问答能力。
    支持三种检索模式：vector_only, hybrid, rrf
    """

    def __init__(self, project_id: str, model_name: Optional[str] = None):
        self.project_id = project_id
        self.retriever = HybridRetriever(
            project_id=project_id,
            model_name=model_name or settings.embedding.default_model,
        )
        self.embedder = Embedder(
            model_name=model_name or settings.embedding.default_model,
        )
        self.llm_model = settings.llm.default_model
        self.llm_provider = settings.llm.default_provider

    # ----------------------------------------------------------
    # 检索
    # ----------------------------------------------------------

    def retrieve(self, query: str, retrieval_type: str = "hybrid",
                 top_k: Optional[int] = None,
                 file_pattern: Optional[str] = None,
                 use_cache: bool = True) -> Tuple[List[RetrievalResult], RetrievalStats]:
        """
        检索相关文档

        Args:
            query: 查询文本
            retrieval_type: 检索类型
            top_k: 返回结果数
            file_pattern: 文件路径过滤
            use_cache: 是否使用 Embedding 缓存

        Returns:
            (检索结果, 检索统计)
        """
        # 获取查询向量（使用缓存）
        if use_cache:
            query_embedding = embedding_cache.get_or_compute(
                model=self.retriever.model_name,
                content=query,
                compute_fn=self.embedder.embed_one,
            )
        else:
            query_embedding = self.embedder.embed_one(query)

        # 执行检索
        results, stats = self.retriever.search(
            query_text=query,
            query_embedding=query_embedding,
            retrieval_type=retrieval_type,
            top_k=top_k,
            file_pattern=file_pattern,
            log_retrieval=True,
        )

        return results, stats

    # ----------------------------------------------------------
    # LLM 生成
    # ----------------------------------------------------------

    def generate(self, query: str, results: List[RetrievalResult],
                 language: str = "zh",
                 system_prompt: Optional[str] = None) -> str:
        """
        根据检索结果生成回答

        Args:
            query: 用户查询
            results: 检索结果
            language: 回答语言
            system_prompt: 自定义系统提示词

        Returns:
            生成的回答
        """
        # 构建上下文
        context_parts = []
        for r in results:
            context_parts.append(
                f"[文件: {r.file_path} (相关度: {r.final_score:.4f})]\n{r.content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # 默认系统提示词
        if not system_prompt:
            system_prompt = f"""你是一个代码仓库智能助手。请基于以下检索到的文档内容，回答用户的问题。

要求：
1. 只使用提供的文档内容回答，不要编造信息
2. 如果文档内容不足以回答问题，请明确说明
3. 引用具体的文件路径作为信息来源
4. 回答语言：{language}
5. 保持回答简洁、准确、有技术深度"""

        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"## 检索到的文档\n\n{context}\n\n## 用户问题\n\n{query}"},
        ]

        # 调用 LLM
        try:
            api_key = settings.embedding.dashscope_api_key
            if not api_key:
                logger.warning("DASHSCOPE_API_KEY not set. Returning mock answer.")
                return self._mock_generate(query, results)

            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=settings.embedding.dashscope_base_url,
            )

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                top_p=settings.llm.top_p,
            )

            return response.choices[0].message.content or ""

        except ImportError:
            logger.error("openai package not installed.")
            return self._mock_generate(query, results)
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"[生成回答时出错: {e}]"

    def _mock_generate(self, query: str, results: List[RetrievalResult]) -> str:
        """Mock 生成（用于测试）"""
        parts = [f"关于「{query}」的检索结果：\n"]
        for r in results[:3]:
            parts.append(f"- [{r.file_path}] (得分: {r.final_score:.3f})")
            parts.append(f"  {r.content[:200]}...")
        return "\n".join(parts)

    # ----------------------------------------------------------
    # 完整 RAG 问答
    # ----------------------------------------------------------

    def answer(self, query: str, retrieval_type: str = "hybrid",
               top_k: Optional[int] = None,
               language: str = "zh",
               use_semantic_cache: bool = True,
               file_pattern: Optional[str] = None) -> RAGContext:
        """
        完整的 RAG 问答流程

        流程：
        1. 检查语义缓存（可选）
        2. 检索相关文档
        3. LLM 生成回答
        4. 记录问答日志

        Args:
            query: 用户查询
            retrieval_type: 检索类型
            top_k: 返回结果数
            language: 回答语言
            use_semantic_cache: 是否使用语义缓存
            file_pattern: 文件路径过滤

        Returns:
            RAGContext
        """
        start_time = time.time()
        ctx = RAGContext(query=query)

        # 1. 检查语义缓存
        if use_semantic_cache:
            cached_answer = semantic_cache.get_exact(self.project_id, query)
            if cached_answer:
                ctx.answer = cached_answer
                ctx.latency_ms = int((time.time() - start_time) * 1000)
                logger.info(f"Semantic cache HIT: {query[:50]}")
                return ctx

        # 2. 检索
        results, stats = self.retrieve(
            query=query,
            retrieval_type=retrieval_type,
            top_k=top_k,
            file_pattern=file_pattern,
        )
        ctx.results = results
        ctx.stats = stats

        if not results:
            ctx.answer = "未找到相关文档。"
            ctx.latency_ms = int((time.time() - start_time) * 1000)
            return ctx

        # 3. 生成回答
        ctx.answer = self.generate(query, results, language=language)

        # 4. 缓存回答
        if use_semantic_cache and ctx.answer:
            try:
                query_embedding = self.embedder.embed_one(query)
                semantic_cache.set_exact(
                    repo_id=self.project_id,
                    query=query,
                    answer=ctx.answer,
                    query_vector=query_embedding,
                )
            except Exception as e:
                logger.debug(f"Semantic cache set error: {e}")

        ctx.latency_ms = int((time.time() - start_time) * 1000)

        # 5. 记录问答日志
        try:
            retrieval_id = None
            if stats:
                # 获取最近一次检索的 ID
                log_result = sync_conn.execute(
                    """SELECT id FROM retrieval_logs
                       WHERE project_id = %s AND query_text = %s
                       ORDER BY created_at DESC LIMIT 1""",
                    (self.project_id, query)
                )
                if log_result:
                    retrieval_id = str(log_result[0]["id"])

            sync_conn.execute(
                """INSERT INTO qa_logs
                   (retrieval_id, project_id, query_text, response_text,
                    model_name, latency_ms)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (retrieval_id, self.project_id, query,
                 ctx.answer, self.llm_model, ctx.latency_ms)
            )
        except Exception as e:
            logger.debug(f"QA log error: {e}")

        logger.info(
            f"RAG answer: query='{query[:50]}', "
            f"{len(results)} results, {ctx.latency_ms}ms"
        )

        return ctx


# ============================================================
# 便捷函数
# ============================================================

def create_rag_engine(project_id: str) -> RAGEngine:
    """创建 RAG 引擎"""
    return RAGEngine(project_id=project_id)
