"""
rag_engine.py — 核心 RAG 引擎
================================

从 rag_optimizer/pipeline/rag_engine.py 提升至 core/ 层，
作为 core/flows/ 和 api/ 的公共 RAG 入口。

整合检索、缓存、LLM 生成、qa_logs 记录，提供完整的 RAG 问答能力。

设计要点:
  - retrieve() — 同步检索（含 Embedding 缓存 + 检索日志）
  - generate() — 同步 LLM 生成（兼容旧调用方）
  - generate_async() — 异步流式 LLM 生成（多 Provider，供 flows 使用）
  - answer() — 完整 RAG 问答流程（同步，含语义缓存 + qa_logs）
  - log_qa() — 独立 qa_logs 写入方法（供外部调用）
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from rag_optimizer.cache.embedding_cache import embedding_cache
from rag_optimizer.cache.semantic_cache import semantic_cache
from rag_optimizer.config.settings import settings
from rag_optimizer.db.connection import sync_conn
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

    用法:
        engine = RAGEngine(project_id="...")

        # 仅检索
        results, stats = engine.retrieve("your query")

        # 完整 RAG 问答（同步）
        ctx = engine.answer("your query")

        # 异步流式生成（多 Provider）
        async for chunk in engine.generate_async("your query", results, provider="google"):
            ...
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
        # 保存最近一次检索的 retrieval_id，供 log_qa() 使用
        self._last_retrieval_id: Optional[str] = None

    # ----------------------------------------------------------
    # 检索
    # ----------------------------------------------------------

    def retrieve(self, query: str, retrieval_type: str = "hybrid",
                 top_k: Optional[int] = None,
                 file_pattern: Optional[str] = None,
                 use_cache: bool = True) -> Tuple[List[RetrievalResult], RetrievalStats]:
        """
        检索相关文档。

        集成 Embedding 缓存和检索日志记录。

        Args:
            query: 查询文本
            retrieval_type: 检索类型 (vector_only, hybrid, rrf)
            top_k: 返回结果数
            file_pattern: 文件路径过滤
            use_cache: 是否使用 Embedding 缓存

        Returns:
            (检索结果列表, 检索统计)
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

        # 执行检索（自动记录检索日志）
        results, stats = self.retriever.search(
            query_text=query,
            query_embedding=query_embedding,
            retrieval_type=retrieval_type,
            top_k=top_k,
            file_pattern=file_pattern,
            log_retrieval=True,
        )

        # 保存 retrieval_id 供后续 log_qa() 使用
        self._last_retrieval_id = stats.retrieval_id

        return results, stats

    # ----------------------------------------------------------
    # 同步 LLM 生成（兼容旧调用方）
    # ----------------------------------------------------------

    def generate(self, query: str, results: List[RetrievalResult],
                 language: str = "zh",
                 system_prompt: Optional[str] = None) -> str:
        """
        根据检索结果生成回答（同步）。

        使用 settings 中配置的默认 LLM（DashScope）。
        主要用于 deepwiki_adapter.py 和 demo.py 等同步场景。

        Args:
            query: 用户查询
            results: 检索结果
            language: 回答语言
            system_prompt: 自定义系统提示词（None 时使用默认模板）

        Returns:
            生成的回答文本
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
    # 异步流式 LLM 生成（多 Provider，供 flows 使用）
    # ----------------------------------------------------------

    async def generate_async(
        self,
        query: str,
        results: List[RetrievalResult],
        provider: str = "dashscope",
        model: Optional[str] = None,
        language: str = "zh",
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        根据检索结果流式生成回答（异步，多 Provider）。

        使用 core.utils.llm.call_llm_stream() 进行异步流式调用，
        支持 dashscope / google / openai / openrouter / ollama。

        Args:
            query: 用户查询
            results: 检索结果
            provider: LLM 提供者
            model: 模型名称
            language: 回答语言
            system_prompt: 自定义系统提示词

        Yields:
            SSE 格式字符串: data: {"content":"文本块"}\n\n
        """
        from core.utils.llm import call_llm_stream

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

        # 异步流式调用
        async for chunk in call_llm_stream(provider, model, messages):
            yield chunk

    # ----------------------------------------------------------
    # 完整 RAG 问答（同步）
    # ----------------------------------------------------------

    def answer(self, query: str, retrieval_type: str = "hybrid",
               top_k: Optional[int] = None,
               language: str = "zh",
               use_semantic_cache: bool = True,
               file_pattern: Optional[str] = None) -> RAGContext:
        """
        完整的 RAG 问答流程（同步）。

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
        self.log_qa(query=query, answer=ctx.answer, latency_ms=ctx.latency_ms)

        logger.info(
            f"RAG answer: query='{query[:50]}', "
            f"{len(results)} results, {ctx.latency_ms}ms"
        )

        return ctx

    # ----------------------------------------------------------
    # 问答日志记录
    # ----------------------------------------------------------

    def log_qa(self, query: str, answer: str,
               latency_ms: int,
               model_name: Optional[str] = None,
               retrieval_id: Optional[str] = None) -> None:
        """
        记录问答日志到 qa_logs 表。

        可在外部调用，例如 SimpleChatFlow 在 LLM 返回后调用此方法。

        Args:
            query: 用户查询
            answer: LLM 回答
            latency_ms: 耗时（毫秒）
            model_name: 模型名称（默认使用 self.llm_model）
            retrieval_id: 关联的检索日志 ID（可选，默认使用最近一次检索的 ID）
        """
        if not self.project_id:
            logger.debug("跳过 qa_logs 记录（无 project_id）")
            return

        # 如果未传入 retrieval_id，使用最近一次检索的 ID
        if retrieval_id is None:
            retrieval_id = self._last_retrieval_id

        try:
            sync_conn.execute(
                """INSERT INTO qa_logs
                   (retrieval_id, project_id, query_text, response_text,
                    model_name, latency_ms)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (retrieval_id, self.project_id, query,
                 answer, model_name or self.llm_model, latency_ms),
            )
            logger.debug(f"✓ 问答日志已记录到 qa_logs (project_id={self.project_id}, retrieval_id={retrieval_id})")
        except Exception as e:
            logger.debug(f"记录 qa_logs 失败: {e}")


# ============================================================
# 便捷函数
# ============================================================


def create_rag_engine(project_id: str) -> RAGEngine:
    """创建 RAG 引擎"""
    return RAGEngine(project_id=project_id)
