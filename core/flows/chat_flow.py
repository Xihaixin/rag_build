"""
chat_flow.py — 用户 Q&A 简单聊天流
===================================

完整复现前端 Ask.tsx 中的简单聊天逻辑。

流程:
  1. 使用 RAGEngine 检索相关文档（含 Embedding 缓存 + 检索日志）
  2. 组装 prompt（系统指令 + 对话历史 + RAG 上下文 + 用户问题）
  3. 调用 LLM 流式回答
  4. 记录问答日志到 qa_logs 表（通过 RAGEngine.log_qa()）
  5. 返回完整回答

依赖:
  - core.flows.base — BaseFlow 公共基类
  - core.models — Message
  - core.prompts.rag — SIMPLE_CHAT_SYSTEM_PROMPT, RAG_TEMPLATE
  - core.rag_engine — RAGEngine（检索 + 日志记录）
"""

import logging
import time
from typing import Any, Dict, List, Optional

from core.flows.base import BaseFlow, call_llm_and_collect
from core.models import Message
from core.prompts.rag import SIMPLE_CHAT_SYSTEM_PROMPT, render_rag_template
from core.rag_engine import RAGEngine

logger = logging.getLogger("core.flows.chat")


class SimpleChatFlow(BaseFlow):
    """
    用户 Q&A 简单聊天流 — 完整复现前端 Ask.tsx 中的简单聊天逻辑。

    流程:
      1. 使用 RAGEngine 检索相关文档
      2. 组装 prompt（系统指令 + 对话历史 + RAG 上下文 + 用户问题）
      3. 调用 LLM 流式回答
      4. 记录问答日志到 qa_logs 表
      5. 返回完整回答

    对应前端 Ask.tsx 中的:
      - handleConfirmAsk() — 发送聊天请求
      - createChatWebSocket() — WebSocket 通信

    对应后端 api/simple_chat.py 中的:
      - build_simple_chat_prompt() — 构建 prompt
      - _handle_simple_chat() — 处理聊天
      - PgvectorRetriever — RAG 检索
    """

    def __init__(
        self,
        repo_url: str,
        provider: str = "google",
        model: str = "gemini-2.0-flash-exp",
        language: str = "zh",
        use_database: bool = True,
    ):
        # 调用 BaseFlow.__init__ 初始化公共属性
        super().__init__(
            repo_url=repo_url,
            provider=provider,
            model=model,
            language=language,
            use_database=use_database,
        )

        # 对话历史
        self.messages: List[Message] = []

        # RAG 引擎（延迟初始化）
        self.rag_engine: Optional[RAGEngine] = None

        logger.info(f"初始化 SimpleChatFlow:")
        logger.info(f"  仓库: {repo_url}")
        logger.info(f"  提供者: {provider}/{model}")
        logger.info(f"  语言: {self.language_name}")

    def _init_rag_engine(self) -> Optional[RAGEngine]:
        """
        初始化 RAG 引擎。

        使用 BaseFlow._find_project_id() 查找 project_id，
        然后创建 RAGEngine 实例。
        """
        if not self.use_database:
            logger.info("跳过 RAG 引擎初始化（use_database=False）")
            return None

        # 如果还没有 project_id，尝试查找
        if not self.project_id:
            self._find_project_id()

        if not self.project_id:
            logger.warning(f"未找到项目: {self.repo_url}，跳过 RAG 引擎")
            return None

        try:
            self.rag_engine = RAGEngine(project_id=self.project_id)
            logger.info(f"✓ RAG 引擎已初始化 (project_id={self.project_id})")
            return self.rag_engine
        except Exception as e:
            logger.warning(f"初始化 RAG 引擎失败: {e}")
            return None

    def _build_context(self, query: str) -> str:
        """
        构建 RAG 上下文文本。

        使用 RAGEngine.retrieve() 进行检索（含 Embedding 缓存 + 检索日志），
        然后将结果格式化为上下文文本。

        对应后端 api/simple_chat.py 中的 build_context_from_results() 函数。
        """
        if not self.rag_engine:
            logger.info("无 RAG 引擎，跳过上下文构建")
            return ""

        try:
            # 使用 RAGEngine 执行检索（含 Embedding 缓存 + 检索日志）
            results, stats = self.rag_engine.retrieve(query, top_k=5)

            if not results:
                logger.info("检索结果为空")
                return ""

            # 构建上下文文本
            context_parts = []
            for i, result in enumerate(results, 1):
                content = getattr(result, "content", "") or ""
                file_path = getattr(result, "file_path", "") or ""
                score = getattr(result, "final_score", 0.0) or getattr(result, "vector_score", 0.0)

                context_parts.append(
                    f"[{i}] 文件: {file_path}\n"
                    f"    相关度: {score:.4f}\n"
                    f"    内容: {content[:500]}..."
                )

            context = "\n\n".join(context_parts)
            logger.info(f"✓ RAG 上下文构建完成 ({len(context)} 字符, {len(results)} 个结果)")
            return context

        except Exception as e:
            logger.warning(f"RAG 检索失败: {e}")
            return ""

    def _build_prompt(
        self, query: str, context: str,
        history: Optional[List[Message]] = None,
    ) -> List[Dict[str, str]]:
        """
        构建聊天 prompt。

        对应后端 api/simple_chat.py 中的 build_simple_chat_prompt() 函数：
          1. 系统指令 (SIMPLE_CHAT_SYSTEM_PROMPT)
          2. 对话历史
          3. RAG 上下文
          4. 用户问题

        使用 render_rag_template() 通过 Jinja2 渲染 RAG_TEMPLATE，
        保留模板的 Jinja2 语法以兼容未来 adalflow 重构。
        """
        messages = []

        # 1. 系统指令
        system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT.format(
            language_name=self.language_name,
            repo_url=self.repo_url,
            repo_type=self.repo_type,
            repo_name=self.repo,
        )
        messages.append({"role": "system", "content": system_prompt})

        # 2. 对话历史
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})

        # 3. RAG 上下文 + 用户问题
        if context:
            # 使用 Jinja2 渲染 RAG_TEMPLATE
            user_prompt = render_rag_template(
                system_prompt=system_prompt,
                output_format_str="",
                input_str=query,
            )
            # 在 <START_OF_CONTEXT> 之前插入上下文文本
            # 因为当前上下文是纯文本字符串，不是 adalflow Document 对象列表
            context_section = (
                f"\n<START_OF_CONTEXT>\n{context}\n<END_OF_CONTEXT>\n"
            )
            # 在 <START_OF_USER_PROMPT> 之前插入
            user_prompt = user_prompt.replace(
                "<START_OF_USER_PROMPT>",
                f"{context_section}<START_OF_USER_PROMPT>",
            )
            messages.append({"role": "user", "content": user_prompt})
        else:
            messages.append({"role": "user", "content": query})

        return messages

    async def chat(
        self, query: str,
        history: Optional[List[Message]] = None,
    ) -> str:
        """
        执行一次聊天问答。

        对应前端 Ask.tsx 中的 handleConfirmAsk() 流程：
          1. 准备请求体（repo_url, messages, provider, model, language）
          2. 通过 WebSocket 发送
          3. 接收流式响应

        对应后端 api/simple_chat.py 中的 _handle_simple_chat() 流程：
          1. 初始化 RAG 引擎
          2. 检索相关文档
          3. 构建 prompt
          4. 调用 LLM 流式生成
          5. 记录问答日志到 qa_logs 表
        """
        logger.info("\n" + "=" * 60)
        logger.info("SimpleChatFlow.chat()")
        logger.info("=" * 60)
        logger.info(f"问题: {query}")

        start_time = time.time()

        # 步骤 1: 初始化 RAG 引擎
        if not self.rag_engine:
            self._init_rag_engine()

        # 步骤 2: 构建 RAG 上下文
        logger.info("步骤 1: RAG 检索...")
        context = self._build_context(query)

        # 步骤 3: 构建 prompt
        logger.info("步骤 2: 构建 prompt...")
        messages = self._build_prompt(query, context, history)
        logger.info(f"  messages 数: {len(messages)}")

        # 步骤 4: 调用 LLM
        logger.info(f"步骤 3: 调用 LLM ({self.provider}/{self.model})...")
        full_response = ""

        try:
            full_response = await call_llm_and_collect(self.provider, self.model, messages)
            logger.info(f"✓ LLM 返回完成 ({len(full_response)} 字符)")
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            full_response = f"[错误] LLM 调用失败: {e}"

        latency_ms = int((time.time() - start_time) * 1000)

        # 记录对话历史
        self.messages.append(Message(role="user", content=query))
        self.messages.append(Message(role="assistant", content=full_response))

        # 步骤 5: 记录问答日志到 qa_logs 表（通过 RAGEngine.log_qa()）
        if self.rag_engine:
            self.rag_engine.log_qa(
                query=query,
                answer=full_response,
                latency_ms=latency_ms,
                model_name=f"{self.provider}/{self.model}",
            )
        else:
            logger.debug("跳过 qa_logs 记录（无 RAG 引擎）")

        return full_response

    def print_conversation(self) -> None:
        """打印对话历史"""
        logger.info("\n" + "=" * 60)
        logger.info("对话历史")
        logger.info("=" * 60)
        for i, msg in enumerate(self.messages):
            role_label = "👤 用户" if msg.role == "user" else "🤖 助手"
            logger.info(f"\n[{i + 1}] {role_label}:")
            # 只打印前 200 字符
            preview = msg.content[:200]
            if len(msg.content) > 200:
                preview += "..."
            logger.info(preview)

    # ── 主入口 ────────────────────────────────────────────────────────────

    async def run(self) -> Dict[str, Any]:
        """
        执行聊天流程主入口（需要外部提供 query）。

        注意：SimpleChatFlow 的 run() 方法需要外部提供 query 参数，
        因此 chat() 方法是主要的调用入口。run() 作为占位符，
        返回当前对话历史。

        Returns:
            Dict with keys: messages, retriever_initialized
        """
        return {
            "messages": self.messages,
            "retriever_initialized": self.rag_engine is not None,
        }
