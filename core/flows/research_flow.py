"""
research_flow.py — 深度研究流
=============================

完整复现前端 Ask.tsx 中的深度研究逻辑。

流程:
  1. 使用 RAGEngine 检索相关文档（含 Embedding 缓存 + 检索日志）
  2. 发送初始研究问题（带 [DEEP RESEARCH] 标记）
  3. 自动继续研究（最多 5 轮迭代）
  4. 每轮检测研究是否完成
  5. 提取研究阶段（计划/更新/结论）
  6. 记录问答日志到 qa_logs 表
  7. 返回完整研究结果

依赖:
  - core.flows.base — BaseFlow 公共基类
  - core.models — Message, ResearchStage
  - core.prompts.rag — DEEP_RESEARCH_*_ITERATION_PROMPT, SIMPLE_CHAT_SYSTEM_PROMPT
  - core.rag_engine — RAGEngine（检索 + 日志记录）
"""

import logging
import time
from typing import Any, Dict, List, Optional

from core.flows.base import BaseFlow, call_llm_and_collect
from core.models import Message, ResearchStage
from core.prompts.rag import (
    SIMPLE_CHAT_SYSTEM_PROMPT,
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
)
from core.rag_engine import RAGEngine

logger = logging.getLogger("core.flows.research")


class DeepResearchFlow(BaseFlow):
    """
    深度研究流 — 完整复现前端 Ask.tsx 中的深度研究逻辑。

    流程:
      1. 使用 RAGEngine 检索相关文档
      2. 发送初始研究问题（带 [DEEP RESEARCH] 标记）
      3. 自动继续研究（最多 5 轮迭代）
      4. 每轮检测研究是否完成
      5. 提取研究阶段（计划/更新/结论）
      6. 记录问答日志到 qa_logs 表
      7. 返回完整研究结果

    对应前端 Ask.tsx 中的:
      - handleConfirmAsk() — 发送初始研究请求
      - continueResearch() — 自动继续研究
      - checkIfResearchComplete() — 检测研究是否完成
      - extractResearchStage() — 提取研究阶段

    对应后端 api/simple_chat.py 中的:
      - build_deep_research_prompt() — 构建研究 prompt
      - _handle_deep_research() — 处理深度研究
    """

    # 研究完成标记 — 对应前端 Ask.tsx 中的 checkIfResearchComplete()
    COMPLETION_MARKERS = [
        "## Final Conclusion",
        "## Conclusion",
        "This concludes our research",
        "## 最终结论",
        "## 结论",
        "本研究至此结束",
    ]

    # 最大迭代次数 — 对应前端 Ask.tsx 中的 MAX_RESEARCH_ITERATIONS
    MAX_ITERATIONS = 5

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

        # 研究状态
        self.messages: List[Message] = []
        self.research_stages: List[ResearchStage] = []
        self.current_iteration: int = 0
        self.is_complete: bool = False
        self.final_answer: str = ""

        # RAG 引擎（延迟初始化）
        self.rag_engine: Optional[RAGEngine] = None

        logger.info(f"初始化 DeepResearchFlow:")
        logger.info(f"  仓库: {repo_url}")
        logger.info(f"  提供者: {provider}/{model}")
        logger.info(f"  语言: {self.language_name}")
        logger.info(f"  最大迭代: {self.MAX_ITERATIONS}")

    def _init_rag_engine(self) -> Optional[RAGEngine]:
        """
        初始化 RAG 引擎。

        使用 BaseFlow._find_project_id() 查找 project_id，
        然后创建 RAGEngine 实例。
        """
        if not self.use_database:
            logger.info("跳过 RAG 引擎初始化（use_database=False）")
            return None

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
        """
        if not self.rag_engine:
            return ""

        try:
            # 使用 RAGEngine 执行检索（含 Embedding 缓存 + 检索日志）
            results, stats = self.rag_engine.retrieve(query, top_k=5)
            if not results:
                return ""

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

            return "\n\n".join(context_parts)
        except Exception as e:
            logger.warning(f"RAG 检索失败: {e}")
            return ""

    def _build_research_prompt(
        self, query: str, iteration: int, context: str,
    ) -> List[Dict[str, str]]:
        """
        构建深度研究 prompt。

        对应后端 api/simple_chat.py 中的 build_deep_research_prompt() 函数。
        根据迭代次数选择不同的 prompt 模板：
          - 第 1 轮: DEEP_RESEARCH_FIRST_ITERATION_PROMPT（研究计划）
          - 中间轮: DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT（研究更新）
          - 最终轮: DEEP_RESEARCH_FINAL_ITERATION_PROMPT（最终结论）

        注意：模板中的占位符为 {repo_type}, {repo_url}, {repo_name},
        {language_name}, {research_iteration}（仅中间迭代）。
        上下文和查询作为用户消息内容拼接，而非通过 .format() 传入。
        """
        messages = []

        # 系统指令
        system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT.format(
            language_name=self.language_name,
            repo_url=self.repo_url,
            repo_type=self.repo_type,
            repo_name=self.repo,
        )
        messages.append({"role": "system", "content": system_prompt})

        # 选择 prompt 模板
        if iteration == 1:
            prompt_template = DEEP_RESEARCH_FIRST_ITERATION_PROMPT
        elif iteration >= self.MAX_ITERATIONS:
            prompt_template = DEEP_RESEARCH_FINAL_ITERATION_PROMPT
        else:
            prompt_template = DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT

        # 构建研究 prompt — 只传入模板中实际存在的占位符
        prompt_kwargs: Dict[str, Any] = {
            "repo_type": self.repo_type,
            "repo_url": self.repo_url,
            "repo_name": self.repo,
            "language_name": self.language_name,
        }
        # 中间迭代模板有 {research_iteration} 占位符
        if iteration > 1 and iteration < self.MAX_ITERATIONS:
            prompt_kwargs["research_iteration"] = str(iteration)

        research_prompt = prompt_template.format(**prompt_kwargs)

        # 拼接上下文和用户查询
        if context:
            research_prompt += (
                f"\n\n<START_OF_CONTEXT>\n{context}\n<END_OF_CONTEXT>\n\n"
            )
        research_prompt += f"\n<query>\n{query}\n</query>"

        messages.append({"role": "user", "content": research_prompt})
        return messages

    def _check_if_complete(self, content: str) -> bool:
        """
        检测研究是否完成。

        对应前端 Ask.tsx 中的 checkIfResearchComplete() 函数（lines 176-209）。
        检查内容中是否包含完成标记。
        """
        for marker in self.COMPLETION_MARKERS:
            if marker in content:
                logger.info(f"  检测到完成标记: '{marker}'")
                return True
        return False

    def _extract_stage(
        self, content: str, iteration: int,
    ) -> Optional[ResearchStage]:
        """
        提取研究阶段。

        对应前端 Ask.tsx 中的 extractResearchStage() 函数（lines 212-253）。
        从 LLM 返回内容中提取 plan/update/conclusion 阶段。
        """
        # 检测阶段类型
        stage_type = "update"
        title = f"研究更新 #{iteration}"

        if iteration == 1:
            # 第一轮通常是研究计划
            if "## Research Plan" in content or "## 研究计划" in content:
                stage_type = "plan"
                title = "研究计划"
        elif iteration >= self.MAX_ITERATIONS or self._check_if_complete(content):
            stage_type = "conclusion"
            title = "最终结论"

        # 提取标题行后的内容作为阶段内容
        # 简单实现：使用整个内容
        return ResearchStage(
            title=title,
            content=content,
            iteration=iteration,
            type=stage_type,
        )

    async def research(self, query: str) -> str:
        """
        执行深度研究。

        对应前端 Ask.tsx 中的完整深度研究流程：
          1. handleConfirmAsk() — 发送初始请求（带 [DEEP RESEARCH] 标记）
          2. continueResearch() — 自动继续研究（最多 5 轮）
          3. checkIfResearchComplete() — 每轮检测是否完成

        对应后端 api/simple_chat.py 中的 _handle_deep_research() 流程。
        """
        logger.info("\n" + "=" * 60)
        logger.info("DeepResearchFlow.research()")
        logger.info("=" * 60)
        logger.info(f"研究问题: {query}")
        logger.info(f"最大迭代: {self.MAX_ITERATIONS}")

        start_time = time.time()

        # 初始化 RAG 引擎
        if not self.rag_engine:
            self._init_rag_engine()

        # 构建初始 RAG 上下文
        context = self._build_context(query)

        # 迭代研究循环
        for iteration in range(1, self.MAX_ITERATIONS + 1):
            self.current_iteration = iteration
            logger.info(f"\n{'─' * 50}")
            logger.info(f"迭代 {iteration}/{self.MAX_ITERATIONS}")
            logger.info(f"{'─' * 50}")

            # 构建研究 prompt
            messages = self._build_research_prompt(query, iteration, context)

            # 如果有对话历史，添加到 messages 中
            for msg in self.messages:
                messages.append({"role": msg.role, "content": msg.content})

            # 调用 LLM
            logger.info(f"调用 LLM ({self.provider}/{self.model})...")
            full_response = ""
            try:
                full_response = await call_llm_and_collect(
                    self.provider, self.model, messages
                )
                logger.info(f"✓ 响应 ({len(full_response)} 字符)")
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
                full_response = f"[错误] LLM 调用失败: {e}"

            # 记录消息
            self.messages.append(Message(role="assistant", content=full_response))

            # 提取研究阶段
            stage = self._extract_stage(full_response, iteration)
            if stage:
                self.research_stages.append(stage)
                logger.info(f"  阶段: {stage.type} - {stage.title}")

            # 检测是否完成
            if self._check_if_complete(full_response):
                logger.info(f"✓ 研究在第 {iteration} 轮完成")
                self.is_complete = True
                self.final_answer = full_response
                break

            # 如果不是最后一轮，准备继续研究
            if iteration < self.MAX_ITERATIONS:
                # 对应前端 Ask.tsx 中 continueResearch() 的逻辑：
                # 添加 "[DEEP RESEARCH] Continue the research" 到消息历史
                continue_prompt = "[DEEP RESEARCH] Continue the research"
                self.messages.append(Message(role="user", content=continue_prompt))
                logger.info("  准备继续下一轮研究...")

        if not self.is_complete:
            logger.info(f"达到最大迭代次数 ({self.MAX_ITERATIONS})，研究结束")
            self.final_answer = self.messages[-1].content if self.messages else ""

        latency_ms = int((time.time() - start_time) * 1000)

        # 记录问答日志到 qa_logs 表（通过 RAGEngine.log_qa()）
        if self.rag_engine:
            self.rag_engine.log_qa(
                query=query,
                answer=self.final_answer,
                latency_ms=latency_ms,
                model_name=f"{self.provider}/{self.model}",
            )
        else:
            logger.debug("跳过 qa_logs 记录（无 RAG 引擎）")

        logger.info(f"\n✓ 深度研究完成")
        logger.info(f"总迭代: {self.current_iteration}")
        logger.info(f"阶段数: {len(self.research_stages)}")

        return self.final_answer

    def print_research_summary(self) -> None:
        """打印研究结果摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("深度研究 — 结果摘要")
        logger.info("=" * 60)

        logger.info(f"研究问题: {self.messages[0].content if self.messages else 'N/A'}")
        logger.info(f"总迭代: {self.current_iteration}/{self.MAX_ITERATIONS}")
        logger.info(f"是否完成: {'是' if self.is_complete else '否'}")
        logger.info(f"")

        logger.info("研究阶段:")
        for i, stage in enumerate(self.research_stages, 1):
            icon = {"plan": "📋", "update": "🔄", "conclusion": "✅"}.get(stage.type, "📝")
            logger.info(f"  {icon} [{i}] {stage.title} (迭代 {stage.iteration})")
            preview = stage.content[:150]
            if len(stage.content) > 150:
                preview += "..."
            logger.info(f"     {preview}")

        logger.info(f"")
        logger.info(f"最终回答长度: {len(self.final_answer)} 字符")
        logger.info(f"提供者: {self.provider}/{self.model}")
        logger.info(f"仓库: {self.repo_url}")

    # ── 主入口 ────────────────────────────────────────────────────────────

    async def run(self) -> Dict[str, Any]:
        """
        执行研究流程主入口（需要外部提供 query）。

        注意：DeepResearchFlow 的 run() 方法需要外部提供 query 参数，
        因此 research() 方法是主要的调用入口。run() 作为占位符，
        返回当前研究状态。

        Returns:
            Dict with keys: research_stages, is_complete, final_answer
        """
        return {
            "research_stages": self.research_stages,
            "is_complete": self.is_complete,
            "final_answer": self.final_answer,
        }
