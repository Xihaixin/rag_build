"""
RAG 类 — 使用 pgvector 后端的 Adalflow 兼容 RAG 实现

替代原始 deepwiki-open 中使用 FAISSRetriever 的 RAG 类，
底层使用 rag_optimizer 的 PgvectorRetriever 和 HybridRetriever。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import adalflow as adal

from api.config import configs, get_model_config
from rag_optimizer.integration.deepwiki_adapter import (
    PgvectorDatabaseManager,
    PgvectorRetriever,
    _create_compat_document,
)

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================


@dataclass
class DialogTurn:
    """对话轮次"""
    user_query: adal.Parameter
    assistant_response: adal.Parameter


class CustomConversation:
    """自定义 Conversation 实现，修复 Adalflow 的 list assignment index out of range 错误"""

    def __init__(self):
        self.dialog_turns: List[DialogTurn] = []

    def append_dialog_turn(self, dialog_turn: DialogTurn):
        self.dialog_turns.append(dialog_turn)


class Memory(adal.core.component.DataComponent):
    """简单的对话管理，使用 DialogTurn 列表"""

    def __init__(self):
        super().__init__()
        self.conversation = CustomConversation()

    def call(self) -> Dict:
        """返回对话历史字典"""
        try:
            result = {}
            for i, turn in enumerate(self.conversation.dialog_turns):
                result[str(i)] = {
                    "user_query": turn.user_query,
                    "assistant_response": turn.assistant_response,
                }
            return result
        except Exception as e:
            logger.error(f"Error accessing dialog turns: {str(e)}")
            return {}

    def add_dialog_turn(self, user_query: str, assistant_response: str) -> bool:
        """
        添加对话轮次

        Args:
            user_query: 用户查询
            assistant_response: 助手回复

        Returns:
            bool: 是否成功
        """
        try:
            dialog_turn = DialogTurn(
                user_query=adal.Parameter(
                    data=user_query,
                    type=adal.ParameterType.INPUT,
                    name="user_query",
                ),
                assistant_response=adal.Parameter(
                    data=assistant_response,
                    type=adal.ParameterType.OUTPUT,
                    name="assistant_response",
                ),
            )
            self.conversation.append_dialog_turn(dialog_turn)
            return True
        except Exception as e:
            logger.error(f"Error adding dialog turn: {str(e)}")
            return False


@dataclass
class RAGAnswer(adal.DataClass):
    """RAG 回答数据结构"""
    rationale: Optional[str] = field(default=None, metadata={"description": "推理过程"})
    answer: Optional[str] = field(default=None, metadata={"description": "最终回答"})


# ============================================================
# RAG 类
# ============================================================


class RAG(adal.Component):
    """
    使用 pgvector 后端的 RAG 类

    兼容原始 deepwiki-open 的 RAG 接口，但底层使用 PostgreSQL + pgvector。
    支持多种 LLM 提供者和嵌入模型。
    """

    def __init__(self, provider: str = "google", model: Optional[str] = None, use_s3: bool = False):
        """
        Args:
            provider: LLM 提供者 (google, openai, openrouter, ollama, bedrock, dashscope)
            model: 模型名称，None 则使用默认模型
            use_s3: 保留参数，兼容原始接口
        """
        super().__init__()

        self.provider = provider
        self.model_name = model
        self.db_manager: Optional[PgvectorDatabaseManager] = None
        self.retriever: Optional[PgvectorRetriever] = None
        self.memory = Memory()
        self._pgvector_mode = True

        # 初始化生成器
        self._init_generator()

        logger.info(f"RAG initialized: provider={provider}, model={model or 'default'}")

    def _init_generator(self):
        """初始化 LLM 生成器"""
        try:
            model_config = get_model_config(provider=self.provider, model=self.model_name)
            self.model_kwargs = model_config.get("model_kwargs", {})
            logger.info(f"Generator config loaded: {self.model_kwargs}")
        except Exception as e:
            logger.warning(f"Failed to load generator config: {e}")
            self.model_kwargs = {"model": self.model_name or "gemini-2.5-flash"}

    def initialize_db_manager(self):
        """初始化数据库管理器"""
        self.db_manager = PgvectorDatabaseManager()
        logger.info("PgvectorDatabaseManager initialized")

    def _validate_and_filter_embeddings(self, documents: List) -> List:
        """
        验证和过滤嵌入向量，确保一致性

        Args:
            documents: 文档列表

        Returns:
            过滤后的文档列表
        """
        if not documents:
            return []

        # 收集所有有效的向量维度
        valid_dims = set()
        for doc in documents:
            if hasattr(doc, "vector") and doc.vector is not None:
                valid_dims.add(len(doc.vector))

        if not valid_dims:
            logger.warning("No documents with valid embeddings found")
            return documents

        # 使用最常见的维度
        from collections import Counter
        most_common_dim = Counter(valid_dims).most_common(1)[0][0]
        logger.info(f"Most common embedding dimension: {most_common_dim}")

        # 过滤掉维度不一致的文档
        filtered = []
        for doc in documents:
            if hasattr(doc, "vector") and doc.vector is not None:
                if len(doc.vector) == most_common_dim:
                    filtered.append(doc)
                else:
                    logger.debug(f"Filtering document with mismatched dimension: {len(doc.vector)}")
            else:
                # 保留没有向量的文档（可能用于关键词检索）
                filtered.append(doc)

        logger.info(f"Filtered {len(documents) - len(filtered)} documents with mismatched dimensions")
        return filtered

    def prepare_retriever(
        self,
        repo_url_or_path: str,
        type: str = "github",
        access_token: Optional[str] = None,
        excluded_dirs: Optional[List[str]] = None,
        excluded_files: Optional[List[str]] = None,
        included_dirs: Optional[List[str]] = None,
        included_files: Optional[List[str]] = None,
        embedder_type: Optional[str] = None,
        is_ollama_embedder: Optional[bool] = None,
    ) -> bool:
        """
        准备检索器 — 使用 PgvectorRetriever 替代 FAISSRetriever

        Args:
            repo_url_or_path: 仓库 URL 或本地路径
            type: 仓库类型 (github, gitlab, bitbucket, gitee, local)
            access_token: 访问令牌
            excluded_dirs: 排除的目录列表
            excluded_files: 排除的文件列表
            included_dirs: 包含的目录列表
            included_files: 包含的文件列表
            embedder_type: 嵌入器类型
            is_ollama_embedder: 是否使用 Ollama 嵌入器

        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"Preparing retriever for: {repo_url_or_path} (type={type})")

            # 初始化数据库管理器
            if not self.db_manager:
                self.initialize_db_manager()

            # 准备数据库（创建或获取项目）
            project_id = self.db_manager.prepare_database(
                repo_url_or_path=repo_url_or_path,
                repo_type=type,
                access_token=access_token,
            )

            # 创建 PgvectorRetriever
            self.retriever = PgvectorRetriever(
                project_id=project_id,
                retrieval_type="hybrid",
                top_k=10,
            )

            logger.info(f"PgvectorRetriever initialized for project {project_id}")
            return True

        except Exception as e:
            logger.error(f"Error preparing retriever: {e}", exc_info=True)
            return False

    def call(self, query: str, language: str = "en") -> Tuple[List, Dict]:
        """
        执行 RAG 检索

        Args:
            query: 用户查询
            language: 语言代码

        Returns:
            Tuple[List, Dict]: (检索结果列表, 对话历史)
        """
        try:
            logger.debug(f"RAG.call: query='{query[:50]}', language={language}")

            # 使用 pgvector 检索
            if self.retriever and isinstance(self.retriever, PgvectorRetriever):
                results = self.retriever(query, k=10)
            else:
                logger.warning("No retriever available")
                results = []

            # 获取对话历史
            memory_output = self.memory() if hasattr(self, "memory") else {}

            return results, memory_output

        except Exception as e:
            logger.error(f"RAG.call error: {e}", exc_info=True)
            return [], {}
