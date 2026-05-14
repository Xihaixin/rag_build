import os
import git
import logging
import dashscope
from http import HTTPStatus
from typing import List, Generator
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 配置日志
logger = logging.getLogger("MiniDeepWiki")
logger.setLevel(level=logging.DEBUG)

log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

file_handler = logging.FileHandler(
    "MiniDeepWiki.log",
    encoding="utf-8"
)

file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

load_dotenv()

# 定义常量
MAX_BATCH_SIZE = 10  # 阿里云百炼嵌入API最大批量大小
VECTOR_STORE_ROOT = "./vector_stores"  # 向量数据库持久化根目录
LLM_MODEL = "qwen-turbo"  # 优先使用稳定且有免费额度的模型
EMBEDDING_MODEL = "text-embedding-v4"

# 提前创建向量存储根目录（避免持久化时目录不存在）
os.makedirs(VECTOR_STORE_ROOT, exist_ok=True)

def _format_path_for_log(p: str) -> str:
    """将路径规范化为可读的日志格式，使用正斜杠分隔，保持相对路径形式。"""
    try:
        return Path(p).as_posix()
    except Exception:
        return str(p).replace('\\', '/')


def _unique_preserve_order(seq: List[str]) -> List[str]:
    """有序去重并保留原始顺序，去掉首尾空白。"""
    seen = set()
    out = []
    for s in seq:
        if s is None:
            continue
        normalized = s.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out

class MinimalDeepWiki:
    """最小化的 DeepWiki"""
    
    def __init__(self, api_key: str = None):
        """
        初始化阿里云百炼客户端
        Args:
            api_key: 阿里云百炼API Key，默认为环境变量 DASHSCOPE_API_KEY
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key 参数")
        
        # 设置DashScope API Key
        dashscope.api_key = self.api_key
        
        # 模型配置
        self.embedding_model = EMBEDDING_MODEL
        self.llm_model = LLM_MODEL
        
        # 实例变量
        self.vector_store = None
        self.repo_path = None
        self.vector_store_path = None  # 对应仓库的向量存储路径
    
    # ---------------------- LLM接口联通测试（移除不兼容参数+优化Prompt） ----------------------
    def test_llm_connection(self, test_question: str = None) -> str:
        """
        测试LLM接口是否联通，不依赖向量检索，单纯验证对话模型工作正常
        """
        # 默认测试问题（简化，避免复杂计算，降低模型处理难度）
        default_question = "请用一句话介绍你自己，并用中文回答。"
        test_q = test_question or default_question
        
        # 构建简洁Prompt（移除result_format依赖，直接在Prompt中约束格式）
        prompt = f"""你是一个专业的AI助手，严格按照以下要求回答：
1.  使用中文回答
2.  内容简洁明了，不超过50字
3.  格式为普通文本（无需复杂Markdown）

用户问题：{test_q}"""
        
        logger.info("开始测试LLM接口联通性...")
        try:
            response = dashscope.Generation.call(
                model=self.llm_model,
                prompt=prompt,
                temperature=0.1,
                top_p=0.8,
                max_tokens=500,  # 降低max_tokens，避免超出限制
                enable_search=False  # 明确关闭搜索，减少模型处理负担
            )
            
            # 强化响应解析
            if response.status_code == HTTPStatus.OK:
                if hasattr(response, 'output') and response.output is not None:
                    if hasattr(response.output, 'text') and response.output.text is not None:
                        answer_text = response.output.text.strip()
                        return answer_text if answer_text else "LLM接口联通成功，但返回空内容"
                    else:
                        logger.warning("响应output中缺少有效text字段（text为None或不存在）")
                        return "LLM接口联通成功，但响应格式异常：未获取到有效文本内容"
                else:
                    logger.warning("响应中缺少有效output字段（output为None）")
                    return "LLM接口联通成功，但响应格式异常：缺少output字段"
            else:
                logger.error(f"LLM接口测试失败: {response.code} - {response.message}")
                return f"LLM接口联通失败：{response.message}"
        except Exception as e:
            logger.error(f"LLM接口调用异常: {e}")
            return f"LLM接口联通异常：{str(e)}"
    
    # ---------------------- 克隆仓库到本地 ----------------------
    def clone_repo(self, repo_url: str, local_path: str = "./repos"):
        """克隆代码仓库，并生成对应的向量存储路径"""
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        target_path = os.path.join(local_path, repo_name)
        self.vector_store_path = os.path.join(VECTOR_STORE_ROOT, repo_name)
        if os.path.exists(target_path):
            logger.info(f"仓库已存在: {_format_path_for_log(target_path)}")
            self.repo_path = target_path
            return target_path
            
        logger.info(f"正在克隆仓库: {repo_url}")
        git.Repo.clone_from(repo_url, target_path, depth=1)
        self.repo_path = target_path
        return target_path
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """调用DashScope文本嵌入API（分批处理）"""
        all_embeddings = []
        
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch_texts = texts[i:i+MAX_BATCH_SIZE]
            logger.info(f"正在处理第 {i//MAX_BATCH_SIZE + 1} 批嵌入请求，批次大小: {len(batch_texts)}")
            
            try:
                resp = dashscope.TextEmbedding.call(
                    model=self.embedding_model,
                    input=batch_texts
                )
                if resp.status_code == HTTPStatus.OK:
                    batch_embeddings = [output['embedding'] for output in resp.output['embeddings']]
                    all_embeddings.extend(batch_embeddings)
                else:
                    logger.error(f"嵌入API错误（第{i//MAX_BATCH_SIZE + 1}批）: {resp.code} - {resp.message}")
                    raise RuntimeError(f"嵌入失败（第{i//MAX_BATCH_SIZE + 1}批）: {resp.message}")
            except Exception as e:
                logger.error(f"嵌入调用异常（第{i//MAX_BATCH_SIZE + 1}批）: {e}")
                raise
        
        return all_embeddings
    
    def index_documents(self, repo_path: str):
        """索引文档并创建向量存储（支持持久化复用）"""
        from langchain.embeddings.base import Embeddings
        class DashScopeEmbeddings(Embeddings):
            def __init__(self, parent):
                self.parent = parent
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.parent._get_embeddings(texts)
            
            def embed_query(self, text: str) -> List[float]:
                return self.parent._get_embeddings([text])[0]
        
        embeddings = DashScopeEmbeddings(self)
        
        # 加载已有向量库
        if self.vector_store_path and os.path.exists(self.vector_store_path):
            logger.info(f"检测到已有向量存储: {_format_path_for_log(self.vector_store_path)}，直接加载")
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=self.vector_store_path,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("向量存储加载完成")
                return
            except Exception as e:
                logger.error(f"加载本地向量存储失败，将重新生成: {e}")
        
        # 生成新向量库
        logger.info("正在读取并分块文档...")
        loader_kwargs={'encoding':'utf-8', 'autodetect_encoding': False}
        loader = DirectoryLoader(
            repo_path,
            glob=["**/*.py", "**/*.md"],
            loader_cls=TextLoader,
            loader_kwargs=loader_kwargs,
            recursive=True,
            exclude=["**/node_modules/**", "**/.git/**", "**/dist/**", "**/build/**"]
        )
        documents = loader.load()
        
        logger.info(f"加载了 {len(documents)} 个文档")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"创建了 {len(chunks)} 个文档块")
        if len(chunks) > 0:
            logger.info("正在生成嵌入向量...")
            try:
                self.vector_store = FAISS.from_documents(chunks, embeddings)
                logger.info("向量存储生成完成")
                if self.vector_store_path:
                    self.vector_store.save_local(self.vector_store_path)
                    logger.info(f"向量存储已持久化到: {_format_path_for_log(self.vector_store_path)}")
            except Exception as e:
                logger.error(f"创建向量存储失败: {e}")
                raise
        else:
            logger.error("没有文档块可以索引")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """检索相关上下文"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化，请先调用 index_documents()")
        
        docs = self.vector_store.similarity_search(query, k=k)
        context_contents = [doc.page_content for doc in docs]
        logger.info(f"检索到 {len(context_contents)} 条相关上下文，第一条上下文长度: {len(context_contents[0]) if context_contents else 0}")
        return context_contents
    
    # ---------------------- 同步回答（移除不兼容参数+简化Prompt） ----------------------
    def answer(self, question: str, language: str = "zh") -> str:
        """RAG 问答（修复LLM无法生成有效文本问题）"""
        contexts = self.retrieve_context(question)

        # 有序去重并只取前三条（避免重复句子进入Prompt）
        unique_contexts = _unique_preserve_order(contexts)[:3]
        context_text = "\n\n---\n\n".join(unique_contexts)
        prompt = f"""你是一个代码仓库助手，基于以下代码库内容回答用户问题，要求：
1.  用{language}语言回答，简洁明了
2.  优先使用代码库中的信息，无相关信息时说明"未查询到相关内容"
3.  格式清晰，无需复杂Markdown

代码库上下文：
{context_text}

用户问题：{question}"""
        
        logger.info(f"构建的prompt总长度: {len(prompt)}")  # 查看优化后的长度
        logger.info(f"正在生成回答...")
        
        try:
            response = dashscope.Generation.call(
                model=self.llm_model,
                prompt=prompt,
                temperature=0.1,
                top_p=0.8,
                max_tokens=1000,  # 合理调整长度
                enable_search=False,
                # 移除 result_format="markdown" （核心修复）
            )
            
            if response.status_code == HTTPStatus.OK:
                if hasattr(response, 'output') and response.output is not None:
                    if hasattr(response.output, 'text') and response.output.text is not None:
                        answer_text = response.output.text.strip()
                        if answer_text:
                            logger.info(f"模型回答: {answer_text}")
                            return answer_text
                        else:
                            logger.info("模型返回空内容（answer_text为空）")
                            return "抱歉，未能生成有效回答（模型返回空内容）。"
                    else:
                        return "抱歉，未能生成有效回答（响应text字段为None）。"
                else:
                    return "抱歉，未能生成有效回答（响应output字段为None）。"
            else:
                logger.error(f"LLM API错误: {response.code} - {response.message}")
                return f"抱歉，生成回答时出错: {response.message}"
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return f"抱歉，生成回答时出现异常: {str(e)}"
    
    # ---------------------- 流式回答（移除不兼容参数+简化Prompt） ----------------------
    def chat_stream(self, question: str, language: str = "zh") -> Generator[str, None, None]:
        """流式回答（修复LLM无法生成有效文本问题）"""
        contexts = self.retrieve_context(question)
        unique_contexts = _unique_preserve_order(contexts)[:3]
        context_text = "\n\n---\n\n".join(unique_contexts)  # 简化上下文，降低Prompt长度
        
        prompt = f"""你是一个代码仓库助手，基于以下代码库内容回答用户问题，要求：
1.  用{language}语言回答，简洁明了
2.  优先使用代码库中的信息，无相关信息时说明"未查询到相关内容"
3.  格式清晰，无需复杂Markdown

代码库上下文：
{context_text}

用户问题：{question}"""
        
        try:
            response = dashscope.Generation.call(
                model=self.llm_model,
                prompt=prompt,
                temperature=0.1,
                top_p=0.8,
                max_tokens=1000,
                enable_search=False,
                stream=True,
                # 移除 result_format="markdown" （核心修复）
            )
            
            has_valid_content = False
            prev_text = ""
            for chunk in response:
                if chunk.status_code == HTTPStatus.OK:
                    if hasattr(chunk, 'output') and chunk.output is not None:
                        if hasattr(chunk.output, 'text') and chunk.output.text is not None:
                            chunk_text = chunk.output.text.strip()
                            if not chunk_text:
                                continue
                            # 如果流式SDK返回累积文本，做增量差分，避免重复输出
                            if prev_text and chunk_text.startswith(prev_text):
                                new_part = chunk_text[len(prev_text):]
                            else:
                                new_part = chunk_text
                            if new_part:
                                has_valid_content = True
                                logger.debug(f"流式模型输出增量: {new_part}")
                                yield new_part
                            prev_text = chunk_text
                else:
                    logger.error(f"流式响应错误: {chunk.code} - {chunk.message}")
                    yield f"\n\n流式回答出错: {chunk.message}"

            if not has_valid_content:
                yield "\n\n抱歉，未能生成有效流式回答（无有效内容返回）。"
        except Exception as e:
            logger.error(f"流式回答失败: {e}")
            yield f"\n\n抱歉，流式回答出现异常: {str(e)}"


def main():
    """主函数：演示所有功能"""
    # 1. 初始化
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    
    wiki = MinimalDeepWiki(api_key=api_key)
    # 2. 测试LLM接口联通性
    logger.info("=" * 60)
    logger.info("开始测试LLM接口联通性")
    logger.info("=" * 60)
    test_result = wiki.test_llm_connection()
    logger.info(f"接口测试结果:\n{test_result}")

    # 3. 克隆仓库+索引文档
    repo_url = "https://gitee.com/xihaishen/polls"
    repo_path = wiki.clone_repo(repo_url)
    wiki.index_documents(repo_path)

    # 4. RAG问答
    logger.info("=" * 60)
    logger.info("开始RAG问答")
    logger.info("=" * 60)
    question = "这个项目的主要功能是什么？"
    answer = wiki.answer(question, language="zh")
    logger.info(f"问题: {question}")
    logger.info(f"回答:\n{answer}")

    # 5. 流式输出
    logger.info("=" * 60)
    logger.info("开始流式回答示例")
    logger.info("=" * 60)
    question2 = "该项目的入口文件在哪里？"
    logger.info(f"问题: {question2}")
    full_answer_parts = []
    for chunk in wiki.chat_stream(question2, language="zh"):
        # 仅聚合流式增量，不在此处逐条记录到日志（仅记录最终汇总）
        full_answer_parts.append(chunk)
    full_answer = "".join(full_answer_parts)
    logger.info(f"流式回答汇总:\n{full_answer}")


if __name__ == "__main__":
    main()