import os
import sys
import json
import weakref
import logging
import subprocess
import adalflow as adal

from pathlib import Path
from LLMClients.dashscope_client import DashscopeClient
from dataclasses import dataclass, field
from adalflow.core.db import LocalDB
from adalflow.core.types import Document, List
from typing import Any, List, Tuple, Dict
from urllib.parse import urlparse, urlunparse, quote
from adalflow.components.data_process import TextSplitter, ToEmbeddings
from default_prompts.prompts import RAG_TEMPLATE, RAG_SYSTEM_PROMPT as system_prompt
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from utils.read_all_documents import read_all_documents, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES

logger = logging.getLogger(__name__)

# 从环境变量中获取阿里云百炼的 API
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", default="")

# 将 API 密钥加载到环境中
if DASHSCOPE_API_KEY:
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

CONFIG_DIR = os.environ.get('DEEPWIKI_CONFIG_DIR', "./config")
CLIENT_CLASSES = {"DashScopeClient": DashscopeClient}


#------------ 加载并管理各种配置 ------------
def get_json_config(filename):
    """
    根据文件名加载对应配置

    参数：
        filename: str 配置文件（json）的名称

    返回值：
        config: dict 一个字典配置对象

    异常值：
        捕捉到异常，返回 {}

    """
    try:
        if CONFIG_DIR:
            config_path = Path(CONFIG_DIR) / filename
        else:
            print("配置目录不存在")
            return
        print(f"Loading configuration from {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)   # .load 方法加载的是一个文件对象
            return config
        
    except Exception as e:
        print(f"Error loading configuration file {filename}: {str(e)}")
        return {}


def load_configs():
    """
    为 LLLM 加载各种配置，如供应商(provider)，向量解析模型(embedder)，仓库克隆的文件过滤规则(repo_file_filters)，以及语言映射规则(lang)
    在这个函数中，为了适配原项目多模型厂商的选择规则，做了很多抽象。实际上，如果已经确定只使用阿里云百炼平台提供的模型，可以直接硬编码。

    返回值：
        config: dict

    示例：
        config = {
            // 虽然以复数形式命名 providers 但目前里面只有一个供应商： dashscope
            // 每个 provider 都必须提供的属性 -> 
            //  : -->  1. default_model: str
            //  : -->  2. supportsCustomModel: bool
            //  : -->  3. model_client: ClientClass
            //  : -->  4. models: dict
            //              - `name`: str
            //              - `temperature`: float
            //              - `top_p`: float

            "providers":{
                ......  
            },
            "embedder":{
                ......
            },
            "retriever":{
                ......
            },
            "text_splitter":{
                ......
            }

            // embedder.json 里面也只提供了一个向量模型--来源于阿里云百炼, 该文件提供三个属性: embedder, retriever, text_splitter.
                embedder_config = 
                {
                    "embedder": {
                        "client_class": "DashscopeClient",
                        "batch_size": 500,
                        "model_kwargs": {
                        "model": "text-embedding-v4",
                        "dimensions": 256,
                        "encoding_format": "float"
                        "model_client": ClientClass,
                        }
                    },
                    "retriever": {
                        "top_k": 20
                    },
                    "text_splitter": {
                        "split_by": "word",
                        "chunk_size": 350,
                        "chunk_overlap": 100
                    }
                }
            // 克隆仓库的文件读取规则
            "file_filters":{
                "excluded_dirs": List[str],
                "excluded_files": List[str]
            },
            "repository":{
                "max_size_mb": int
            },
            // 获取语言映射配置，与前端用户的选择进行配合
            "supported_languages": dict,
            "default": str

    """
    configs = {}
    
    # 获取大语言模型提供商配置
    generator_config = {
        "default_provider": "dashscope",
        "providers": {
            "dashscope": {
                "default_model": "qwen-plus",
                "supportsCustomModel": True,
                "model_client": CLIENT_CLASSES["DashScopeClient"],  
                "models": {
                    "qwen-plus": {
                        "temperature": 0.7,
                        "top_p": 0.8
                    },
                    "qwen-turbo": {
                        "temperature": 0.7,
                        "top_p": 0.8
                    },
                }
            }
        }
    }
    configs["providers"] = generator_config.get("providers", {})

    # 获取向量解析模型配置
    embedder_config = get_json_config("embedder.json")
    class_name = embedder_config["embedder"]["client_class"]
    embedder_config["embedder"]["model_client"] = CLIENT_CLASSES[class_name]
    for key in ["embedder", "embedder_ollama", "retriever", "text_splitter"]:
        if key in embedder_config:
            configs[key] = embedder_config[key]

    # 获取克隆仓库的文件读取规则
    repo_config = get_json_config("repo.json")
    if repo_config:
        for key in ["file_filters", "repository"]:
            if key in repo_config:
                configs[key] = repo_config[key]
    
    # 获取语言映射规则配置
    lang_config = get_json_config("lang.json")
    if lang_config:
        configs["lang_config"] = lang_config
    
    return configs

configs = load_configs()

def get_model_config(provider="dashscope", model=None):
    """
    从全局的配置对象 configs 中获取 LLM 模型

    参数：
        provider: str = "dashscope"
        model: str = None

    注意：
        对于 providers 配置，我们暂时没有写入到某一个特定的 json 文件中，而是采用硬编码的方式，具体可以查看 load_configs
        函数中 generator_config 变量的配置。

    返回值：
        result: dict 一个包含 LLM 模型配置的字典
        result = {
            "model_client": ClientClass,
            "model_kwargs": {
                "model": str,
                "temperature": float,
                "top_p": float
            }
        }
    
    """
    if "providers" not in configs:
        raise ValueError("Provider configuration not loaded")
    
    # 这一步直接获取到 dashscope 所在的键值对
    provider_config = configs["providers"].get(provider)
    if not provider_config:
        raise ValueError(f"Configuration for provider '{provider}' not found")

    model_client = provider_config.get("model_client")
    if not model_client:
        raise ValueError(f"Model client not specified for provider '{provider}'")

    if not model:
        model = provider_config.get("default_model")
        if not model:
            raise ValueError(f"No default model specified for provider '{provider}'")

    # 获取模型参数
    model_params = {}
    if model in provider_config.get("models", {}):
        model_params = provider_config["models"][model]
    else:
        default_model = provider_config.get("default_model")
        model_params = provider_config["models"][default_model] 

    result = {
        "model_client": model_client,
    }               

    result["model_kwargs"] = {"model": model, **model_params}  # 这里对 model_params 进行了解包

    return result


def get_embedder() -> adal.Embedder:
    """
    从全局配置变量 configs 中获取向量模型

    注意：
        在该函数内部构造了返回对象所需要的两个参数：model_client, model_kwargs
        model_client: ClinetClass
        model_kwargs: {
            "model": str,
            "dimensions": int,
            "encoding_format": str
        }
    
    返回值：
        embedder: adal.Embedder

    """
    embedder_config = configs["embedder"]

    model_client_class = embedder_config["model_client"]
    if "initialize_kwargs" in embedder_config:
        model_client = model_client_class(**embedder_config["initialize_kwargs"])
    else:
        model_client = model_client_class()

    embedder = adal.Embedder(
        model_client=model_client,
        model_kwargs=embedder_config["model_kwargs"],
    )
    return embedder


def get_adalflow_default_root_path():
    """从配置环境中获取文件缓存路径"""
    root = None
    if sys.platform == "win32":
        root = os.getenv("APPDATA", r"D:\\tmp\\adalflow") 
    else:
        root = os.path.join(os.path.expanduser("~"), ".adalflow")
    return root

def download_repo(repo_url: str, local_path: str, type: str = "gitee", access_token: str = None) -> str:
    """
    根据用户给出的 git 仓库地址，将其下载到本地

    参数：
        repo_url: str 仓库 web 地址
        local_path: str 本地存储的文件路径
        type: str 远程 git 仓库类型，默认为国内 gitee 
        access_token: str = None 私人仓库访问权限
    
    返回值：
        result: 程序调用 git 进行克隆的结果


    """
    try:
        # 检查 git 工具是否安装
        logger.info(f"Preparing to clone repository to {local_path}")
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 检查仓库是否存在
        if os.path.exists(local_path) and os.listdir(local_path):
            # Directory exists and is not empty
            logger.warning(f"Repository already exists at {local_path}. Using existing repository.")
            return f"Using existing repository at {local_path}"

        # 确保本地路径存在
        os.makedirs(local_path, exist_ok=True)

        # 如果提供 access_token 参数，那么构造 git 仓库的克隆 url
        clone_url = repo_url
        if access_token:
            parsed = urlparse(repo_url)
            # Determine the repository type and format the URL accordingly
            if type == "github":
                # Format: https://{token}@{domain}/owner/repo.git
                # Works for both github.com and enterprise GitHub domains
                clone_url = urlunparse((parsed.scheme, f"{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif type == "gitlab":
                # Format: https://oauth2:{token}@gitlab.com/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"oauth2:{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif type == "bitbucket":
                # Format: https://x-token-auth:{token}@bitbucket.org/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"x-token-auth:{access_token}@{parsed.netloc}", parsed.path, '', '', ''))
            elif type == "gitee":
                clone_url = urlunparse((parsed.scheme, f"{access_token}@{parsed.netloc}",parsed.path, '', '', ''))
            logger.info("Using access token for authentication") # TODO：这里gitee的设置有问题

        # 克隆 git 仓库
        logger.info(f"Cloning repository from {repo_url} to {local_path}")
        # We use repo_url in the log to avoid exposing the token in logs
        result = subprocess.run(
            ["git", "clone", "--depth=1", "--single-branch", clone_url, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info(f"Repository:{repo_url} cloned successfully")
        return result.stdout.decode("utf-8")

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8')
        # Sanitize error message to remove any tokens
        if access_token and access_token in error_msg:
            error_msg = error_msg.replace(access_token, "***TOKEN***")
        raise ValueError(f"Error during cloning: {error_msg}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {str(e)}")

def prepare_data_pipeline() -> adal.Sequential:
    """
    准备数据管道的基础配置

    注意：
        这个函数构造了两个组件，分别是“文本分块”和“向量嵌入”，最后将它们组装为一个序列组件，它就是我们的“数据转换”组件


    返回值：
        data_transformer: adal.Sequential 一个数据转换器组件，这部分代码的逻辑可以查看 https://adalflow.sylph.ai/tutorials/db.html 
        你可以将它打印出来，结果可能如下：

        Sequential(
            (0): TextSplitter(split_by=word, split_length=350, split_overlap=100)
            (1): ToEmbeddings(
                batch_size=500
                (embedder): Embedder(
                        model_kwargs={'model': 'text-embedding-v4', 'dimensions': 256, 'encoding_format': 'float'},
                        (model_client): DashscopeClient()
                    )
                (batch_embedder): BatchEmbedder(
                        (embedder): Embedder(
                            model_kwargs={'model': 'text-embedding-v4', 'dimensions': 256, 'encoding_format': 'float'},
                            (model_client): DashscopeClient()
                        )
                    )
                )
            )


    示例：
        文本分割器：
            chunk_size: 每一个文本分割块的大小
            chunk_overlap: 两个分块之间重合部分的大小 
            slit_by: 规定了分割规则，即分割时最小的单位。支持"word", "sentence", "page", "passage", 和 "token"
        
        >>> "Hello, world!" -> ["Hello, " ,"world!"] # 如果你设置 split_by = "word"
        >>> text_splitter = TextSplitter(split_by="word",chunk_size=5,chunk_overlap=1)
        >>> # 示例文档
        >>> doc = Document(
                text="Example text. More example text. Even more text to illustrate.",
                id="doc1"
            )
        >>> # 进行文档分割
        >>> splitted_docs = text_splitter.call(documents=[doc])
        >>> for doc in splitted_docs:
            print(doc)
    
    """
    splitter = TextSplitter(**configs["text_splitter"])
    embedder_config = configs.get("embedder", {})
    embedder = get_embedder()

    # 使用批处理，规定其大小
    batch_size = embedder_config.get("batch_size", 500)
    embedder_transformer = ToEmbeddings(
        embedder=embedder, batch_size=batch_size
    )

    data_transformer = adal.Sequential(
        splitter, embedder_transformer
    )  # sequential will chain together splitter and embedder
    return data_transformer


def transform_documents_and_save_to_db(
    documents: List[Document], db_path: str, is_ollama_embedder: bool = None
) -> LocalDB:
    """
    本地数据库对象，将用户传入的 Document 对象列表进行以特定的方式进行转换并加载到数据库中，最后对数据库状态进行持久化，存储为 pkl 文件

    参数：
        documents: List[Document] 它是一个列表，内部元素是 Document 
        db_path: str 这个是数据库保存的文件路径
        is_ollama_embedder: bool 暂时没有配置本地 ollama 

    返回值：
        db: LocalDB() 对象实例，
    """

    data_transformer = prepare_data_pipeline(is_ollama_embedder)

    # 将文档资源保存到本地数据库
    db = LocalDB()
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    return db

class DatabaseManager:
    """
    对本地数据库实例对象进行管理，包括：创建，加载，转换，持久化

    注意：
        数据管理器实际上对克隆的 git 仓库进行管理，主要是对它的本地数据存储路径数据，项目文档解析数据，数据库状态持久化
        准备检索器 prepare_retriever -> prepare_database -> prepare_db_index -> List[TransformedDocument] 生成转换后的文档对象列表
    类属性：
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths: dict = {
            "save_repo_dir":str, // 保存克隆在本地的 git 项目仓库
            "save_db_file": .pkl //这个文件最终是用来保存数据库(db)的状态
        }

        self.repo_paths["save_db_file"] 对应的 pkl 文件在 self._create_repo 中被创建，此时为空。在为数据实例建立索引时被修改

    返回值：
        数据管理器实例

    """

    def __init__(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = {}

    def prepare_database(
            self, 
            repo_url_or_path: str, 
            type: str = "gitee", 
            access_token: str = None, 
            is_ollama_embedder: bool = None,
            excluded_dirs: List[str] = None, excluded_files: List[str] = None,
            included_dirs: List[str] = None, included_files: List[str] = None
        ) -> List[Document]:
        """准备数据库"""

        self.reset_database()
        self._create_repo(repo_url_or_path, type, access_token)
        return self.prepare_db_index(
                        is_ollama_embedder=is_ollama_embedder, 
                        excluded_dirs=excluded_dirs, 
                        excluded_files=excluded_files,
                        included_dirs=included_dirs, 
                        included_files=included_files
                    )

    def reset_database(self):
        """重置数据管理器的各属性"""
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        """从web 仓库的 url 路径中提取仓库名"""

        url_parts = repo_url_or_path.rstrip('/').split('/')

        if repo_type in ["gitee", "gitlab", "bitbucket"] and len(url_parts) >= 5:
            # GitHub URL format: https://gitee.com/owner/repo
            # GitLab URL format: https://gitlab.com/owner/repo or https://gitlab.com/group/subgroup/repo
            # Bitbucket URL format: https://bitbucket.org/owner/repo
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            repo_name = f"{owner}_{repo}"
        else:
            repo_name = url_parts[-1].replace(".git", "")
        return repo_name

    def _create_repo(self, repo_url_or_path: str, repo_type: str = "gitee", access_token: str = None) -> None:
        """
        克隆 git 远程仓库或直接加载本地 git 项目，并在本地创建统一格式的项目保存路径

        参数：
            repo_url_or_path: str
            repo_type: str = "gitee"
            access_token: str = None

        返回值：None
            该函数内部改变了管理器实例的属性值：
            repo_paths: dict = {
                "save_repo_dir": "/repos/<repo_name>"
                "save_db_file": "/database/<repo_name>.pkl",
            }
            repo_url_or_path: str -> web_url | path_uri
        
        """

        logger.info(f"Preparing repo storage for {repo_url_or_path}...")

        try:
            root_path = get_adalflow_default_root_path()

            os.makedirs(root_path, exist_ok=True)
            # url
            if repo_url_or_path.startswith("https://") or repo_url_or_path.startswith("http://"):
                # Extract the repository name from the URL
                repo_name = self._extract_repo_name_from_url(repo_url_or_path, repo_type)
                logger.info(f"Extracted repo name: {repo_name}")

                save_repo_dir = os.path.join(root_path, "repos", repo_name)

                # Check if the repository directory already exists and is not empty
                if not (os.path.exists(save_repo_dir) and os.listdir(save_repo_dir)):
                    # Only download if the repository doesn't exist or is empty
                    download_repo(repo_url_or_path, save_repo_dir, repo_type, access_token)
                else:
                    logger.info(f"Repository already exists at {save_repo_dir}. Using existing repository.")
            else:  # local path
                repo_name = os.path.basename(repo_url_or_path)
                save_repo_dir = repo_url_or_path

            save_db_file = os.path.join(root_path, "databases", f"{repo_name}.pkl")
            os.makedirs(save_repo_dir, exist_ok=True)
            os.makedirs(os.path.dirname(save_db_file), exist_ok=True)

            self.repo_paths = {
                "save_repo_dir": save_repo_dir,
                "save_db_file": save_db_file,
            }
            self.repo_url_or_path = repo_url_or_path
            logger.info(f"Repo paths: {self.repo_paths}")

        except Exception as e:
            logger.error(f"Failed to create repository structure: {e}")
            raise

    def prepare_db_index(
            self, 
            is_ollama_embedder: bool = None, 
            excluded_dirs: List[str] = None, 
            excluded_files: List[str] = None,
            included_dirs: List[str] = None, 
            included_files: List[str] = None
        ) -> List[Document]:
        """
        建立数据库，并对文档数据进行转换，同时建立起数据对应的索引

        参数：
            is_ollama_embedder: 是否为本地模型
            excluded_dirs: 排除在外的目录
            excluded_files: 排除在外的文件
            include_dirs: 包含在内的目录
            include_files: 包含在内的文件

        返回值：
            transformed_docs: List[Document] -> 经过转换后的文档数据 | 文档分块 | 向量嵌入
        
        """

        # 检查数据库
        if self.repo_paths and os.path.exists(self.repo_paths["save_db_file"]):
            logger.info("Loading existing database...")
            try:
                self.db = LocalDB.load_state(self.repo_paths["save_db_file"])
                documents = self.db.get_transformed_data(key="split_and_embed")
                if documents:
                    logger.info(f"Loaded {len(documents)} documents from existing database")
                    return documents
            except Exception as e:
                logger.error(f"Error loading existing database: {e}")
                

        # 准备数据库
        logger.info("Creating new database...")
        documents = read_all_documents(
            self.repo_paths["save_repo_dir"],
            is_ollama_embedder=is_ollama_embedder,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )
        self.db = transform_documents_and_save_to_db(
            documents, self.repo_paths["save_db_file"], is_ollama_embedder=is_ollama_embedder
        )
        logger.info(f"Total documents: {len(documents)}")
        transformed_docs = self.db.get_transformed_data(key="split_and_embed")
        logger.info(f"Total transformed documents: {len(transformed_docs)}")
        return transformed_docs

    def prepare_retriever(self, repo_url_or_path: str, type: str = "gitee", access_token: str = None):
        
        return self.prepare_database(repo_url_or_path, type, access_token)

## -------- 自定义数据模型 --------

@dataclass
class UserQuery:
    query_str: str

@dataclass
class AssistantResponse:
    response_str: str

@dataclass
class DialogTurn:
    id: str
    user_query: UserQuery
    assistant_response: AssistantResponse

class CustomConversation:
    """自定义实现对话功能，以修复列表赋值索引超出范围的错误"""

    def __init__(self):
        self.dialog_turns = []

    def append_dialog_turn(self, dialog_turn):
        """安全地将对话回合添加到对话中"""
        if not hasattr(self, 'dialog_turns'):
            self.dialog_turns = []
        self.dialog_turns.append(dialog_turn)

class Memory(adal.core.component.DataComponent):
    """
    通过对话轮次列表实现简单的对话管理，继承自 adal.core.component.DataComponent

    注意：
        在 Memory 的父类 DataComponent 中定义了一个魔术方法 __call__(), 而该方法内部返回了一个对自身 self.call() 方法的调用
        因此，当我们使用 Memory 的实例，并像函数那样调用实例时，该实例方法 call 就会被触发，其底层是从父类继承过来的 __call__ 方法

        >>> mery = Memory()
        >>> mery() 

        ```
        # 父类 DataComponent 内部的定义
        def __call__(self, *args, **kwargs):
            return self.call(*args, **kwargs)

        ```

        该实例在 RAG 实例初始化属性时进行组装

    
    """

    def __init__(self):
        super().__init__()
        # Use our custom implementation instead of the original Conversation class
        self.current_conversation = CustomConversation()

    def call(self) -> Dict:
        """Return the conversation history as a dictionary."""
        all_dialog_turns = {}
        try:
            # Check if dialog_turns exists and is a list
            if hasattr(self.current_conversation, 'dialog_turns'):
                if self.current_conversation.dialog_turns:
                    logger.info(f"Memory content: {len(self.current_conversation.dialog_turns)} turns")
                    for i, turn in enumerate(self.current_conversation.dialog_turns):
                        if hasattr(turn, 'id') and turn.id is not None:
                            all_dialog_turns[turn.id] = turn
                            logger.info(f"Added turn {i+1} with ID {turn.id} to memory")
                        else:
                            logger.warning(f"Skipping invalid turn object in memory: {turn}")
                else:
                    logger.info("Dialog turns list exists but is empty")
            else:
                logger.info("No dialog_turns attribute in current_conversation")
                # Try to initialize it
                self.current_conversation.dialog_turns = []
        except Exception as e:
            logger.error(f"Error accessing dialog turns: {str(e)}")
            # Try to recover
            try:
                self.current_conversation = CustomConversation()
                logger.info("Recovered by creating new conversation")
            except Exception as e2:
                logger.error(f"Failed to recover: {str(e2)}")

        logger.info(f"Returning {len(all_dialog_turns)} dialog turns from memory")
        return all_dialog_turns

@dataclass
class RAGAnswer(adal.DataClass):
    """
    定义了一个名为 RAGAnswer 的数据类,用于强制 LLM 按照指定的结构返回答案

    属性值：
        - rationale: 存储推理过程(思维链)
        - answer: 存储最终答案(要求用 Markdown 格式)
    
    metadata 里的 desc 会被自动注入到 LLM 的 prompt 中,告诉模型每个字段应该填什么内容。
    
    注意：
        那个 metadata={"desc": "..."} 不只是注释。当你把 RAGAnswer 传给 Generator 时:
        1. AdalFlow 会自动生成 JSON Schema
        2. 把 desc 注入到 system prompt
        3. 要求 LLM 按照 schema 返回 JSON
        4. 自动验证和解析 JSON 成 Python 对象

        RAGAnswer 继承自 adal.DataClass, 因此也获得了父类两个属性：   
        1. __input_fields__: List[str] = []
        2. __output_fields__: List[str] = []

        被 @dataclass 识别的字段（有类型注解的属性），会以结构化的方式存储在类的 __dataclass_fields__ 属性中（这是一个字典）；而字段的具体值则存储在类的实例对象中（比如 self.rationale、self.answer）

    示例：
        >>> print(RAGAnswer.to_json_signature())
        >>> Output: 
                  {
                        "rationale": "Chain of thoughts for the answer."
                        "answer": "Answer to the user query, formatted in markdown for beautiful rendering with react-markdown, DO NOT include ``` triple backticks fences at beginning or end of your answer."
                    
                  }
        >>> my_instance = RAGAnswer(
                        rationale="first,user ask me who create the Appale company. I remember it's Jobs cofound it."
                        answer="Jobs"
                    )
        >>> print(my_instance.to_json_example())
        >>> Output:
                  {
                        "rationale": "first,user ask me who create the Appale company. I remember it's Jobs cofound it."
                        "answer": "Jobs"
                  }

    """
    rationale: str = field(default="", metadata={"desc": "Chain of thoughts for the answer."})
    answer: str = field(default="", metadata={"desc": "Answer to the user query, formatted in markdown for beautiful rendering with react-markdown, DO NOT include ``` triple backticks fences at beginning or end of your answer."})

    __output_fields__ = ["rationale", "answer"]

class RAG(adal.Component):
    """ 对开源仓库进行向量化，如果需要加载一个新的仓库，请先调用 prepare_retriever """

    def __init__(self, provider="dashscope", model=None, use_s2: bool = False):
        super().__init__()
        self.provider = provider
        self.model = model
        self.embedder = get_embedder()

        self.memory = Memory()
        self.is_ollama_embedder = None
        self_weakref = weakref.ref(self)    # 弱引用，避免循环引用导致内存泄漏
        def single_string_embedder(query):
            # Accepts either a string or a list, always returns embedding for a single string
            if isinstance(query, list):
                if len(query) != 1:
                    raise ValueError("Ollama embedder only supports a single string")
                query = query[0]
            instance = self_weakref()
            assert instance is not None, "RAG instance is no longer available, but the query embedder was called."
            return instance.embedder(input=query)
        
        self.query_embedder = single_string_embedder if self.is_ollama_embedder else self.embedder
        self.initialize_db_manager()

        # 设置输出解析器：https://adalflow.sylph.ai/apis/components/components.output_parsers.dataclass_parser.html#components.output_parsers.dataclass_parser.DataClassParser
        data_parser = adal.DataClassParser(data_class=RAGAnswer, return_data_class=True)
        format_instructions = data_parser.get_output_format_str() + """
IMPORTANT FORMATTING RULES:
1. DO NOT include your thinking or reasoning process in the output
2. Provide only the final, polished answer
3. DO NOT include ```markdown fences at the beginning or end of your answer
4. DO NOT wrap your response in any kind of fences
5. Start your response directly with the content
6. The content will already be rendered as markdown
7. Do not use backslashes before special characters like [ ] { } in your answer
8. When listing tags or similar items, write them as plain text without escape characters
9. For pipe characters (|) in text, write them directly without escaping them"""

        generator_config = get_model_config(self.provider, self.model)

        # 配置生成器
        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": format_instructions,
                "conversation_history": self.memory(),
                "system_prompt": system_prompt,
                "contexts":None,
            },
            model_client=generator_config["model_client"](),
            model_kwargs=generator_config["model_kwargs"],
            output_processors=data_parser,
        )


    def initialize_db_manager(self):
        """使用本地存储初始化数据库管理器"""
        self.db_manager = DatabaseManager()
        self.transformed_docs = []

    def _validate_and_filter_embeddings(self, documents: List) -> List:
        """
        过滤掉尺寸不一致的文档

        参数：
            self: RAG 类的实例
            documents: 文档列表

        注意：
            该函数确保所有保留的文档具有一致的嵌入维度，这对于构建向量索引（如 FAISS）是必要的，因为 FAISS 要求所有向量具有相同的维度。
        
        返回值：
            valid_document: 经过过滤的文档列表，其中所有文档的嵌入向量具有相同的维度，如果输入为空或无有效嵌入，则返回空列表
        """
        if not documents:
            logger.warning("No documents provided for embedding validation")
            return []
        
        valid_documents = []
        embedding_sizes = {}

        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                logger.warning(f"Document {i} has no embedding vector, skipping")
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    logger.warning(f"Document {i} has invalid embedding vector type: {type(doc.vector)}, skipping")
                    continue

                if embedding_size == 0:
                    logger.warning(f"Document {i} has empty embedding vector, skipping")
                    continue    
                
                embedding_sizes[embedding_size] = embedding_sizes.get(embedding_size, 0) + 1                            
            
            except Exception as e:
                logger.warning(f"Error checking embedding size for document {i}: {str(e)}, skipping")
                continue

        if not embedding_sizes:
            logger.error("No valid embeddings found in any documents")
            return []
        
        # 从字典 embedding_sizes 的所有键中，找出对应值最大的那个键
        target_size = max(embedding_sizes.keys(), key=lambda k: embedding_sizes[k])
        logger.info(f"Target embedding size: {target_size} (found in {embedding_sizes[target_size]} documents)")

        for size, count in embedding_sizes.items():
            if size != target_size:
                logger.warning(f"Found {count} documents with incorrect embedding size {size}, will be filtered out")
        
        # 第二轮：根据目标向量大小对文档进行筛选
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    continue

                if embedding_size == target_size:
                    valid_documents.append(doc)
                else:
                    # Log which document is being filtered out
                    file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                    logger.warning(f"Filtering out document '{file_path}' due to embedding size mismatch: {embedding_size} != {target_size}")

            except Exception as e:
                file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                logger.warning(f"Error validating embedding for document '{file_path}': {str(e)}, skipping")
                continue

        logger.info(f"Embedding validation complete: {len(valid_documents)}/{len(documents)} documents have valid embeddings")

        if len(valid_documents) == 0:
            logger.error("No documents with valid embeddings remain after filtering")
        elif len(valid_documents) < len(documents):
            filtered_count = len(documents) - len(valid_documents)
            logger.warning(f"Filtered out {filtered_count} documents due to embedding issues")

        return valid_documents
    
    # 在 DataBase 类中也有一个 prepare_retriever 方法
    def prepare_retriever(
            self, 
            repo_url_or_path: str, 
            type: str = "gitee", 
            access_token: str = None,
            excluded_dirs: List[str] = None, 
            excluded_files: List[str] = None,
            included_dirs: List[str] = None, 
            included_files: List[str] = None
        ):
        """
        初始化 RAG 检索器，为 LLM 后续检索增强做准备
        
        参数：
            repo_url_or_path: 项目的 web 仓库地址或本地文件路径
            type: 类型 | 默认为 “gitee”
            access_token: 用户提供的私密仓库的授权
            excluded_dirs: 排除在外的目录
            excluded_files: 排除在外的文件
            included_dirs: 包含在内的目录
            included_files: 包含在内的文件

        注意：
            检索时用的retrieve_embedder必须和生成文档 Embedding 的模型一致，否则用户问题的向量和文档向量不在同一空间，检索结果会完全错误。
            FAISS 的作用：在 RAG 中，用户提问后会先将问题转为 Embedding，再用 FAISS 从知识库中快速找到 Top-N 相似的文档，这些文档作为上下文传给 LLM，LLM 结合上下文生成答案（而非纯靠自身训练数据）
        
        返回值：None | 该方法主要作用是修改类的实例属性：初始化完成的 self.retriver -- FAISS 检索器实例


        """
        self.initialize_db_manager()
        self.repo_url_or_path = repo_url_or_path
        self.transformed_docs = self.db_manager.prepare_database(
            repo_url_or_path,
            type,
            access_token,
            is_ollama_embedder=self.is_ollama_embedder,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
        )
        logger.info(f"Loaded {len(self.transformed_docs)} documents for retrieval")

        self.transformed_docs = self._validate_and_filter_embeddings(self.transformed_docs)

        if not self.transformed_docs:
            raise ValueError("No valid documents with embeddings found. Cannot create retriever.")
        logger.info(f"Using {len(self.transformed_docs)} documents with valid embeddings for retrieval")

        try:
            # 使用合适的嵌入器进行检索
            retrieve_embedder = self.query_embedder if self.is_ollama_embedder else self.embedder
            self.retriever = FAISSRetriever(
                **configs["retriever"],
                embedder=retrieve_embedder,
                documents=self.transformed_docs,
                document_map_func=lambda doc: doc.vector,
            )        
            logger.info("FAISS retriever created successfully")
        except Exception as e:
            logger.error(f"Error creating FAISS retriever: {str(e)}")
            # Try to provide more specific error information
            if "All embeddings should be of the same size" in str(e):
                logger.error("Embedding size validation failed. This suggests there are still inconsistent embedding sizes.")
                # Log embedding sizes for debugging
                sizes = []
                for i, doc in enumerate(self.transformed_docs[:10]):  # Check first 10 docs
                    if hasattr(doc, 'vector') and doc.vector is not None:
                        try:
                            if isinstance(doc.vector, list):
                                size = len(doc.vector)
                            elif hasattr(doc.vector, 'shape'):
                                size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                            elif hasattr(doc.vector, '__len__'):
                                size = len(doc.vector)
                            else:
                                size = "unknown"
                            sizes.append(f"doc_{i}: {size}")
                        except:
                            sizes.append(f"doc_{i}: error")
                logger.error(f"Sample embedding sizes: {', '.join(sizes)}")
            raise

    def call(self, query: str, language: str = "zh") -> Tuple[List]:
        """接收用户的查询语句(query)，调用之前初始化后的 FAISS 检索器获取相关文档，处理后返回检索结果"""
        try:
            retrieved_documents = self.retriever(query)

            # Fill in the documents
            retrieved_documents[0].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retrieved_documents[0].doc_indices
            ]

            return retrieved_documents

        except Exception as e:
            logger.error(f"Error in RAG call: {str(e)}")

            # Create error response
            error_response = RAGAnswer(
                rationale="Error occurred while processing the query.",
                answer=f"I apologize, but I encountered an error while processing your question. Please try again or rephrase your question."
            )
            return error_response, []                                    

if __name__ == "__main__":
    repo_url = "https://gitee.com/xihaishen/polls"
    # generate_wiki_locally(repo_url)
    rag_pipeline = RAG(provider="dashscope", model="quen-turo")