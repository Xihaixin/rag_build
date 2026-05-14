import pickle
import logging
from typing import List

from adalflow.core import LocalDB
from adalflow.components.retriever.faiss_retriever import FAISSRetriever

# 配置日志输出
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SafeUnpickler(pickle.Unpickler):
    """安全的 Pickle 加载器，缺失模块时返回占位对象而非抛出异常"""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except:
            # 缺失模块时返回一个占位对象
            class Placeholder:
                def __repr__(self):
                    return f"<Missing {module}.{name}>"
            return Placeholder


class TransformDatabase:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.origin_documents = None

    def prepare_exist_db_index(self):
        try:
            # 临时替换 __setstate__，跳过 transformer 重建（避免缺失模块或参数导致失败）
            original_setstate = LocalDB.__setstate__
            LocalDB.__setstate__ = lambda self, state: self.__dict__.update(state)

            try:
                # 使用 SafeUnpickler 加载 pickle 数据，避免缺失模块导致反序列化失败
                with open(self.repo_path, "rb") as f:
                    self.db = SafeUnpickler(f).load()
                logger.info(f"DB loaded successfully. Items: {len(self.db.items)}, Transformed keys: {list(self.db.transformed_items.keys())}")
            finally:
                # 恢复原始的 __setstate__
                LocalDB.__setstate__ = original_setstate

            # 获取转换后的文档数据
            origin_documents = self.db.get_transformed_data(key="split_and_embed")
            if origin_documents:
                logger.info(f"Loaded {len(origin_documents)} documents from existing database")
                self.origin_documents = origin_documents
                return origin_documents
            else:
                logger.warning("No transformed data found with key 'split_and_embed'")

        except Exception as e:
            logger.error(f"Error loading existing database: {e}", exc_info=True)

        return None


if __name__ == "__main__":

    # 替换成你的 .pkl 文件路径
    # pkl_path = r"C:\Users\lenovo\AppData\Roaming\adalflow\databases\polls.pkl"  -> 这里应该使用 LocalDatabase 进行加载
    # vector_path = r"D:\ProgramFile2_OR\Python_Study_System\OpenStudy\rag_build\vector_stores\polls\index.pkl"
    # with open(vector_path, "rb") as f:
    #     data = SafeUnpickler(f).load()

    # print("✅ 加载成功！")
    # print("类型:", type(data))
    # print("内容预览:", data)

    repo_path = r"C:\Users\lenovo\AppData\Roaming\adalflow\databases\gitingest.pkl"
    transform_db = TransformDatabase(repo_path)
    origin_documents = transform_db.prepare_exist_db_index()
    if origin_documents:
        print(f"[OK] 成功加载 {len(origin_documents)} 个文档")
        print(f"第一个文档预览: {str(origin_documents[0])}")
    else:
        print("[FAIL] 加载失败，请检查日志")
