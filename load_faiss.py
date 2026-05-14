import faiss
import numpy as np

def inspect_faiss_file(faiss_path: str, top_n: int = 5):
    # 1. 加载索引
    index = faiss.read_index(faiss_path)
    print("=" * 60)
    print("📌 FAISS 索引基础信息")
    print("=" * 60)
    print(f"索引对象类型: {type(index)}")
    print(f"向量维度 d: {index.d}")
    print(f"总向量数量: {index.ntotal}")
    print(f"是否已训练: {index.is_trained}")

    # 2. 判断是否带自定义 ID（IndexIDMap）
    id_map = None
    if isinstance(index, faiss.IndexIDMap):
        id_map = index.id_map
        print(f"✅ 包含自定义向量ID，ID总数: {len(id_map)}")
    else:
        print("❌ 未包装 IndexIDMap，无自定义ID")

    print("\n" + "=" * 60)
    print(f"🔍 查看前 {top_n} 条向量数据")
    print("=" * 60)

    # 3. 尝试重建原始向量
    if hasattr(index, "reconstruct_n") and index.ntotal > 0:
        take_num = min(top_n, index.ntotal)
        vecs = index.reconstruct_n(0, take_num)
        print(f"成功重建前 {take_num} 条原始向量")
        print(f"向量数组形状: {vecs.shape}")
        print("\n第一条向量示例：")
        print(vecs[0])

        # 4. 对应 ID 输出
        if id_map is not None:
            print("\n对应前{}个ID：".format(take_num))
            print(id_map[:take_num])
    else:
        print("⚠️ 当前是 PQ/压缩类索引，无法还原原始浮点向量，只能存编码，看不到真实原始嵌入")

if __name__ == "__main__":
    # ========== 在这里改你的 .faiss 文件路径 ==========
    FAISS_FILE_PATH = r".\vector_stores\polls\index.faiss"
    # 查看前多少条
    SHOW_TOP = 10
    # ==============================================

    inspect_faiss_file(FAISS_FILE_PATH, top_n=SHOW_TOP)