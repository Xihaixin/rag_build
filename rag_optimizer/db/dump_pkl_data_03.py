#!/usr/bin/env python3
"""dump_pkl_data.py — 将所有 pkl 数据库的内容导出到文本文件"""
 
import os
import glob
import pickle
 
 
# ============================================================
# 补丁：让 Component._restore_value 跳过组件恢复
# ============================================================
# adalflow 的 Component 使用 to_dict() / from_dict() 进行序列化，
# 反序列化时 _restore_value 会通过 EntityMapping 查找类并调用
# from_dict，这绕过了 Unpickler.find_class 的拦截。
#
# 解决方案：替换 _restore_value，让它只做简单的 dict/list 递归，
# 跳过 "type"+"data" 的类实例恢复逻辑。

_ORIGINAL_RESTORE_VALUE = None

def patch_restore_value():
    """替换 Component._restore_value 为安全版本"""
    global _ORIGINAL_RESTORE_VALUE
    try:
        from adalflow.core.component import Component
        _ORIGINAL_RESTORE_VALUE = Component._restore_value

        def safe_restore_value(value):
            """安全的 _restore_value：跳过类实例恢复，只处理 dict/list"""
            if isinstance(value, dict):
                if "_pickle_data" in value:
                    import pickle
                    return pickle.loads(bytes.fromhex(value["_pickle_data"]))
                if "_ordered_dict" in value and value["_ordered_dict"]:
                    from collections import OrderedDict
                    return OrderedDict(
                        (safe_restore_value(k), safe_restore_value(v))
                        for k, v in value["data"]
                    )
                # 跳过 "type"+"data" 的类实例恢复，直接返回原始 dict
                return {k: safe_restore_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [safe_restore_value(v) for v in value]
            return value

        Component._restore_value = staticmethod(safe_restore_value)
        return True
    except ImportError:
        return False


def unpatch_restore_value():
    """恢复原始 Component._restore_value"""
    global _ORIGINAL_RESTORE_VALUE
    if _ORIGINAL_RESTORE_VALUE is not None:
        try:
            from adalflow.core.component import Component
            Component._restore_value = _ORIGINAL_RESTORE_VALUE
        except ImportError:
            pass
 
 
# ============================================================
# 自定义 Unpickler：拦截缺失模块
# ============================================================
 
class PklDumpUnpickler(pickle.Unpickler):
    """
    双层防御的第一层：在 pickle 层面拦截缺失模块。
    主要处理 numpy 等非 adalflow 的依赖。
    adalflow 组件类的拦截交给 __setstate__ 补丁处理。
    """
 
    # 需要正常还原的类
    REAL_CLASSES = {}
 
    @classmethod
    def _ensure_real_classes_loaded(cls):
        if cls.REAL_CLASSES:
            return
        try:
            from adalflow.core.db import LocalDB
            cls.REAL_CLASSES[('adalflow.core.db', 'LocalDB')] = LocalDB
        except ImportError:
            pass
        try:
            from adalflow.core.types import Document
            cls.REAL_CLASSES[('adalflow.core.types', 'Document')] = Document
        except ImportError:
            pass
 
    def find_class(self, module, name):
        self._ensure_real_classes_loaded()
 
        # 需要正常还原的类
        key = (module, name)
        if key in self.REAL_CLASSES:
            return self.REAL_CLASSES[key]
 
        # 缺失的非 adalflow 模块 → Dummy
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return self._make_dummy(module, name)
 
    @staticmethod
    def _make_dummy(module, name):
        class Dummy:
            def __init__(self, *args, **kwargs):
                pass
            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)
            def __getattr__(self, attr):
                return None
            def __call__(self, *args, **kwargs):
                return Dummy()
            def __repr__(self):
                return f"<Dummy {module}.{name}>"
        Dummy.__module__ = module
        Dummy.__qualname__ = name
        return Dummy
 
 
def safe_load_pkl(pkl_path: str):
    """
    安全加载 pkl 文件，双层防御：
      1. Component._restore_value 补丁：跳过 "type"+"data" 的类实例恢复
      2. 自定义 Unpickler：处理缺失的非 adalflow 模块
    """
    patched = patch_restore_value()
    try:
        with open(pkl_path, "rb") as f:
            db = PklDumpUnpickler(f).load()
        return db
    finally:
        if patched:
            unpatch_restore_value()
 
 
# ============================================================
# 主逻辑
# ============================================================
 
root_path = r"C:\Users\lenovo\AppData\Roaming\adalflow"
db_dir = os.path.join(root_path, "databases")
 
pkl_files = glob.glob(os.path.join(db_dir, "*.pkl"))
if not pkl_files:
    print(f"未找到 pkl 文件，请检查路径: {db_dir}")
    exit(1)
 
print(f"找到 {len(pkl_files)} 个 pkl 文件:")
for pkl_path in pkl_files:
    file_name = os.path.basename(pkl_path).split(".")[0]
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"pkl_data_dump_{file_name}.txt")
    print(f"  - {os.path.basename(pkl_path)}")
 
    with open(output_path, "w", encoding="utf-8") as out:
        repo_name = os.path.splitext(os.path.basename(pkl_path))[0]
        out.write(f"\n{'='*80}\n")
        out.write(f"REPO: {repo_name}\n")
        out.write(f"FILE: {pkl_path}\n")
        out.write(f"{'='*80}\n\n")
 
        try:
            db = safe_load_pkl(pkl_path)
        except Exception as e:
            out.write(f"加载失败: {e}\n\n")
            import traceback
            out.write(traceback.format_exc())
            continue
 
        # 输出 DB 整体结构信息
        out.write(f"DB 类型: {type(db).__name__}\n")
        out.write(f"DB 名称: {getattr(db, 'name', 'N/A')}\n")
 
        items = getattr(db, 'items', [])
        out.write(f"原始文档数 (items): {len(items)}\n")
 
        transformed_items = getattr(db, 'transformed_items', {})
        for key in transformed_items:
            out.write(f"转换数据 key='{key}', 文档数: {len(transformed_items[key])}\n")
        out.write("\n")
 
        # ---- 原始文档 ----
        out.write(f"--- 原始文档 (共 {len(items)} 条) ---\n\n")
        for i, doc in enumerate(items):
            meta = getattr(doc, 'meta_data', {}) or {}
            text = getattr(doc, 'text', '')
            if not isinstance(meta, dict):
                meta = {}
            out.write(f"[{i}] file_path={meta.get('file_path')}, "
                    f"type={meta.get('type')}, "
                    f"is_code={meta.get('is_code')}, "
                    f"is_implementation={meta.get('is_implementation')}, "
                    f"token_count={meta.get('token_count')}\n")
            out.write(f"    text_length={len(text)}\n")
            out.write(f"    text_full=<<<\n{text}\n>>>\n\n")
 
        # ---- 转换后文档（分块 + 嵌入） ----
        for key in transformed_items:
            transformed = transformed_items[key]
            if transformed is None:
                out.write(f"--- 转换数据 key='{key}' 无数据 ---\n\n")
                continue
 
            out.write(f"--- 转换后文档 key='{key}' (共 {len(transformed)} 条) ---\n\n")
            for i, doc in enumerate(transformed):
                meta = getattr(doc, 'meta_data', {}) or {}
                text = getattr(doc, 'text', '')
                vec = getattr(doc, 'vector', None)
                if not isinstance(meta, dict):
                    meta = {}
                vec_dim = len(vec) if vec is not None else 0
                vec_sample = (str(vec[:10].tolist()) if hasattr(vec, 'tolist')
                              else str(vec[:10]) if vec is not None else "None")
                out.write(f"[{i}] file_path={meta.get('file_path')}, "
                        f"chunk_text_len={len(text)}, "
                        f"vector_dim={vec_dim}\n")
                out.write(f"    is_code={meta.get('is_code')}, "
                        f"token_count={meta.get('token_count')}\n")
                out.write(f"    vector_sample(first10)={vec_sample}\n")
                out.write(f"    chunk_full=<<<\n{text}\n>>>\n\n")
 
    print(f"  → {output_path}")
 
print(f"\n全部导出完成")