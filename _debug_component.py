from adalflow.core.component import Component
import inspect

print("MRO:", [c.__name__ for c in Component.__mro__])
print("has __setstate__:", hasattr(Component, '__setstate__'))
print("has __reduce__:", hasattr(Component, '__reduce__'))
print("has __reduce_ex__:", hasattr(Component, '__reduce_ex__'))
print("file:", inspect.getfile(Component))

# 检查 __setstate__ 是在哪个类上定义的
for cls in Component.__mro__:
    if '__setstate__' in cls.__dict__:
        print(f"__setstate__ defined on: {cls.__name__}")
        break
else:
    print("__setstate__ not defined on any class in MRO")

# 检查 __getstate__
for cls in Component.__mro__:
    if '__getstate__' in cls.__dict__:
        print(f"__getstate__ defined on: {cls.__name__}")
        break

# 检查 __reduce__ / __reduce_ex__
for attr in ['__reduce__', '__reduce_ex__', '__getnewargs__', '__getnewargs_ex__']:
    for cls in Component.__mro__:
        if attr in cls.__dict__:
            print(f"{attr} defined on: {cls.__name__}")
            break

# 查看 Component 的 __init__ 签名
sig = inspect.signature(Component.__init__)
print(f"Component.__init__ signature: {sig}")

# 查看 Component 源码中与 pickle 相关的部分
source_lines = inspect.getsource(Component).split('\n')
for i, line in enumerate(source_lines):
    if any(kw in line for kw in ['__setstate__', '__getstate__', '__reduce__', 'pickle', '_restore']):
        print(f"  L{i+1}: {line.strip()}")
