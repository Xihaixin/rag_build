"""
缓存模块 — Redis 缓存层

包含：
- redis_client.py:     Redis 连接管理
- embedding_cache.py:  Embedding 缓存（内容哈希）
- semantic_cache.py:   语义缓存（相似问题直接返回）
- repo_lock.py:        仓库处理分布式锁
"""
