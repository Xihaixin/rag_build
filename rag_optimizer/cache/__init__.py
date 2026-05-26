"""
缓存模块 — Redis 缓存层

包含：
- redis_client.py:     Redis 连接管理
- embedding_cache.py:  Embedding 缓存（内容哈希）
- semantic_cache.py:   语义缓存（相似问题直接返回）
- wiki_cache.py:       Wiki 双层缓存（Redis + PostgreSQL，Cache-Aside 模式）
- repo_lock.py:        仓库处理分布式锁
"""
