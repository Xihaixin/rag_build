"""
Wiki 缓存 — Redis + PostgreSQL 双层缓存

采用 Cache-Aside 模式：
- 读：先查 Redis（热缓存）→ 未命中则查 PostgreSQL → 回填 Redis
- 写：先写 PostgreSQL（持久化）→ 再写/更新 Redis
- 删：删除 PostgreSQL 数据 → 同时删除 Redis 键

Redis key 格式:
  wiki:cache:{project_id}:{language}          — Wiki 结构 + 元数据
  wiki:page:{project_id}:{page_slug}:{language} — 单个 Wiki 页面
  wiki:projects:list                           — 已处理项目列表（缓存）
"""

import json
import logging
from typing import Any, Dict, List, Optional

from rag_optimizer.cache.redis_client import redis_client
from rag_optimizer.config.settings import settings
from rag_optimizer.db.repository import (
    WikiCacheRepository,
    WikiPageRepository,
)

logger = logging.getLogger(__name__)


# Redis key 前缀
CACHE_PREFIX = "wiki:cache"
PAGE_PREFIX = "wiki:page"
PROJECTS_KEY = "wiki:projects:list"

# 默认 TTL（秒）：7 天
DEFAULT_TTL = 7 * 24 * 3600


class WikiCacheManager:
    """Wiki 双层缓存管理器（Redis + PostgreSQL）"""

    # ── 内部 key 构建 ──────────────────────────────────────────────

    @staticmethod
    def _cache_key(project_id: str, language: str) -> str:
        return f"{CACHE_PREFIX}:{project_id}:{language}"

    @staticmethod
    def _page_key(project_id: str, page_slug: str, language: str) -> str:
        return f"{PAGE_PREFIX}:{project_id}:{page_slug}:{language}"

    # ── 读操作：Cache-Aside ────────────────────────────────────────

    def get_cache(self, project_id: str, language: str) -> Optional[dict]:
        """获取 Wiki 缓存（Redis → PostgreSQL）"""
        # 1. 先查 Redis
        redis_key = self._cache_key(project_id, language)
        cached = redis_client.get(redis_key)
        if cached:
            try:
                data = json.loads(cached)
                logger.debug(f"Wiki cache HIT (Redis): {project_id}/{language}")
                return data
            except (json.JSONDecodeError, TypeError):
                pass

        # 2. Redis 未命中，查 PostgreSQL
        record = WikiCacheRepository.get_by_project(project_id, language)
        if not record:
            return None

        # 3. 回填 Redis
        try:
            redis_client.set(redis_key, json.dumps(record, ensure_ascii=False), ttl=DEFAULT_TTL)
        except Exception as e:
            logger.debug(f"Failed to backfill Redis cache: {e}")

        logger.debug(f"Wiki cache MISS (Redis), loaded from DB: {project_id}/{language}")
        return record

    def get_page(self, project_id: str, page_slug: str, language: str) -> Optional[dict]:
        """获取单个 Wiki 页面（Redis → PostgreSQL）"""
        # 1. 先查 Redis
        redis_key = self._page_key(project_id, page_slug, language)
        cached = redis_client.get(redis_key)
        if cached:
            try:
                data = json.loads(cached)
                logger.debug(f"Wiki page HIT (Redis): {page_slug}")
                return data
            except (json.JSONDecodeError, TypeError):
                pass

        # 2. Redis 未命中，查 PostgreSQL
        record = WikiPageRepository.get_by_slug(project_id, page_slug, language)
        if not record:
            return None

        # 3. 回填 Redis
        try:
            redis_client.set(redis_key, json.dumps(record, ensure_ascii=False, default=str), ttl=DEFAULT_TTL)
        except Exception as e:
            logger.debug(f"Failed to backfill Redis page cache: {e}")

        logger.debug(f"Wiki page MISS (Redis), loaded from DB: {page_slug}")
        return record

    # ── 写操作：先写 PostgreSQL，再写 Redis ────────────────────────

    def save_cache(
        self,
        project_id: str,
        language: str,
        structure_json: dict,
        repo_owner: Optional[str] = None,
        repo_name: Optional[str] = None,
        repo_type: Optional[str] = None,
        repo_url: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """保存 Wiki 缓存（PostgreSQL → Redis）"""
        # 1. 先写 PostgreSQL
        record_id = WikiCacheRepository.upsert(
            project_id=project_id,
            language=language,
            structure_json=structure_json,
            repo_owner=repo_owner,
            repo_name=repo_name,
            repo_type=repo_type,
            repo_url=repo_url,
            provider=provider,
            model=model,
        )

        # 2. 再写 Redis
        try:
            redis_key = self._cache_key(project_id, language)
            cache_data = {
                "id": record_id,
                "project_id": project_id,
                "language": language,
                "structure_json": structure_json,
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "repo_type": repo_type,
                "repo_url": repo_url,
                "provider": provider,
                "model": model,
            }
            redis_client.set(redis_key, json.dumps(cache_data, ensure_ascii=False), ttl=DEFAULT_TTL)
        except Exception as e:
            logger.debug(f"Failed to update Redis cache: {e}")

        # 3. 清除项目列表缓存（强制下次重新加载）
        try:
            redis_client.delete(PROJECTS_KEY)
        except Exception:
            pass

        return record_id

    def save_page(
        self,
        project_id: str,
        page_slug: str,
        title: str,
        content_md: str,
        language: str = "zh",
        is_comprehensive: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """保存单个 Wiki 页面（PostgreSQL → Redis）"""
        # 1. 先写 PostgreSQL
        page_id = WikiPageRepository.upsert(
            project_id=project_id,
            page_slug=page_slug,
            title=title,
            content_md=content_md,
            language=language,
            is_comprehensive=is_comprehensive,
            provider=provider,
            model=model,
            source_chunks=source_chunks,
        )

        # 2. 再写 Redis
        try:
            redis_key = self._page_key(project_id, page_slug, language)
            page_data = {
                "id": page_id,
                "project_id": project_id,
                "page_slug": page_slug,
                "title": title,
                "content_md": content_md,
                "language": language,
                "is_comprehensive": is_comprehensive,
                "provider": provider,
                "model": model,
            }
            redis_client.set(redis_key, json.dumps(page_data, ensure_ascii=False), ttl=DEFAULT_TTL)
        except Exception as e:
            logger.debug(f"Failed to update Redis page cache: {e}")

        return page_id

    # ── 删操作：同时删除 PostgreSQL 和 Redis ──────────────────────

    def delete_cache(self, project_id: str, language: str) -> bool:
        """删除 Wiki 缓存（PostgreSQL + Redis）"""
        # 1. 删除 Redis
        try:
            redis_key = self._cache_key(project_id, language)
            redis_client.delete(redis_key)
        except Exception as e:
            logger.debug(f"Failed to delete Redis cache: {e}")

        # 2. 删除 PostgreSQL
        result = WikiCacheRepository.delete(project_id, language)

        # 3. 清除项目列表缓存
        try:
            redis_client.delete(PROJECTS_KEY)
        except Exception:
            pass

        return result

    def delete_project_cache(self, project_id: str):
        """删除项目的所有缓存（PostgreSQL + Redis）"""
        # 1. 删除 Redis（按模式匹配）
        try:
            cache_keys = redis_client.keys(f"{CACHE_PREFIX}:{project_id}:*")
            page_keys = redis_client.keys(f"{PAGE_PREFIX}:{project_id}:*")
            for key in (cache_keys or []) + (page_keys or []):
                redis_client.delete(key)
        except Exception as e:
            logger.debug(f"Failed to delete Redis project keys: {e}")

        # 2. 删除 PostgreSQL
        WikiCacheRepository.delete_by_project(project_id)
        WikiPageRepository.delete_by_project(project_id)

        # 3. 清除项目列表缓存
        try:
            redis_client.delete(PROJECTS_KEY)
        except Exception:
            pass

    # ── 项目列表缓存 ──────────────────────────────────────────────

    def get_cached_projects(self) -> Optional[List[dict]]:
        """获取缓存的项目列表（Redis）"""
        cached = redis_client.get(PROJECTS_KEY)
        if cached:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    def set_cached_projects(self, projects: List[dict]):
        """缓存项目列表（Redis）"""
        try:
            redis_client.set(PROJECTS_KEY, json.dumps(projects, ensure_ascii=False, default=str), ttl=DEFAULT_TTL)
        except Exception as e:
            logger.debug(f"Failed to cache projects list: {e}")


# 全局单例
wiki_cache_manager = WikiCacheManager()
