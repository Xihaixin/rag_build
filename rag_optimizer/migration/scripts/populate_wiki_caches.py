"""
迁移脚本：从 wiki_pages 表数据填充 wiki_caches 表

当 wiki_caches 表缺失记录但 wiki_pages 表已有数据时，
此脚本会为每个项目/语言组合重建 wiki_caches 记录。

用法：
    cd /d d:/ProgramFile2_OR/Python_Study_System/OpenStudy/rag_build
    .venv/cripts/python scripts/populate_wiki_caches.py
"""

import json
import logging
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_optimizer.db.connection import sync_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_projects_with_pages():
    """获取所有在 wiki_pages 中有数据的项目"""
    result = sync_conn.execute("""
        SELECT DISTINCT
            p.id,
            p.name,
            p.owner,
            p.repo_url,
            p.repo_type
        FROM projects p
        INNER JOIN wiki_pages wp ON wp.project_id = p.id
        ORDER BY p.name
    """)
    return [dict(r) for r in result] if result else []


def get_languages_for_project(project_id: str):
    """获取项目在 wiki_pages 中存在的语言"""
    result = sync_conn.execute(
        "SELECT DISTINCT language FROM wiki_pages WHERE project_id = %s",
        (project_id,)
    )
    return [r["language"] for r in result] if result else []


def get_pages_for_project(project_id: str, language: str):
    """获取项目在指定语言下的所有 wiki 页面"""
    result = sync_conn.execute(
        "SELECT page_slug, title, content_md FROM wiki_pages WHERE project_id = %s AND language = %s ORDER BY created_at",
        (project_id, language)
    )
    return [dict(r) for r in result] if result else []


def check_cache_exists(project_id: str, language: str) -> bool:
    """检查 wiki_caches 是否已有记录"""
    result = sync_conn.execute(
        "SELECT id FROM wiki_caches WHERE project_id = %s AND language = %s",
        (project_id, language)
    )
    return bool(result)


def build_structure_json(project_name: str, project_id: str, pages: list) -> dict:
    """从 wiki_pages 数据重建 structure_json"""
    rebuilt_pages = []
    for p in pages:
        rebuilt_pages.append({
            "id": p["page_slug"],
            "title": p["title"],
            "content": p["content_md"] or "",
            "filePaths": [],
            "importance": "medium",
            "relatedPages": [],
        })

    return {
        "id": project_id,
        "title": project_name,
        "description": f"Wiki for {project_name}",
        "pages": rebuilt_pages,
        "sections": [],
        "rootSections": [],
    }


def populate_cache_for_project(project: dict, language: str, pages: list) -> bool:
    """为项目/语言组合创建 wiki_caches 记录"""
    if check_cache_exists(project["id"], language):
        logger.info(f"  [SKIP] Cache already exists: {project['name']}/{language}")
        return False

    structure_json = build_structure_json(project["name"], project["id"], pages)

    try:
        sync_conn.execute(
            """INSERT INTO wiki_caches
               (project_id, language, structure_json,
                repo_owner, repo_name, repo_type, repo_url,
                provider, model)
               VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)""",
            (
                project["id"],
                language,
                json.dumps(structure_json, ensure_ascii=False),
                project.get("owner"),
                project["name"],
                project.get("repo_type", "github"),
                project.get("repo_url", ""),
                None,  # provider
                None,  # model
            )
        )
        logger.info(f"  [OK] Created cache: {project['name']}/{language} ({len(pages)} pages)")
        return True
    except Exception as e:
        logger.error(f"  [FAIL] {project['name']}/{language}: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("Starting wiki_caches population from wiki_pages data")
    logger.info("=" * 60)

    projects = get_projects_with_pages()
    logger.info(f"Found {len(projects)} projects with wiki pages")

    total_created = 0
    total_skipped = 0
    total_failed = 0

    for project in projects:
        logger.info(f"\nProject: {project['name']} (id: {project['id']})")
        languages = get_languages_for_project(project["id"])
        logger.info(f"  Languages: {languages}")

        for lang in languages:
            pages = get_pages_for_project(project["id"], lang)
            logger.info(f"  Processing: {project['name']}/{lang} ({len(pages)} pages)")

            if populate_cache_for_project(project, lang, pages):
                total_created += 1
            elif check_cache_exists(project["id"], lang):
                total_skipped += 1
            else:
                total_failed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"Summary: {total_created} created, {total_skipped} skipped, {total_failed} failed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
