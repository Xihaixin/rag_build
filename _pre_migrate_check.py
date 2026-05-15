"""检查并清理旧数据，为新的迁移做准备"""
import psycopg2

conn = psycopg2.connect(host='localhost', port=5432, dbname='rag_optimizer', user='postgres', password='200310')
conn.autocommit = True
cur = conn.cursor()

# 检查 gitingest 项目
cur.execute("SELECT id, name FROM projects WHERE name = 'gitingest'")
row = cur.fetchone()
if row:
    project_id = row[0]
    print(f"Found project: id={project_id}, name={row[1]}")
    
    # 检查关联数据
    cur.execute("SELECT COUNT(*) FROM raw_documents WHERE project_id = %s", (project_id,))
    doc_count = cur.fetchone()[0]
    print(f"  raw_documents: {doc_count}")
    
    cur.execute("""SELECT COUNT(*) FROM document_chunks dc 
                   JOIN raw_documents rd ON dc.document_id = rd.id 
                   WHERE rd.project_id = %s""", (project_id,))
    chunk_count = cur.fetchone()[0]
    print(f"  document_chunks: {chunk_count}")
    
    cur.execute("SELECT COUNT(*) FROM chunk_embeddings_dim256 WHERE project_id = %s", (project_id,))
    embed_count = cur.fetchone()[0]
    print(f"  chunk_embeddings_dim256: {embed_count}")
    
    # 检查文档内容长度
    cur.execute("""SELECT file_path, LENGTH(content) as content_len 
                   FROM raw_documents WHERE project_id = %s 
                   ORDER BY content_len DESC LIMIT 5""", (project_id,))
    print("\nTop 5 documents by content length:")
    for r in cur.fetchall():
        print(f"  {r[0]}: {r[1]} chars")
    
    # 询问是否清理
    print("\n⚠ 旧数据存在，建议先清理再重新迁移。")
    print("运行以下 SQL 清理（通过 pgAdmin 或 psql）：")
    print(f"  DELETE FROM chunk_embeddings_dim256 WHERE project_id = '{project_id}';")
    print(f"  DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM raw_documents WHERE project_id = '{project_id}');")
    print(f"  DELETE FROM document_versions WHERE document_id IN (SELECT id FROM raw_documents WHERE project_id = '{project_id}');")
    print(f"  DELETE FROM raw_documents WHERE project_id = '{project_id}';")
    print(f"  DELETE FROM pipeline_logs WHERE project_id = '{project_id}';")
    print(f"  DELETE FROM ingestion_jobs WHERE project_id = '{project_id}';")
    print(f"  DELETE FROM projects WHERE id = '{project_id}';")
else:
    print("No gitingest project found - clean state, ready for migration")

cur.close()
conn.close()
