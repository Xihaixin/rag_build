import psycopg2
from psycopg2 import OperationalError

def connect_db():
    try:
        # 数据库连接参数
        dbname = "hello_postgre"
        user = "postgres"
        password = "200310"
        host = "localhost"
        port = "5432"

        # 连接数据库
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
        )
        print("✅ PostgreSQL 连接成功！")
        return conn

    except UnicodeDecodeError as e:
        print(f"❌ Unicode 解码错误：{e}")
        return None
    except OperationalError as e:
        print(f"❌ 数据库连接失败：{e}")
        return None

if __name__ == "__main__":
    conn = connect_db()
    if conn:
        conn.close()