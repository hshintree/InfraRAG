"""
Test database connection with new configuration.
"""

import os
from database_storage import DataStorage

def main():
    os.environ.setdefault('DB_HOST', 'localhost')
    os.environ.setdefault('DB_PORT', '5433')
    os.environ.setdefault('DB_NAME', 'infra_rag')
    os.environ.setdefault('DB_USER', 'postgres')
    os.environ.setdefault('DB_PASSWORD', 'changeme_local_pw')
    
    try:
        storage = DataStorage()
        with storage.get_session() as db:
            from sqlalchemy import text
            result = db.execute(text("SELECT 1 as test"))
            print("✅ Database connection successful!")
            
            result = db.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'"))
            if result.fetchone():
                print("✅ pgvector extension is installed")
            else:
                print("❌ pgvector extension not found")
                
            result = db.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            tables = [row[0] for row in result.fetchall()]
            print(f"✅ Database tables: {tables}")
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
