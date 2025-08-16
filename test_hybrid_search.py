"""
Test script for hybrid search functionality using PostgreSQL database.
"""

import os
from database_storage import DataStorage

def main():
    os.environ.setdefault('DB_HOST', 'localhost')
    os.environ.setdefault('DB_PORT', '5433')
    os.environ.setdefault('DB_NAME', 'infra_rag')
    os.environ.setdefault('DB_USER', 'postgres')
    os.environ.setdefault('DB_PASSWORD', 'changeme_local_pw')
    
    storage = DataStorage()
    
    print("Testing database connection...")
    try:
        with storage.get_session() as db:
            from sqlalchemy import text
            result = db.execute(text("SELECT 1 as test"))
            print("✅ Database connection successful!")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return
    
    total_chunks = storage.get_chunk_count()
    print(f"Total chunks in database: {total_chunks:,}")
    
    if total_chunks == 0:
        print("No chunks found. Run ingest_to_database.py first.")
        return
    
    test_queries = [
        "indemnification liability damages",
        "purchase price adjustment",
        "governing law arbitration",
        "force majeure",
        "representations warranties"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")
        
        results = storage.search_chunks(query, limit=3)
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Source: {result['source']}")
            print(f"  Section: {result['section_id']}")
            print(f"  Rank: {result['rank']:.4f}")
            print(f"  Content: {result['content'][:150]}...")

if __name__ == "__main__":
    main()
