"""
Database ingestion script for InfraRAG system.
Processes all documents and stores chunks in PostgreSQL with embeddings.
"""

import os
from pathlib import Path
from src.ingestion import DocumentIngestionPipeline
from database_storage import DataStorage

def main():
    os.environ.setdefault('DB_HOST', 'localhost')
    os.environ.setdefault('DB_PORT', '5434')
    os.environ.setdefault('DB_NAME', 'infrarag_db')
    os.environ.setdefault('DB_USER', 'infrarag_user')
    os.environ.setdefault('DB_PASSWORD', 'infrarag_secure_pw_2024')
    
    pipeline = DocumentIngestionPipeline()
    storage = DataStorage()
    
    data_dir = Path("data")
    document_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.xml"))
    
    print(f"Found {len(document_files)} documents to process")
    
    all_chunks = []
    for file_path in document_files:
        try:
            print(f"Processing {file_path}...")
            document = pipeline.ingest_document(str(file_path))
            chunks = pipeline.chunk_document(document)
            all_chunks.extend(chunks)
            
            print(f"✅ Processed {file_path}: {len(chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"\nTotal chunks to save: {len(all_chunks)}")
    
    pipeline.save_chunks_to_database(all_chunks)
    
    total_chunks = storage.get_chunk_count()
    print(f"Database now contains {total_chunks:,} total chunks")

if __name__ == "__main__":
    main()
