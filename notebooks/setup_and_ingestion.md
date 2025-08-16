# InfraRAG Setup and Data Ingestion Guide

This notebook provides step-by-step instructions for setting up the InfraRAG system and ingesting legal documents into the PostgreSQL database.

## Prerequisites

1. Docker and Docker Compose installed
2. Conda environment activated: `conda activate infra-rag`
3. All dependencies installed from `setup/requirements.txt`

## Step 1: Start Supporting Services

Start all required services using Docker Compose:

```bash
cd setup
docker-compose -f docker-compose.rag.yml up -d
```

This will start:
- **PostgreSQL** on port 5433 (main database with pgvector)
- **pgAdmin** on port 5050 (database management UI)
- **Qdrant** on port 6333 (vector database)
- **OpenSearch** on port 9200 (BM25 search engine)
- **OpenSearch Dashboards** on port 5601 (search UI)

## Step 2: Set Environment Variables

Set the required environment variables for database connection:

```bash
export DB_HOST=localhost
export DB_PORT=5433
export DB_NAME=infra_rag
export DB_USER=postgres
export DB_PASSWORD=changeme_local_pw
export OPENSEARCH_URL=http://localhost:9200
export OPENSEARCH_INDEX=agreements
export EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Step 3: Verify Database Connection

Test the database connection:

```python
from database_storage import DataStorage

try:
    storage = DataStorage()
    with storage.get_session() as db:
        result = db.execute("SELECT 1 as test")
        print("✅ Database connection successful!")
        
        result = db.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        if result.fetchone():
            print("✅ pgvector extension is installed")
        else:
            print("❌ pgvector extension not found")
            
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print("Make sure Docker services are running and try again.")
```

## Step 4: Run Data Ingestion

Process all documents in the data folder:

```python
from src.ingestion import DocumentIngestionPipeline
from database_storage import DataStorage
import os

pipeline = DocumentIngestionPipeline()
storage = DataStorage()

data_dir = "data"
document_files = []
for file in os.listdir(data_dir):
    if file.endswith(('.pdf', '.xml')):
        document_files.append(os.path.join(data_dir, file))

print(f"Found {len(document_files)} documents to process")

all_chunks = []
for file_path in document_files:
    try:
        print(f"Processing {file_path}...")
        document = pipeline.ingest_document(file_path)
        chunks = pipeline.chunk_document(document)
        all_chunks.extend(chunks)
        
        print(f"✅ Processed {file_path}: {len(chunks)} chunks")
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

print(f"\nTotal chunks to save: {len(all_chunks)}")

pipeline.save_chunks_to_database(all_chunks)
```

## Step 5: Verify Ingestion Results

Check the ingestion results:

```python
total_chunks = storage.get_chunk_count()
chunks_with_embeddings = storage.get_chunks_with_embeddings_count()

print(f"Total chunks in database: {total_chunks:,}")
print(f"Chunks with embeddings: {chunks_with_embeddings:,}")

search_results = storage.search_chunks("indemnification liability", limit=5)
print(f"\nSearch results for 'indemnification liability': {len(search_results)} results")

for i, result in enumerate(search_results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Source: {result['source']}")
    print(f"Section: {result['section_id']}")
    print(f"Rank: {result['rank']:.4f}")
    print(f"Content: {result['content'][:200]}...")
```

## Step 6: Access Database Management Tools

### pgAdmin (Database Management)
- URL: http://localhost:5050
- Email: admin@example.com
- Password: ChangeMe123!

### OpenSearch Dashboards (Search Analytics)
- URL: http://localhost:5601

## Troubleshooting

### Database Connection Issues
- Ensure Docker services are running: `docker-compose -f setup/docker-compose.rag.yml ps`
- Check PostgreSQL logs: `docker logs postgres-infrarag`
- Verify environment variables are set correctly

### Port Conflicts
- PostgreSQL uses port 5433 (not default 5432)
- pgAdmin uses port 5050
- Make sure these ports are available

### Performance Issues
- Monitor database performance through pgAdmin
- Consider adding more indices for frequently queried fields
- Adjust PostgreSQL configuration for larger datasets

## Next Steps

After successful ingestion:
1. Test hybrid search functionality combining PostgreSQL full-text search with vector similarity
2. Implement embedding generation for vector search capabilities
3. Set up the retrieval pipeline for document generation
4. Configure the web interface for document drafting
