## InfraRAG Runbook (execution.md)

This is a concise reference for starting services, (re)ingesting data, building the clause graph, and testing retrieval.

### 0) Prereqs
- Environment: activate your env and load `.env` (Modal keys optional)
```bash
conda activate infra-rag  # or your venv
export USE_MODAL_EMBED=1  # optional: offload embeddings to Modal
source .env               # DB creds, Modal, etc.
```
- Start Postgres + pgAdmin
```bash
docker compose -f docker-compose.local.yml up -d
```

### 1) Ingest documents + embed (one-time or after parser changes)
- Ingest everything in `./data` and embed:
```bash
python database/ingest_to_database.py --data-dir ./data            # local CPU embeddings
python database/ingest_to_database.py --data-dir ./data --use-modal # Modal embeddings
```
- Only embed missing rows (skip ingest), useful after a partial run:
```bash
python database/ingest_to_database.py --backfill-missing            # local CPU
python database/ingest_to_database.py --backfill-missing --use-modal # Modal
```

Notes
- Use `--data-dir` when you need to re-chunk/re-embed after parser changes.
- Use `--backfill-missing` when data is already in `clauses` but `embedding IS NULL`.

### 2) Re-index (force clean) when chunking changed
- Option A: Purge all clauses, then re-ingest
```bash
PGPASSWORD=changeme_local_pw psql -h localhost -p 5433 -U postgres -d infra_rag -c "TRUNCATE TABLE clauses"
# Bigger batches and more parallel remote calls
export EMBED_BATCH_SIZE=256
export EMBED_PARALLEL=8

# Then ingest (uses Modal)
python database/ingest_to_database.py --data-dir ./data --use-modal
```
- Option B: Purge one document, then re-ingest
```bash
DOC=39638_sanitized
PGPASSWORD=changeme_local_pw psql -h localhost -p 5433 -U postgres -d infra_rag -c "DELETE FROM clauses WHERE document_id='${DOC}'"
python database/ingest_to_database.py --data-dir ./data --use-modal
```

### 3) Clause graph (edges) and recursive retrieval
- Build/populate edges (adjacent, refers_to, defines):
```bash
python database/test_search.py --populate-graph
```
- Hybrid search (BM25 + vector, RRF):
```bash
python database/test_search.py "Interest rates" --hybrid --pool-n 200 --top-k 5
```
- Recursive retrieval (BFS over graph; follows refers_to/defines):
```bash
python database/test_search.py "Interest rates" --recursive --hops 2 --top-k 8
```

### 4) Sanity / debugging
- Check if specific phrases “bled” into unrelated sections:
```bash
docker exec -i infra-postgres psql -U postgres -d infra_rag -f - \
  < /Users/hakeemshindy/Desktop/InfraRAG/adapters/debug.sql
```

- Quick index stats in Python:
```bash
python -c 'from src.indexing import PgIndexer; print(PgIndexer().get_index_stats())'
```

### 5) Programmatic usage (examples)
- Hybrid with must-match filters and sparse weighting (in code):
```python
from adapters.retrieval_adapter import search_hybrid
res = search_hybrid(
  "interest rates arbitral award",
  pool_n=200,
  top_k=8,
  sparse_weight=0.7,                  # weight sparse > dense for legal queries
  must_tokens=["pre-award","arbitral award","360","LIBOR"],
)
for r in res: print(r["rrf"], r["title"])  # etc.
```
- Recursive retrieval (follows clause references):
```python
from adapters.retrieval_adapter import populate_clause_graph, retrieve_recursive
populate_clause_graph()
res = retrieve_recursive("Interest rates", k=8, max_hops=2)


checking db:
docker exec -i infra-postgres psql -U postgres -d infra_rag -c "SELECT document_id, COUNT(*) AS clauses FROM clauses GROUP BY 1 ORDER BY 1"
docker exec -i infra-postgres psql -U postgres -d infra_rag -f - \
  < /Users/hakeemshindy/Desktop/InfraRAG/adapters/test.sql
```

### 6) Tuning knobs (env vars)
- `IVFFLAT_LISTS` (default 200): vector index list count
- `IVFFLAT_PROBES` (default 10): number of probes per query
- `RRF_K` (default 60): RRF damping constant
- `EMBED_MODEL` (default sentence-transformers/all-MiniLM-L6-v2)

### 7) Troubleshooting
- First Modal run can take ~10 minutes (image build + model download). Subsequent runs are fast.
- If CLI says “No missing embeddings to backfill” but you changed chunking, re-run a full ingest with `--data-dir` (not `--backfill-missing`) or purge rows per §2.
- If FTS errors: adapter auto-creates/maintains `fts` (weighted) and `fts_simple` via trigger; no manual updates needed. 