import os
from typing import List, Dict, Any, Optional

import psycopg
from sentence_transformers import SentenceTransformer

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "infra_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "changeme_local_pw")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_MODAL_EMBED = os.getenv("USE_MODAL_EMBED", "0") in {"1", "true", "True"}

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
	global _model
	if _model is None:
		_model = SentenceTransformer(EMBED_MODEL_NAME)
	return _model


def _embed_query(query: str) -> List[float]:
	if USE_MODAL_EMBED:
		from adapters.modal_embedding import embed_texts_remote
		return embed_texts_remote([query])[0]
	model = _get_model()
	vec = model.encode([query], normalize_embeddings=True)[0]
	return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def _vector_to_sql_literal(vec: List[float]) -> str:
	return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _search_pg(query_vec: List[float], limit: int = 15) -> List[Dict[str, Any]]:
	qvec = _vector_to_sql_literal(query_vec)
	sql = (
		"SELECT document_id, section_id, title, content, "
		"  1.0 / (1.0 + (embedding <-> %s::vector)) AS score "
		"FROM clauses "
		"WHERE embedding IS NOT NULL "
		"ORDER BY embedding <-> %s::vector "
		"LIMIT %s"
	)
	params = (qvec, qvec, limit)
	results: List[Dict[str, Any]] = []
	conn = psycopg.connect(
		host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
	)
	try:
		with conn.cursor() as cur:
			cur.execute(sql, params)
			for row in cur.fetchall():
				results.append(
					{
						"document_id": row[0],
						"section_id": row[1],
						"title": row[2],
						"content": row[3],
						"score": float(row[4]) if row[4] is not None else 0.0,
						"id": f"{row[0]}:{row[1]}",
					}
				)
	finally:
		conn.close()
	return results


def search(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
	qvec = _embed_query(query)
	return _search_pg(qvec, limit=top_k)


# Ingestion helper to index files from data/ into Postgres

def ingest_data_dir(data_dir: str = "./data") -> Dict[str, Any]:
	"""Parse XML/PDF files, chunk, embed, and index into Postgres.

	Returns counts and simple stats.
	"""
	from pathlib import Path
	from src.ingestion import DocumentIngestionPipeline
	from src.indexing import PgIndexer

	path = Path(data_dir)
	files: List[str] = []
	for ext in ("*.xml", "*.pdf"):
		files.extend([str(p) for p in path.glob(ext)])

	pipeline = DocumentIngestionPipeline()
	indexer = PgIndexer()

	num_docs = 0
	num_chunks = 0
	for f in files:
		doc = pipeline.ingest_document(f)
		chunks = pipeline.chunk_document(doc)
		texts = [c.content for c in chunks]
		if USE_MODAL_EMBED:
			from adapters.modal_embedding import embed_texts_remote
			embs = embed_texts_remote(texts)
		else:
			model = _get_model()
			embs = model.encode(texts, normalize_embeddings=True).tolist()
		indexer.upsert_document(doc)
		indexer.index_chunks(doc, chunks, embs)
		num_docs += 1
		num_chunks += len(chunks)

	stats = {"documents_indexed": num_docs, "chunks_indexed": num_chunks}
	stats.update(indexer.get_index_stats())
	return stats 