"""
Postgres-only indexing system for legal documents using pgvector.
Indexes chunks into PostgreSQL and supports vector search.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import psycopg

from .schema import ProcessedChunk, LegalDocument


class PgIndexer:
	"""Postgres-based indexer using pgvector"""

	def __init__(self):
		self.db_host = os.getenv("DB_HOST", "localhost")
		self.db_port = int(os.getenv("DB_PORT", "5433"))
		self.db_name = os.getenv("DB_NAME", "infra_rag")
		self.db_user = os.getenv("DB_USER", "postgres")
		self.db_password = os.getenv("DB_PASSWORD", "changeme_local_pw")

	def _connect(self):
		return psycopg.connect(
			host=self.db_host,
			port=self.db_port,
			dbname=self.db_name,
			user=self.db_user,
			password=self.db_password,
		)

	def upsert_document(self, document: LegalDocument):
		"""Insert or update a document row"""
		sql = (
			"INSERT INTO documents (document_id, title, document_type, jurisdiction, governing_law, industry)\n"
			"VALUES (%s, %s, %s, %s, %s, %s)\n"
			"ON CONFLICT (document_id) DO UPDATE SET\n"
			"  title=EXCLUDED.title,\n"
			"  document_type=EXCLUDED.document_type,\n"
			"  jurisdiction=EXCLUDED.jurisdiction,\n"
			"  governing_law=EXCLUDED.governing_law,\n"
			"  industry=EXCLUDED.industry"
		)
		with self._connect() as conn, conn.cursor() as cur:
			meta = document.metadata
			cur.execute(
				sql,
				(
					meta.document_id,
					meta.title,
					meta.document_type,
					meta.jurisdiction,
					meta.governing_law,
					meta.industry,
				),
			)

	def index_chunks(self, document: LegalDocument, chunks: List[ProcessedChunk], embeddings: Optional[List[List[float]]] = None):
		"""Insert chunks into clauses table. If embeddings provided, store them."""
		insert_sql = (
			"INSERT INTO clauses (document_id, section_id, title, content, tags, embedding)\n"
			"VALUES (%s, %s, %s, %s, %s, %s)"
		)
		with self._connect() as conn, conn.cursor() as cur:
			for i, chunk in enumerate(chunks):
				section_title = None
				for s in document.sections:
					if s.id == chunk.metadata.section_id:
						section_title = s.title
						break
				emb = None
				if embeddings and i < len(embeddings):
					emb = "[" + ",".join(f"{x:.8f}" for x in embeddings[i]) + "]"
				cur.execute(
					insert_sql,
					(
						document.metadata.document_id,
						chunk.metadata.section_id,
						section_title or chunk.metadata.section_id,
						chunk.content,
						chunk.metadata.tags,
						emb,
					),
				)

	def search(self, query_vector: List[float], limit: int = 20) -> List[Dict[str, Any]]:
		"""Vector search over clauses using pgvector"""
		qvec = "[" + ",".join(f"{x:.8f}" for x in query_vector) + "]"
		sql = (
			"SELECT document_id, section_id, title, content, 1.0 / (1.0 + (embedding <-> %s::vector)) AS score\n"
			"FROM clauses WHERE embedding IS NOT NULL\n"
			"ORDER BY embedding <-> %s::vector LIMIT %s"
		)
		with self._connect() as conn, conn.cursor() as cur:
			cur.execute(sql, (qvec, qvec, limit))
			rows = cur.fetchall()
		return [
			{"document_id": r[0], "section_id": r[1], "title": r[2], "content": r[3], "score": float(r[4])}
			for r in rows
		]

	def get_index_stats(self) -> Dict[str, Any]:
		"""Basic counts from tables"""
		with self._connect() as conn, conn.cursor() as cur:
			cur.execute("SELECT COUNT(*) FROM documents")
			docs = cur.fetchone()[0]
			cur.execute("SELECT COUNT(*) FROM clauses")
			clauses = cur.fetchone()[0]
		return {"documents": docs, "clauses": clauses}


def main():
	"""CLI for Postgres-only indexing"""
	import argparse
	import json
	from sentence_transformers import SentenceTransformer
	from .ingestion import DocumentIngestionPipeline

	parser = argparse.ArgumentParser(description="Index legal documents into Postgres (pgvector)")
	parser.add_argument("files", nargs="+", help="Documents to ingest and index")
	parser.add_argument("--no-embed", action="store_true", help="Skip embeddings (no vector search)")
	args = parser.parse_args()

	pipeline = DocumentIngestionPipeline()
	indexer = PgIndexer()

	model = None
	if not args.no_embed:
		model = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

	for f in args.files:
		doc = pipeline.ingest_document(f)
		chunks = pipeline.chunk_document(doc)
		embs = None
		if model:
			texts = [c.content for c in chunks]
			embs = model.encode(texts, normalize_embeddings=True).tolist()
		indexer.upsert_document(doc)
		indexer.index_chunks(doc, chunks, embs)

	print(json.dumps(indexer.get_index_stats(), indent=2))


if __name__ == "__main__":
	main()
