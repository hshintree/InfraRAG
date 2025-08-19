"""
Postgres-only indexing system for legal documents using pgvector.
Indexes chunks into PostgreSQL and supports vector search.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
import re
import psycopg

from .schema import ProcessedChunk, LegalDocument

# Heuristics for clause typing / defined terms
_CLAUSE_MAP = {
	"indemn": "Indemnities",
	"liabil": "Indemnities",
	"cap ": "Indemnities",
	" basket": "Indemnities",
	"definition": "Definitions",
	"definit": "Definitions",
	"payment": "Payment",
	"price": "Payment",
	"true-up": "Payment",
	"title": "Title & Risk",
	"risk": "Title & Risk",
	"delivery": "Delivery",
	"force majeure": "Force Majeure",
	"change in law": "Change in Law",
	"governing law": "Governing Law",
	" arbitration": "Dispute Resolution",
	"dispute": "Dispute Resolution",
	"notices": "Notices",
	"termination": "Termination",
	"conditions precedent": "Conditions Precedent",
	" cp ": "Conditions Precedent",
	"insurance": "Insurance",
	"guarantee": "Guarantee",
	"assignment": "Assignment",
	"audit": "Audit & Inspection",
	"meter": "Quantity & Metering",
	"quantity": "Quantity & Metering",
	"quality": "Quality & Specs",
	"specification": "Quality & Specs",
	"tax": "Tax",
	"shipping": "Shipping/Transport",
	"vessel": "Shipping/Transport",
	"tanker": "Shipping/Transport",
	"demurrage": "Shipping/Transport",
}


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

	def _column_exists(self, cur, table: str, column: str) -> bool:
		cur.execute(
			"""
			SELECT 1
			FROM information_schema.columns
			WHERE table_name=%s AND column_name=%s
			""",
			(table, column),
		)
		return cur.fetchone() is not None

	def _ensure_optional_schema(self):
		"""Ensure helpful columns and unique index exist for idempotent upserts."""
		with self._connect() as conn, conn.cursor() as cur:
			# optional columns
			cur.execute(
				"""
				ALTER TABLE IF EXISTS clauses
					ADD COLUMN IF NOT EXISTS seq int,
					ADD COLUMN IF NOT EXISTS clause_type text,
					ADD COLUMN IF NOT EXISTS defined_terms text[],
					ADD COLUMN IF NOT EXISTS content_hash text,
					ADD COLUMN IF NOT EXISTS heading_number text,
					ADD COLUMN IF NOT EXISTS heading_level int,
					ADD COLUMN IF NOT EXISTS parent_heading_number text
				"""
			)
			# simple index to filter by heading
			cur.execute("CREATE INDEX IF NOT EXISTS clauses_heading_idx ON clauses(heading_number)")
			# backfill hashes where missing
			cur.execute(
				"UPDATE clauses SET content_hash = md5(coalesce(content,'')) WHERE content_hash IS NULL"
			)
			# remove duplicates before unique index
			cur.execute(
				"""
				DELETE FROM clauses t
				USING (
				  SELECT id,
						 row_number() OVER (PARTITION BY document_id, section_id, content_hash ORDER BY id) AS rn
				  FROM clauses
				) d
				WHERE t.id=d.id AND d.rn>1
				"""
			)
			# unique index for idempotency
			cur.execute(
				"""
				DO $$
				BEGIN
				  IF NOT EXISTS (
					SELECT 1 FROM pg_indexes
					WHERE schemaname='public' AND indexname='uniq_clause'
				  ) THEN
					EXECUTE 'CREATE UNIQUE INDEX uniq_clause ON clauses(document_id, section_id, content_hash)';
				  END IF;
				END$$;
				"""
			)
			conn.commit()

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

	def _infer_clause_type(self, title: Optional[str], content: str) -> Optional[str]:
		text = ((title or "") + "\n" + (content or "")).lower()
		for k, v in _CLAUSE_MAP.items():
			if k in text:
				return v
		return None

	def _extract_defined_terms(self, content: str) -> List[str]:
		terms: List[str] = []
		terms += re.findall(r'"([A-Z][^"]{1,60})"', content)
		terms += re.findall(r'\b([A-Z]{2,})\b', content)
		# de-dup and cap size
		return list(dict.fromkeys([t.strip() for t in terms if t.strip()]))[:50]

	def index_chunks(self, document: LegalDocument, chunks: List[ProcessedChunk], embeddings: Optional[List[List[float]]] = None):
		"""Insert chunks into clauses table with enrichment. If embeddings provided, store them."""
		self._ensure_optional_schema()
		with self._connect() as conn, conn.cursor() as cur:
			# discover optional columns
			has_seq = self._column_exists(cur, "clauses", "seq")
			has_type = self._column_exists(cur, "clauses", "clause_type")
			has_defs = self._column_exists(cur, "clauses", "defined_terms")
			has_hash = self._column_exists(cur, "clauses", "content_hash")
			has_hn = self._column_exists(cur, "clauses", "heading_number")
			has_hlvl = self._column_exists(cur, "clauses", "heading_level")
			has_parent = self._column_exists(cur, "clauses", "parent_heading_number")
			# build insert columns
			cols = ["document_id","section_id","title","content","tags","embedding"]
			if has_seq: cols.append("seq")
			if has_type: cols.append("clause_type")
			if has_defs: cols.append("defined_terms")
			if has_hash: cols.append("content_hash")
			if has_hn: cols.append("heading_number")
			if has_hlvl: cols.append("heading_level")
			if has_parent: cols.append("parent_heading_number")
			col_list = ", ".join(cols)
			placeholders = ", ".join(["%s"] * len(cols))
			insert_sql = f"INSERT INTO clauses ({col_list}) VALUES ({placeholders})"
			# upsert if we have content_hash
			if has_hash:
				insert_sql += " ON CONFLICT (document_id, section_id, content_hash) DO UPDATE SET title=EXCLUDED.title, content=EXCLUDED.content, tags=EXCLUDED.tags, embedding=EXCLUDED.embedding"
			for i, chunk in enumerate(chunks, start=1):
				# find section title
				section_title = None
				for s in document.sections:
					if s.id == chunk.metadata.section_id:
						section_title = s.title
						break
				emb = None
				if embeddings and (i - 1) < len(embeddings):
					emb = "[" + ",".join(f"{x:.8f}" for x in embeddings[i - 1]) + "]"
				row: List[Any] = [
					document.metadata.document_id,
					chunk.metadata.section_id,
					section_title or chunk.metadata.section_id,
					chunk.content,
					chunk.metadata.tags,
					emb,
				]
				if has_seq:
					row.append(i)
				if has_type:
					row.append(self._infer_clause_type(section_title, chunk.content))
				if has_defs:
					row.append(self._extract_defined_terms(chunk.content))
				if has_hash:
					row.append(hashlib.md5((chunk.content or "").encode("utf-8")).hexdigest())
				if has_hn:
					row.append(getattr(chunk.metadata, 'heading_number', None))
				if has_hlvl:
					row.append(getattr(chunk.metadata, 'heading_level', None))
				if has_parent:
					row.append(getattr(chunk.metadata, 'parent_heading_number', None))
				cur.execute(insert_sql, tuple(row))

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
