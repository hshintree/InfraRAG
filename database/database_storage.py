import os
from typing import Any, Dict, List, Optional

import psycopg


class DatabaseStorage:
	"""Thin Postgres wrapper for documents and clause chunks using pgvector."""

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

	def healthcheck(self) -> bool:
		try:
			with self._connect() as conn, conn.cursor() as cur:
				cur.execute("SELECT 1")
				cur.fetchone()
			return True
		except Exception:
			return False

	def upsert_document(self, document: Dict[str, Any]) -> None:
		"""Upsert a document row given a dict matching src.schema.DocumentMetadata."""
		sql = (
			"INSERT INTO documents (document_id, title, document_type, jurisdiction, governing_law, industry) "
			"VALUES (%s, %s, %s, %s, %s, %s) "
			"ON CONFLICT (document_id) DO UPDATE SET "
			"title=EXCLUDED.title, document_type=EXCLUDED.document_type, jurisdiction=EXCLUDED.jurisdiction, "
			"governing_law=EXCLUDED.governing_law, industry=EXCLUDED.industry"
		)
		meta = document
		with self._connect() as conn, conn.cursor() as cur:
			cur.execute(
				sql,
				(
					meta["document_id"],
					meta.get("title"),
					meta.get("document_type"),
					meta.get("jurisdiction"),
					meta.get("governing_law"),
					meta.get("industry"),
				),
			)

	def insert_clause(self, document_id: str, section_id: str, title: Optional[str], content: str, tags: List[str], embedding: Optional[List[float]]) -> None:
		"""Insert a single clause row."""
		insert_sql = (
			"INSERT INTO clauses (document_id, section_id, title, content, tags, embedding) "
			"VALUES (%s, %s, %s, %s, %s, %s)"
		)
		emb_literal = None
		if embedding is not None:
			emb_literal = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"
		with self._connect() as conn, conn.cursor() as cur:
			cur.execute(insert_sql, (document_id, section_id, title or section_id, content, tags, emb_literal))

	def search(self, query_vector: List[float], limit: int = 15) -> List[Dict[str, Any]]:
		qvec = "[" + ",".join(f"{x:.8f}" for x in query_vector) + "]"
		sql = (
			"SELECT document_id, section_id, title, content, 1.0 / (1.0 + (embedding <-> %s::vector)) AS score "
			"FROM clauses WHERE embedding IS NOT NULL ORDER BY embedding <-> %s::vector LIMIT %s"
		)
		with self._connect() as conn, conn.cursor() as cur:
			cur.execute(sql, (qvec, qvec, limit))
			rows = cur.fetchall()
		return [
			{"document_id": r[0], "section_id": r[1], "title": r[2], "content": r[3], "score": float(r[4])}
			for r in rows
		] 