from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row

try:
	from sentence_transformers import SentenceTransformer
except Exception as e:
	SentenceTransformer = None  # will error on use if missing


# ---------------------------
# Config / Environment
# ---------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))  # <- default 5432 (Docker)
DB_NAME = os.getenv("DB_NAME", "infra_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "changeme_local_pw")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_MODAL_EMBED = os.getenv("USE_MODAL_EMBED", "0") in {"1", "true", "True"}

# pgvector tuning
IVFFLAT_LISTS = int(os.getenv("IVFFLAT_LISTS", "200"))
IVFFLAT_PROBES = int(os.getenv("IVFFLAT_PROBES", "10"))

# Hybrid RRF param
RRF_K = int(os.getenv("RRF_K", "60"))

# Neighbor/definition expansion
NEIGHBOR_WINDOW = int(os.getenv("NEIGHBOR_WINDOW", "1"))
MAX_DEF_MATCH = int(os.getenv("MAX_DEF_MATCH", "6"))

# Domain synonyms (expand sparse query)
DOMAIN_SYNONYMS = {
	# LNG shipping
	"lng tanker": ["lng carrier", "carrier", "tanker", "vessel", "ship"],
	"shipping cost": ["freight", "demurrage", "charter", "transportation cost"],
	"berth": ["jetty", "loading arm", "flange"],
	# Payment, interest
	"interest rate": ["late payment interest", "default interest", "prime rate", "sofr", "libor"],
	"price adjustment": ["true-up", "purchase price adjustment"],
	"currency": ["fx", "foreign exchange", "conversion", "usd", "eur", "gbp"],
	# Risk/Title/Delivery
	"title": ["title and risk", "risk of loss", "incoterms", "fob", "delivery point", "transfer of title"],
	# Legal boilerplate
	"force majeure": ["act of god", "epidemic", "pandemic", "strikes", "governmental action"],
	"change in law": ["regulatory change", "laws and regulations", "legal change"],
	"indemnity": ["indemnification", "liability", "cap", "basket", "survival"],
	"arbitration": ["seat", "venue", "rules", "icc", "lcia", "uncitral"],
	"governing law": ["law", "jurisdiction"],
	"assignment": ["transfer", "novation"],
	"insurance": ["liability insurance", "coverage", "limits"],
	"guarantee": ["parent guarantee", "performance guarantee", "letter of credit"],
	"termination": ["termination for convenience", "termination for cause", "default"],
	"notices": ["notice", "addresses", "delivery of notice"],
	"audit": ["inspection", "records", "books and records"],
	"tax": ["withholding", "gross-up", "vat", "sales tax"],
	# Power
	"transmission service": ["interconnection", "metering", "wheeling", "ancillary services"],
}

# Internal flags
_MODEL: Optional[SentenceTransformer] = None
_EMBED_DIM: Optional[int] = None
_SCHEMA_READY = False
_FTS_READY = False
_VEC_READY = False
_UNIQ_READY = False


# ---------------------------
# Utilities
# ---------------------------
def _connect():
	return psycopg.connect(
		host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
	)


def _get_model() -> SentenceTransformer:
	global _MODEL, _EMBED_DIM
	if USE_MODAL_EMBED:
		return None  # handled elsewhere
	if _MODEL is None:
		if SentenceTransformer is None:
			raise RuntimeError("sentence-transformers is required; please install it.")
		_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
		try:
			_EMBED_DIM = int(_MODEL.get_sentence_embedding_dimension())
		except Exception:
			_EMBED_DIM = None
	return _MODEL


def _embed_query(query: str) -> List[float]:
	if USE_MODAL_EMBED:
		from adapters.modal_embedding import embed_texts_remote
		return embed_texts_remote([query])[0]
	model = _get_model()
	vec = model.encode([query], normalize_embeddings=True)[0]
	return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def _embed_texts(texts: List[str]) -> List[List[float]]:
	if USE_MODAL_EMBED:
		from adapters.modal_embedding import embed_texts_remote
		return embed_texts_remote(texts)
	model = _get_model()
	arr = model.encode(texts, normalize_embeddings=True)
	return arr.tolist() if hasattr(arr, "tolist") else [list(v) for v in arr]


def _vector_to_sql_literal(vec: List[float]) -> str:
	return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _column_exists(cur, table: str, column: str) -> bool:
	cur.execute(
		"""
		SELECT 1
		FROM information_schema.columns
		WHERE table_name=%s AND column_name=%s
		""",
		(table, column),
	)
	return cur.fetchone() is not None


def _table_has_rows(cur, table: str) -> bool:
	cur.execute(f"SELECT 1 FROM {table} LIMIT 1")
	return cur.fetchone() is not None


# ---------------------------
# Ensures (DDL, indexes, etc.)
# ---------------------------
def _ensure_schema() -> None:
	global _SCHEMA_READY
	if _SCHEMA_READY:
		return
	with _connect() as conn, conn.cursor() as cur:
		# optional columns used by context expansion
		cur.execute(
			"""
			ALTER TABLE IF EXISTS clauses
			  ADD COLUMN IF NOT EXISTS seq int,
			  ADD COLUMN IF NOT EXISTS clause_type text,
			  ADD COLUMN IF NOT EXISTS defined_terms text[]
			"""
		)
		conn.commit()
		_SCHEMA_READY = True


def _ensure_vector_index() -> None:
	"""Ensure cosine op, ivfflat index, and embedding dim match the model (if we can detect it)."""
	global _VEC_READY
	if _VEC_READY:
		return
	with _connect() as conn, conn.cursor() as cur:
		# Ensure pgvector ext
		cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
		conn.commit()

		# Adjust embedding dimension if we know it and table has rows/column exists
		if _EMBED_DIM is None and not USE_MODAL_EMBED:
			_get_model()  # sets _EMBED_DIM

		if _column_exists(cur, "clauses", "embedding") and _EMBED_DIM:
			# Fetch current dim; if mismatched, alter
			cur.execute(
				"""
				SELECT atttypmod
				FROM pg_attribute a
				JOIN pg_class c ON a.attrelid=c.oid
				WHERE c.relname='clauses' AND a.attname='embedding'
				"""
			)
			row = cur.fetchone()
			if row and row[0] not in (None, -1):
				current_dim = row[0] - 4  # pgvector encodes dim as atttypmod-4
				if current_dim != _EMBED_DIM:
					cur.execute(
						f"ALTER TABLE clauses ALTER COLUMN embedding TYPE vector({_EMBED_DIM})"
					)

		# Create IVFFLAT cosine index (use distinct name to avoid clash with existing l2 ops)
		cur.execute(
			f"""
			DO $$
			BEGIN
			  IF NOT EXISTS (
				SELECT 1 FROM pg_indexes
				WHERE schemaname='public' AND indexname='idx_clauses_embedding_cosine_ivfflat'
			  ) THEN
				EXECUTE 'CREATE INDEX idx_clauses_embedding_cosine_ivfflat
					 ON clauses USING ivfflat (embedding vector_cosine_ops) WITH (lists = {IVFFLAT_LISTS})';
			  END IF;
			END$$;
			"""
		)
		conn.commit()
		_VEC_READY = True


def _ensure_fts() -> None:
	"""Weighted tsvector maintained via trigger (title^A + content^B) + unaccent + GIN index."""
	global _FTS_READY
	if _FTS_READY:
		return
	with _connect() as conn, conn.cursor() as cur:
		cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
		# Ensure fts column exists
		cur.execute(
			"""
			ALTER TABLE IF EXISTS clauses
			  ADD COLUMN IF NOT EXISTS fts tsvector
			"""
		)
		# Create or replace trigger function to maintain fts
		cur.execute(
			"""
			CREATE OR REPLACE FUNCTION clauses_fts_update() RETURNS trigger AS $$
			BEGIN
			  NEW.fts :=
			    setweight(to_tsvector('english', unaccent(coalesce(NEW.title,''))), 'A') ||
			    setweight(to_tsvector('english', unaccent(coalesce(NEW.content,''))), 'B');
			  RETURN NEW;
			END
			$$ LANGUAGE plpgsql;
			"""
		)
		# Create trigger if not exists
		cur.execute(
			"""
			DO $$
			BEGIN
			  IF NOT EXISTS (
				SELECT 1 FROM pg_trigger WHERE tgname = 'clauses_fts_trigger'
			  ) THEN
				CREATE TRIGGER clauses_fts_trigger
				BEFORE INSERT OR UPDATE OF title, content ON clauses
				FOR EACH ROW EXECUTE FUNCTION clauses_fts_update();
			  END IF;
			END$$;
			"""
		)
		# Backfill any NULL fts values once
		cur.execute(
			"""
			UPDATE clauses SET fts =
			  setweight(to_tsvector('english', unaccent(coalesce(title,''))), 'A') ||
			  setweight(to_tsvector('english', unaccent(coalesce(content,''))), 'B')
			WHERE fts IS NULL
			"""
		)
		# Index on fts
		cur.execute(
			"CREATE INDEX IF NOT EXISTS clauses_fts_idx ON clauses USING gin(fts)"
		)
		conn.commit()
		_FTS_READY = True


def _ensure_uniqueness_guard() -> None:
	"""Add content_hash + unique index to prevent exact dupes per (document_id, section_id, content_hash)."""
	global _UNIQ_READY
	if _UNIQ_READY:
		return
	with _connect() as conn, conn.cursor() as cur:
		cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
		cur.execute(
			"ALTER TABLE IF EXISTS clauses ADD COLUMN IF NOT EXISTS content_hash text"
		)
		cur.execute(
			"UPDATE clauses SET content_hash = md5(coalesce(content,'')) WHERE content_hash IS NULL"
		)
		# Remove duplicates first (keep smallest id per unique triplet)
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
		# Create unique index if missing
		cur.execute(
			"""
			DO $$
			BEGIN
			  IF NOT EXISTS (
				SELECT 1 FROM pg_indexes
				WHERE schemaname='public' AND indexname='uniq_clause'
			  ) THEN
				EXECUTE 'CREATE UNIQUE INDEX uniq_clause
					 ON clauses(document_id, section_id, content_hash)';
			  END IF;
			END$$;
			"""
		)
		conn.commit()
		_UNIQ_READY = True


def maintenance_compact_and_dedupe() -> Dict[str, Any]:
	"""Optional: call manually to remove duplicate rows and refresh FTS."""
	with _connect() as conn, conn.cursor() as cur:
		# Count dup groups
		cur.execute(
			"""
			SELECT COUNT(*) FROM (
			  SELECT document_id, section_id, COUNT(*)
			  FROM clauses
			  GROUP BY 1,2 HAVING COUNT(*)>1
			) x
			"""
		)
		dup_groups = cur.fetchone()[0]

		# Delete newer duplicates, keep smallest id per (document_id, section_id, content_hash)
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
		# Refresh FTS column fully
		cur.execute(
			"""
			UPDATE clauses SET content_tsv =
			  setweight(to_tsvector('english', unaccent(coalesce(title,''))), 'A') ||
			  setweight(to_tsvector('english', unaccent(coalesce(content,''))), 'B')
			"""
		)
		conn.commit()
	return {"duplicate_groups_before": dup_groups}


# ---------------------------
# Query helpers
# ---------------------------
_QUOTE = re.compile(r'["]')

def _expand_for_fts(query: str) -> str:
	"""Add domain synonyms for sparse side using websearch syntax (OR)."""
	ql = query.lower()
	extras: List[str] = []
	for key, syns in DOMAIN_SYNONYMS.items():
		if any(tok in ql for tok in key.split()):
			for s in syns:
				s = _QUOTE.sub("", s)  # strip quotes
				extras.append(f'"{s}"' if " " in s else s)
	if not extras:
		return query
	return f'{query} OR ' + " OR ".join(sorted(set(extras)))


def _attach_neighbors_and_defs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Best-effort context expansion: ±1 neighbors (requires seq), definitions (requires clause_type/defined_terms)."""
	if not rows:
		return rows
	with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
		have_seq = _column_exists(cur, "clauses", "seq")
		have_ct = _column_exists(cur, "clauses", "clause_type")
		have_defs = _column_exists(cur, "clauses", "defined_terms")

		for r in rows:
			r["neighbors"] = []
			r["definitions"] = []

			if have_seq:
				cur.execute("SELECT seq FROM clauses WHERE id=%s", (r["id"],))
				row = cur.fetchone()
				if row and row["seq"] is not None:
					seq = int(row["seq"])
					cur.execute(
						"""
						SELECT section_id, title, content
						FROM clauses
						WHERE document_id=%s AND seq BETWEEN %s AND %s
						ORDER BY seq
						""",
						(r["document_id"], seq - NEIGHBOR_WINDOW, seq + NEIGHBOR_WINDOW),
					)
					r["neighbors"] = [
						{"section_id": x["section_id"], "title": x["title"], "content": x["content"]}
						for x in cur.fetchall()
						if x is not None
					]

			# naive uppercase-defined-terms fallback if defined_terms col is missing
			candidate_terms: List[str] = []
			if have_defs:
				cur.execute("SELECT defined_terms FROM clauses WHERE id=%s", (r["id"],))
				termrow = cur.fetchone()
				if termrow and termrow["defined_terms"]:
					candidate_terms = termrow["defined_terms"][:MAX_DEF_MATCH]
			else:
				# Extract likely defined terms: Words in Title Case or ALLCAPS in quotes
				caps = re.findall(r"\b([A-Z][A-Za-z0-9\-]{2,})\b", r.get("content", ""))
				candidate_terms = list({t for t in caps if len(t) <= 30})[:MAX_DEF_MATCH]

			if have_ct:
				# fetch Definitions sections in same document that contain any of the terms
				if candidate_terms:
					cur.execute(
						"""
						SELECT section_id, title, content
						FROM clauses
						WHERE document_id=%s
						  AND clause_type='Definitions'
						  AND (
								EXISTS (
								  SELECT 1 FROM unnest(defined_terms) t
								  WHERE t = ANY(%s)
								)
							OR content ILIKE ANY(%s)
						  )
						LIMIT %s
						""",
						(
							r["document_id"],
							candidate_terms,
							["%" + t + "%" for t in candidate_terms],
							MAX_DEF_MATCH,
						),
					)
					r["definitions"] = [
						{"section_id": x["section_id"], "title": x["title"], "content": x["content"]}
						for x in cur.fetchall()
						if x is not None
					]
	return rows


# ---------------------------
# Dense-only search (cosine)
# ---------------------------
def _search_dense(query_vec: List[float], limit: int = 15) -> List[Dict[str, Any]]:
	qvec = _vector_to_sql_literal(query_vec)
	with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
		cur.execute(f"SET LOCAL ivfflat.probes = {int(IVFFLAT_PROBES)}")
		cur.execute(
			"""
			SELECT id, document_id, section_id, title, content,
				   (1.0 - (embedding <-> %s::vector)) AS cos_sim
			FROM clauses
			WHERE embedding IS NOT NULL
			ORDER BY embedding <-> %s::vector
			LIMIT %s
			""",
			(qvec, qvec, limit),
		)
		rows = cur.fetchall()
	# De-dupe by (document_id, section_id)
	seen = set()
	out = []
	for r in rows:
		key = (r["document_id"], r["section_id"])
		if key in seen:
			continue
		seen.add(key)
		out.append(
			{
				"id": r["id"],
				"document_id": r["document_id"],
				"section_id": r["section_id"],
				"title": r["title"],
				"content": r["content"],
				"cos_sim": float(r["cos_sim"]) if r["cos_sim"] is not None else None,
			}
		)
	return out


def search(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
	"""Dense-only (cosine) search."""
	_ensure_schema()
	_ensure_vector_index()
	qvec = _embed_query(query)
	rows = _search_dense(qvec, limit=top_k)
	return _attach_neighbors_and_defs(rows)


# ---------------------------
# Hybrid search (RRF fused)
# ---------------------------
def search_hybrid(
	query: str,
	pool_n: int = 200,
	top_k: int = 20,
	probes: Optional[int] = None,
	rrf_k: Optional[int] = None,
	expand_sparse: bool = True,
) -> List[Dict[str, Any]]:
	"""Cosine ANN + weighted FTS, fused by RRF. Returns rows with rrf score + context."""
	_ensure_schema()
	_ensure_vector_index()
	_ensure_fts()
	_ensure_uniqueness_guard()

	qvec = _embed_query(query)
	qvec_lit = _vector_to_sql_literal(qvec)
	q_fts = _expand_for_fts(query) if expand_sparse else query
	probes = probes or IVFFLAT_PROBES
	rrf_k = rrf_k or RRF_K

	with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
		cur.execute(f"SET LOCAL ivfflat.probes = {int(probes)}")
		cur.execute(
			"""
			WITH params AS (
			  SELECT %s::vector AS qvec,
					 websearch_to_tsquery('english', %s) AS q,
					 %s::int AS pool_n,
					 %s::int AS k
			),
			dense AS (
			  SELECT id,
					 row_number() OVER (ORDER BY embedding <-> (SELECT qvec FROM params)) AS r_dense
			  FROM clauses
			  WHERE embedding IS NOT NULL
			  ORDER BY embedding <-> (SELECT qvec FROM params)
			  LIMIT (SELECT pool_n FROM params)
			),
			sparse AS (
			  SELECT id,
					 row_number() OVER (
					   ORDER BY ts_rank_cd(fts, (SELECT q FROM params), 32) DESC
					 ) AS r_sparse
			  FROM clauses
			  WHERE fts @@ (SELECT q FROM params)
			  ORDER BY ts_rank_cd(fts, (SELECT q FROM params), 32) DESC
			  LIMIT (SELECT pool_n FROM params)
			),
			unioned AS (
			  SELECT id, r_dense, NULL::int AS r_sparse FROM dense
			  UNION ALL
			  SELECT id, NULL::int, r_sparse FROM sparse
			),
			agg AS (
			  SELECT id, MIN(r_dense) AS r_dense, MIN(r_sparse) AS r_sparse
			  FROM unioned GROUP BY id
			),
			fused AS (
			  SELECT id,
					 COALESCE(1.0/((SELECT k FROM params) + r_dense), 0.0) +
					 COALESCE(1.0/((SELECT k FROM params) + r_sparse), 0.0) AS rrf
			  FROM agg
			)
			SELECT c.id, c.document_id, c.section_id, c.title, c.content, f.rrf
			FROM fused f
			JOIN clauses c ON c.id=f.id
			ORDER BY f.rrf DESC
			LIMIT %s
			""",
			(qvec_lit, q_fts, pool_n, rrf_k, top_k),
		)
		rows = cur.fetchall()

	# De-dupe by (document_id, section_id)
	seen = set()
	uniq = []
	for r in rows:
		key = (r["document_id"], r["section_id"])
		if key in seen:
			continue
		seen.add(key)
		uniq.append(
			{
				"id": r["id"],
				"document_id": r["document_id"],
				"section_id": r["section_id"],
				"title": r["title"],
				"content": r["content"],
				"rrf": float(r["rrf"]),
			}
		)

	return _attach_neighbors_and_defs(uniq)


# ---------------------------
# Ingestion helpers (unchanged surface)
# ---------------------------
def ingest_data_dir(data_dir: str = "./data") -> Dict[str, Any]:
	"""Parse XML/PDF files, chunk, embed, and index into Postgres. Returns counts & stats."""
	from pathlib import Path
	from src.ingestion import DocumentIngestionPipeline
	from src.indexing import PgIndexer

	_ensure_schema()
	_ensure_vector_index()
	_ensure_fts()
	_ensure_uniqueness_guard()

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
		embs = _embed_texts(texts)
		indexer.upsert_document(doc)
		indexer.index_chunks(doc, chunks, embs)
		num_docs += 1
		num_chunks += len(chunks)

	return {"documents_indexed": num_docs, "chunks_indexed": num_chunks}


# ---------------------------
# High-level API
# ---------------------------
def retrieve(
	query: str,
	top_k: int = 20,
	pool_n: int = 200,
	use_hybrid: bool = True,
	expand_sparse: bool = True,
) -> List[Dict[str, Any]]:
	"""End-to-end retrieval: hybrid by default; dense-only if use_hybrid=False."""
	if use_hybrid:
		return search_hybrid(
			query=query, pool_n=pool_n, top_k=top_k, expand_sparse=expand_sparse
		)
	qvec = _embed_query(query)
	rows = _search_dense(qvec, limit=top_k)
	return _attach_neighbors_and_defs(rows)


# ---------------------------
# Clause graph schema & population
# ---------------------------

def _ensure_graph_schema() -> None:
	with _connect() as conn, conn.cursor() as cur:
		# Add heading_number column for numeric section identifiers
		cur.execute(
			"""
			ALTER TABLE IF EXISTS clauses
			  ADD COLUMN IF NOT EXISTS heading_number text
			"""
		)
		# Graph table
		cur.execute(
			"""
			CREATE TABLE IF NOT EXISTS clause_edges (
			  src_clause_id bigint REFERENCES clauses(id),
			  dst_clause_id bigint REFERENCES clauses(id),
			  relation text CHECK (relation IN ('refers_to','defines','adjacent','same_type')),
			  weight float DEFAULT 1.0
			)
			"""
		)
		# Unique + lookup indexes
		cur.execute(
			"""
			CREATE UNIQUE INDEX IF NOT EXISTS clause_edges_unique
			ON clause_edges(src_clause_id, dst_clause_id, relation)
			"""
		)
		cur.execute(
			"""
			CREATE INDEX IF NOT EXISTS clause_edges_src
			ON clause_edges(src_clause_id, relation)
			"""
		)
		conn.commit()


def _backfill_heading_numbers(document_id: Optional[str] = None) -> None:
	"""Derive heading_number from section_id or title heuristics for fast mapping."""
	with _connect() as conn, conn.cursor() as cur:
		params = []
		where = ""
		if document_id:
			where = "WHERE document_id = %s"
			params.append(document_id)
		# Prefer numeric section_id; else parse leading number tokens from title
		cur.execute(
			f"""
			UPDATE clauses
			SET heading_number = COALESCE(
				NULLIF(regexp_replace(section_id, '[^0-9\.]', '', 'g'), ''),
				substring(title from '(^[0-9]+(?:\.[0-9]+)*)')
			)
			{where}
			""",
			tuple(params),
		)
		conn.commit()


def _populate_adjacent_edges(document_id: Optional[str] = None) -> int:
	with _connect() as conn, conn.cursor() as cur:
		params = []
		where = ""
		if document_id:
			where = "WHERE document_id = %s"
			params.append(document_id)
		# Use ID order per document as a proxy sequence
		cur.execute(
			f"""
			WITH ordered AS (
			  SELECT id, document_id,
					lag(id) OVER (PARTITION BY document_id ORDER BY id) AS prev_id,
					lead(id) OVER (PARTITION BY document_id ORDER BY id) AS next_id
			  FROM clauses
			  {where}
			)
			INSERT INTO clause_edges (src_clause_id, dst_clause_id, relation, weight)
			SELECT id, prev_id, 'adjacent', 1.0 FROM ordered WHERE prev_id IS NOT NULL
			ON CONFLICT DO NOTHING;
			""",
			tuple(params),
		)
		cur.execute(
			f"""
			WITH ordered AS (
			  SELECT id, document_id,
					lag(id) OVER (PARTITION BY document_id ORDER BY id) AS prev_id,
					lead(id) OVER (PARTITION BY document_id ORDER BY id) AS next_id
			  FROM clauses
			  {where}
			)
			INSERT INTO clause_edges (src_clause_id, dst_clause_id, relation, weight)
			SELECT id, next_id, 'adjacent', 1.0 FROM ordered WHERE next_id IS NOT NULL
			ON CONFLICT DO NOTHING;
			""",
			tuple(params),
		)
		conn.commit()
		# Can't easily return rows inserted; ignore
		return 0


def _populate_refers_to_edges(document_id: Optional[str] = None) -> None:
	"""Scan content for references like 'Section 6.2' or 'Clause 14.1' and link within same document."""
	_backfill_heading_numbers(document_id)
	with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
		doc_clause_map: Dict[str, Dict[str, int]] = {}
		# Build mapping document_id -> {heading_number: id}
		params = []
		where = ""
		if document_id:
			where = "WHERE document_id = %s"
			params.append(document_id)
		cur.execute(f"SELECT id, document_id, heading_number FROM clauses {where}", tuple(params))
		for row in cur.fetchall():
			hn = row["heading_number"]
			if not hn:
				continue
			doc_clause_map.setdefault(row["document_id"], {})[hn] = row["id"]
		# Scan contents for refs
		cur.execute(f"SELECT id, document_id, content FROM clauses {where}", tuple(params))
		rows = cur.fetchall()
		ref_re = re.compile(r"\b(?:Section|Clause|§|¶)?\s*([0-9]+(?:\.[0-9]+)*)")
		for r in rows:
			text = r.get("content") or ""
			candidates = set(m.group(1) for m in ref_re.finditer(text))
			if not candidates:
				continue
			mapping = doc_clause_map.get(r["document_id"], {})
			for num in candidates:
				dst = mapping.get(num)
				if dst:
					cur.execute(
						"INSERT INTO clause_edges (src_clause_id, dst_clause_id, relation, weight) VALUES (%s,%s,'refers_to',1.0) ON CONFLICT DO NOTHING",
						(r["id"], dst),
					)
		conn.commit()


def _populate_defines_edges(document_id: Optional[str] = None) -> None:
	"""Link from Definitions clauses to clauses using those terms."""
	with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
		params = []
		where = ""
		if document_id:
			where = "WHERE document_id = %s"
			params.append(document_id)
		# Get candidate definition clauses by title heuristic
		cur.execute(
			f"SELECT id, document_id, title, content FROM clauses {where}",
			tuple(params),
		)
		defs_by_doc: Dict[str, List[Dict[str, Any]]] = {}
		for row in cur.fetchall():
			title = (row.get("title") or "").lower()
			if "definition" in title:
				defs_by_doc.setdefault(row["document_id"], []).append(row)
		# Build rows per doc to scan
		cur.execute(
			f"SELECT id, document_id, content FROM clauses {where}",
			tuple(params),
		)
		all_rows = cur.fetchall()
		for doc, def_rows in defs_by_doc.items():
			terms: List[str] = []
			for d in def_rows:
				text = d.get("content") or ""
				# "Term" means ... or Title Case terms
				terms += re.findall(r'"([^\"]{2,40})"\s*(?:means|shall\s+mean)', text, flags=re.IGNORECASE)
			# de-dup and limit
			terms = [t.strip() for t in terms if t.strip()]
			terms = list(dict.fromkeys(terms))[:128]
			if not terms:
				continue
			# link to any clause containing term
			like_list = [f"%{t}%" for t in terms]
			for row in all_rows:
				if row["document_id"] != doc:
					continue
				content = row.get("content") or ""
				if any(t in content for t in terms):
					for d in def_rows:
						cur.execute(
							"INSERT INTO clause_edges (src_clause_id, dst_clause_id, relation, weight) VALUES (%s,%s,'defines',1.0) ON CONFLICT DO NOTHING",
							(d["id"], row["id"]),
						)
		conn.commit()


def populate_clause_graph(document_id: Optional[str] = None) -> None:
	"""Public helper: ensure schema and populate adjacent/refers_to/defines edges."""
	_ensure_graph_schema()
	_backfill_heading_numbers(document_id)
	_populate_adjacent_edges(document_id)
	_populate_refers_to_edges(document_id)
	_populate_defines_edges(document_id)


def _fetch_edges(src_ids: List[int], relation_in: Tuple[str, ...] = ("refers_to","defines")) -> List[int]:
	if not src_ids:
		return []
	with _connect() as conn, conn.cursor() as cur:
		cur.execute(
			"SELECT DISTINCT dst_clause_id FROM clause_edges WHERE src_clause_id = ANY(%s) AND relation = ANY(%s)",
			(src_ids, list(relation_in)),
		)
		rows = cur.fetchall()
	return [r[0] for r in rows if r and r[0] is not None]


def _fetch_clause_rows(ids: List[int]) -> List[Dict[str, Any]]:
	if not ids:
		return []
	with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
		cur.execute(
			"SELECT id, document_id, section_id, title, content FROM clauses WHERE id = ANY(%s)",
			(ids,),
		)
		return [dict(r) for r in cur.fetchall()]


def retrieve_recursive(query: str, k: int = 8, max_hops: int = 2) -> List[Dict[str, Any]]:
	"""Hybrid search seeds + follow refers_to/defines edges up to max_hops."""
	_ensure_schema()
	_ensure_vector_index()
	_ensure_fts()
	_ensure_graph_schema()
	seeds = search_hybrid(query, pool_n=200, top_k=k)
	seen = {int(r["id"]) for r in seeds}
	frontier = [int(r["id"]) for r in seeds]
	hops = 0
	out = list(seeds)
	while frontier and hops < max_hops:
		dsts = _fetch_edges(frontier, relation_in=("refers_to","defines"))
		new_ids = [i for i in dsts if i not in seen]
		if not new_ids:
			break
		ctx = _fetch_clause_rows(new_ids)
		out.extend(ctx)
		for i in new_ids:
			seen.add(i)
		frontier = new_ids
		hops += 1
	return out 