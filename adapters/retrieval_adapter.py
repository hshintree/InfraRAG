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
DB_PORT = int(os.getenv("DB_PORT", "5433"))  # <- default 5433 (Docker)
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
_SECT_READY = False


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
		_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
		try:
			_EMBED_DIM = int(_MODEL.get_sentence_embedding_dimension())
		except Exception:
			_EMBED_DIM = None
	return _MODEL


def _embed_query(query: str) -> List[float]:
	if USE_MODAL_EMBED:
		from adapters.modal_embedding import embed_texts_remote
		return embed_texts_remote([query])[0]
	try:
		model = _get_model()
		vec = model.encode([query], normalize_embeddings=True)[0]
		return vec.tolist() if hasattr(vec, "tolist") else list(vec)
	except Exception:
		# Fallback to remote embeddings if local model fails (e.g., HF rate limit or torch issue)
		from adapters.modal_embedding import embed_texts_remote
		return embed_texts_remote([query])[0]


def _embed_texts(texts: List[str]) -> List[List[float]]:
	if USE_MODAL_EMBED:
		from adapters.modal_embedding import embed_texts_remote
		return embed_texts_remote(texts)
	try:
		model = _get_model()
		arr = model.encode(texts, normalize_embeddings=True)
		return arr.tolist() if hasattr(arr, "tolist") else [list(v) for v in arr]
	except Exception:
		from adapters.modal_embedding import embed_texts_remote
		return embed_texts_remote(texts)


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
	"""Weighted tsvector maintained via trigger (title^A + content^B) + unaccent + GIN index.
	Also maintain a parallel simple-dict FTS (keeps numbers) for numeric phrases.
	"""
	global _FTS_READY
	if _FTS_READY:
		return
	with _connect() as conn, conn.cursor() as cur:
		cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
		# Ensure columns exist
		cur.execute(
			"""
			ALTER TABLE IF EXISTS clauses
			  ADD COLUMN IF NOT EXISTS fts tsvector,
			  ADD COLUMN IF NOT EXISTS fts_simple tsvector
			"""
		)
		# Trigger function maintains both fts columns
		cur.execute(
			"""
			CREATE OR REPLACE FUNCTION clauses_fts_update() RETURNS trigger AS $$
			BEGIN
			  NEW.fts :=
			    setweight(to_tsvector('english', unaccent(coalesce(NEW.title,''))), 'A') ||
			    setweight(to_tsvector('english', unaccent(coalesce(NEW.content,''))), 'B');
			  NEW.fts_simple := to_tsvector('simple', unaccent(coalesce(NEW.title,'') || ' ' || coalesce(NEW.content,'')));
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
			UPDATE clauses SET 
			  fts = setweight(to_tsvector('english', unaccent(coalesce(title,''))), 'A') ||
			        setweight(to_tsvector('english', unaccent(coalesce(content,''))), 'B'),
			  fts_simple = to_tsvector('simple', unaccent(coalesce(title,'') || ' ' || coalesce(content,'')))
			WHERE fts IS NULL OR fts_simple IS NULL
			"""
		)
		# Indexes
		cur.execute("CREATE INDEX IF NOT EXISTS clauses_fts_idx ON clauses USING gin(fts)")
		cur.execute("CREATE INDEX IF NOT EXISTS clauses_fts_simple_idx ON clauses USING gin(fts_simple)")
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


# ---------------------------
# Query helpers
# ---------------------------
_QUOTE = re.compile(r'["]')

# New token/constraint helpers for MUST/NOT/SHOULD
from typing import Any as _Any  # alias to avoid confusion in helper signatures

def _expand_token_variants(tok: str) -> List[str]:
	"""Produce simple ILIKE-friendly variants for hyphens/spaces and a few domain terms."""
	t = (tok or "").strip()
	if not t:
		return []
	v: set[str] = {t}
	# normalize hyphens/spaces
	hy = t.replace("–", "-").replace("—", "-")
	sp = hy.replace("-", " ")
	hy2 = sp.replace(" ", "-")
	v.update({hy, sp, hy2, t.lower(), sp.lower(), hy2.lower()})
	# domain tweaks
	tl = t.lower()
	if tl == "pre-award":
		v.update(["pre award", "preaward"])
	if tl == "arbitral award":
		v.add("arbitration award")
	if tl in {"360 day", "360-day", "360 days", "360-day basis"}:
		v.update(["360 day", "360-day", "360 day basis", "360-day basis", "360 day year", "360-day year"])
	# de-dup preserving first occurrence (case-insensitive)
	out: List[str] = []
	seen: set[str] = set()
	for x in v:
		xl = x.lower()
		if xl in seen:
			continue
		seen.add(xl)
		out.append(x)
	return out


def _build_must_sql(alias: str, tokens: Optional[List[str]], params: List[_Any]) -> str:
	"""Build AND-of( OR-of(ILIKE variants) ) predicate for must tokens; appends params."""
	toks = tokens or []
	groups: List[str] = []
	for tok in toks:
		vars_ = _expand_token_variants(tok)
		if not vars_:
			continue
		ors: List[str] = []
		for v in vars_:
			ors.append(f"{alias}.content ILIKE %s")
			params.append(f"%{v}%")
		groups.append("(" + " OR ".join(ors) + ")")
	return "TRUE" if not groups else "(" + " AND ".join(groups) + ")"


def _build_must_not_sql(alias: str, tokens: Optional[List[str]], params: List[_Any]) -> str:
	"""Build NOT( OR-of(ILIKE variants) ) predicate for must_not tokens; appends params."""
	toks = tokens or []
	ors: List[str] = []
	for tok in toks:
		for v in _expand_token_variants(tok):
			ors.append(f"{alias}.content ILIKE %s")
			params.append(f"%{v}%")
	return "TRUE" if not ors else "(NOT (" + " OR ".join(ors) + "))"


def _expand_should_patterns(tokens: Optional[List[str]]) -> List[str]:
	"""Expand should tokens; include hyphen/space variants and a light LIBOR→SOFR nudge."""
	toks = tokens or []
	out: List[str] = []
	for t in toks:
		out.extend(_expand_token_variants(t))
		if t.strip().lower() == "libor":
			out.extend(["SOFR"])  # optional modern synonym
	# de-dup case-insensitively, keep first occurrence
	seen: set[str] = set(); res: List[str] = []
	for x in out:
		xl = x.lower()
		if xl in seen:
			continue
		seen.add(xl); res.append(x)
	return res


def _build_should_score_sql(alias: str, patterns: List[str], params: List[_Any]) -> str:
	"""Returns expression averaging CASE(ILIKE) over should patterns; appends params."""
	if not patterns:
		return "0.0"
	terms: List[str] = []
	for p in patterns:
		params.append(f"%{p}%")
		terms.append(f"CASE WHEN {alias}.content ILIKE %s THEN 1.0 ELSE 0.0 END")
	return "(" + " + ".join(terms) + f") / {len(patterns)}"


# Column-aware variants for MUST/NOT/SHOULD on arbitrary text columns
def _build_must_sql_on(alias: str, column: str, tokens: Optional[List[str]], params: List[_Any]) -> str:
	"""AND-of(OR-of(ILIKE variants)) on alias.column; appends params in order."""
	toks = tokens or []
	groups: List[str] = []
	for tok in toks:
		vars_ = _expand_token_variants(tok)
		if not vars_:
			continue
		ors: List[str] = []
		for v in vars_:
			ors.append(f"{alias}.{column} ILIKE %s")
			params.append(f"%{v}%")
		groups.append("(" + " OR ".join(ors) + ")")
	return "TRUE" if not groups else "(" + " AND ".join(groups) + ")"


def _build_must_not_sql_on(alias: str, column: str, tokens: Optional[List[str]], params: List[_Any]) -> str:
	"""NOT(OR-of(ILIKE variants)) on alias.column; appends params."""
	toks = tokens or []
	ors: List[str] = []
	for tok in toks:
		for v in _expand_token_variants(tok):
			ors.append(f"{alias}.{column} ILIKE %s")
			params.append(f"%{v}%")
	return "TRUE" if not ors else "(NOT (" + " OR ".join(ors) + "))"


def _build_should_score_sql_on(alias: str, column: str, patterns: List[str], params: List[_Any]) -> str:
	"""AVG(CASE ILIKE) score over patterns on alias.column; appends params."""
	if not patterns:
		return "0.0"
	terms: List[str] = []
	for p in patterns:
		params.append(f"%{p}%")
		terms.append(f"CASE WHEN {alias}.{column} ILIKE %s THEN 1.0 ELSE 0.0 END")
	return "(" + " + ".join(terms) + f") / {len(patterns)}"


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


def _ensure_perf_indexes() -> None:
	"""Ensure trigram and common lookup indexes exist for faster ILIKE and filters."""
	with _connect() as conn, conn.cursor() as cur:
		# Extensions
		cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
		# Trigram GIN on large text columns used with ILIKE
		cur.execute("CREATE INDEX IF NOT EXISTS clauses_content_trgm_idx ON clauses USING gin (content gin_trgm_ops)")
		cur.execute("CREATE INDEX IF NOT EXISTS clauses_title_trgm_idx   ON clauses USING gin (title   gin_trgm_ops)")
		# Cheap filters
		cur.execute("CREATE INDEX IF NOT EXISTS clauses_heading_idx ON clauses(heading_number)")
		cur.execute("CREATE INDEX IF NOT EXISTS clauses_doc_idx     ON clauses(document_id)")
		cur.execute("CREATE INDEX IF NOT EXISTS clauses_clause_type_idx ON clauses(clause_type)")
		# Documents metadata
		cur.execute("CREATE INDEX IF NOT EXISTS documents_law_idx      ON documents(governing_law)")
		cur.execute("CREATE INDEX IF NOT EXISTS documents_industry_idx ON documents(industry)")
		cur.execute("CREATE INDEX IF NOT EXISTS documents_type_idx     ON documents(document_type)")
		# Section blobs existence gate (indexes created in section schema ensure)
		cur.execute(
			"""
			DO $$
			BEGIN
			  IF EXISTS (
				SELECT 1 FROM information_schema.tables
				WHERE table_name='section_blobs'
			  ) THEN
				PERFORM 1;
			  END IF;
			END$$;
			"""
		)
		conn.commit()


def _attach_neighbors_and_defs(rows: List[Dict[str, Any]], neighbor_window: Optional[int] = None, max_def_match: Optional[int] = None) -> List[Dict[str, Any]]:
	"""Best-effort context expansion batched: neighbors by doc/seq ranges; definitions per doc."""
	if not rows:
		return rows
	window = NEIGHBOR_WINDOW if neighbor_window is None else int(neighbor_window)
	def_cap = MAX_DEF_MATCH if max_def_match is None else int(max_def_match)
	ids = [int(r["id"]) for r in rows if r.get("id") is not None]

	with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
		have_seq = _column_exists(cur, "clauses", "seq")
		have_ct = _column_exists(cur, "clauses", "clause_type")
		have_defs = _column_exists(cur, "clauses", "defined_terms")

		for r in rows:
			r["neighbors"] = []
			r["definitions"] = []

		if not have_seq and not have_ct:
			return rows

		# 1) fetch seq/doc for all rows in one shot
		seq_map: Dict[int, Optional[int]] = {}
		doc_map: Dict[int, str] = {}
		if have_seq and ids:
			cur.execute("SELECT id, document_id, seq FROM clauses WHERE id = ANY(%s)", (ids,))
			for x in cur.fetchall():
				seq_map[int(x["id"])] = None if x["seq"] is None else int(x["seq"])
				doc_map[int(x["id"])] = x["document_id"]

		# 2) neighbors: batch by (document_id, min_seq..max_seq)
		if have_seq and window > 0:
			by_doc: Dict[str, List[int]] = {}
			for r in rows:
				sid = int(r["id"]) if r.get("id") is not None else None
				if sid is None:
					continue
				if sid in doc_map and seq_map.get(sid) is not None:
					by_doc.setdefault(doc_map[sid], []).append(seq_map[sid])
			# one query per doc
			for doc_id, seqs in by_doc.items():
				lo = min(seqs) - window
				hi = max(seqs) + window
				cur.execute(
					"""
					SELECT seq, section_id, title, content
					FROM clauses
					WHERE document_id=%s AND seq BETWEEN %s AND %s
					ORDER BY seq
					""",
					(doc_id, lo, hi),
				)
				neighbors_all = cur.fetchall()
				by_seq: Dict[int, Dict[str, Any]] = {}
				for n in neighbors_all:
					s = int(n["seq"]) if n["seq"] is not None else None
					if s is not None:
						by_seq[s] = {"section_id": n["section_id"], "title": n["title"], "content": n["content"]}
				# attach per row
				for r in rows:
					sid = int(r["id"]) if r.get("id") is not None else None
					if sid is None or doc_map.get(sid) != doc_id:
						continue
					s = seq_map.get(sid)
					if s is None:
						continue
					r["neighbors"] = [by_seq[s2] for s2 in range(s - window, s + window + 1) if s2 in by_seq]

		# 3) definitions: batch term collection then one pass per doc
		if have_ct:
			terms_by_doc: Dict[str, List[str]] = {}
			if have_defs and ids:
				cur.execute("SELECT id, document_id, defined_terms FROM clauses WHERE id = ANY(%s)", (ids,))
				for x in cur.fetchall():
					if not x["defined_terms"]:
						continue
					did = x["document_id"]
					terms_by_doc.setdefault(did, [])
					terms_by_doc[did].extend(list(x["defined_terms"]))
			else:
				for r in rows:
					text = r.get("content") or ""
					caps = re.findall(r"\b([A-Z][A-Za-z0-9\-]{2,})\b", text)
					if caps:
						terms_by_doc.setdefault(r["document_id"], []).extend(caps)
			# Ensure every doc in the result set is processed (enables pure fallback)
			docs_in_rows = {r["document_id"] for r in rows if r.get("document_id")}
			for d in docs_in_rows:
				terms_by_doc.setdefault(d, [])
			# de-dupe and cap
			for d in list(terms_by_doc):
				dedup = list(dict.fromkeys(t.strip() for t in terms_by_doc[d] if t.strip()))
				terms_by_doc[d] = dedup[:def_cap]
			# fetch matching Definitions once per doc and attach; if none, fall back to leading Definitions
			for doc_id, terms in terms_by_doc.items():
				defs: List[Dict[str, Any]] = []
				if terms:
					like_list = ["%" + t + "%" for t in terms]
					cur.execute(
						"""
						SELECT section_id, title, content
						FROM clauses
						WHERE document_id=%s
						  AND clause_type='Definitions'
						  AND (content ILIKE ANY(%s)
						       OR EXISTS (SELECT 1 FROM unnest(defined_terms) t WHERE t = ANY(%s)))
						""",
						(doc_id, like_list, terms),
					)
					defs = [dict(x) for x in cur.fetchall()]
				if not defs:
					cur.execute(
						"""
						SELECT section_id, title, content
						FROM clauses
						WHERE document_id=%s
						  AND clause_type='Definitions'
						ORDER BY COALESCE(seq, 1<<30) ASC
						LIMIT %s
						""",
						(doc_id, def_cap),
					)
					defs = [dict(x) for x in cur.fetchall()]
				for r in rows:
					if r["document_id"] == doc_id:
						r["definitions"] = defs[:def_cap]
	return rows


# ---------------------------
# Section blobs schema/build
# ---------------------------
def _ensure_section_blobs_schema() -> None:
	"""Create a real table (not MV) for aggregated sections + embeddings."""
	global _SECT_READY, _EMBED_DIM
	if _SECT_READY:
		return
	with _connect() as conn, conn.cursor() as cur:
		# table
		cur.execute(
			"""
			CREATE TABLE IF NOT EXISTS section_blobs (
			  id BIGSERIAL PRIMARY KEY,
			  document_id text NOT NULL,
			  heading_number text NOT NULL,
			  title text,
			  section_text text,
			  embedding vector
			)
			"""
		)
		# heading filter / lookup
		cur.execute("CREATE INDEX IF NOT EXISTS section_blobs_doc_head_idx ON section_blobs(document_id, heading_number)")
		cur.execute("CREATE INDEX IF NOT EXISTS section_blobs_head_like_idx ON section_blobs (heading_number text_pattern_ops)")
		# FTS (expression index). Must match query expression exactly.
		cur.execute("CREATE INDEX IF NOT EXISTS section_blobs_fts_idx ON section_blobs USING gin (to_tsvector('english', section_text))")
		# Vector extension + index with cosine
		cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
		# Avoid type ALTER during parallel searches; assume set by maintenance/rebuild
		# Cosine IVFFLAT index (idempotent create)
		cur.execute(
			f"""
			DO $$
			BEGIN
			  IF NOT EXISTS (
				SELECT 1 FROM pg_indexes
				WHERE schemaname='public' AND indexname='idx_section_blobs_embedding_cosine_ivfflat'
			  ) THEN
				EXECUTE 'CREATE INDEX idx_section_blobs_embedding_cosine_ivfflat
					 ON section_blobs USING ivfflat (embedding vector_cosine_ops) WITH (lists = {IVFFLAT_LISTS})';
			  END IF;
			END$$;
			"""
		)
		conn.commit()
	_SECT_READY = True


def _rebuild_section_blobs(reembed: bool = True, batch: int = 64) -> None:
	"""Rebuild the section table by aggregating clauses. Safe to run repeatedly."""
	_ensure_section_blobs_schema()
	texts: List[Tuple[str, str, Optional[str], str]] = []
	with _connect() as conn, conn.cursor() as cur:
		# Aggregate from clauses
		cur.execute(
			"""
			WITH agg AS (
			  SELECT
				document_id,
				heading_number,
				MIN(title) AS title,
				string_agg(content, E'\n\n' ORDER BY seq) AS section_text
			  FROM clauses
			  WHERE heading_number IS NOT NULL
			  GROUP BY document_id, heading_number
			)
			SELECT document_id, heading_number, title, section_text
			FROM agg
			"""
		)
		rows = cur.fetchall()
		# Start fresh
		cur.execute("TRUNCATE section_blobs")
		conn.commit()
		for r in rows:
			texts.append((r[0], r[1], r[2], r[3]))

		# Insert in batches, with embeddings
		for i in range(0, len(texts), batch):
			chunk = texts[i:i+batch]
			section_texts = [t[3] or "" for t in chunk]
			embs: List[List[float]] = [[] for _ in chunk]
			if reembed:
				embs = _embed_texts(section_texts)
			# Prepare literals
			emb_lits = [(_vector_to_sql_literal(e) if e else None) for e in embs]
			args = []
			for (doc, head, title, text), el in zip(chunk, emb_lits):
				args.append((doc, head, title, text, el))
			with conn.cursor() as cur2:
				cur2.executemany(
					"INSERT INTO section_blobs (document_id, heading_number, title, section_text, embedding) "
					"VALUES (%s, %s, %s, %s, %s::vector)",
					args
				)
			conn.commit()


def _ensure_section_blobs_ready() -> None:
	"""One-time bootstrap: if table empty, build it."""
	_ensure_section_blobs_schema()
	with _connect() as conn, conn.cursor() as cur:
		cur.execute("SELECT COUNT(*) FROM section_blobs")
		n = cur.fetchone()[0]
	if int(n) == 0:
		_rebuild_section_blobs(reembed=True)


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
	sparse_weight: float = 0.6,
	must_tokens: Optional[List[str]] = None,
	must_not_tokens: Optional[List[str]] = None,
	should_tokens: Optional[List[str]] = None,
	should_weight: float = 0.05,
	heading_like: Optional[str] = None,
	filter_doc_type: Optional[str] = None,
	filter_industry: Optional[str] = None,
	filter_law: Optional[str] = None,
	filter_seat: Optional[str] = None,
	filter_doc_id: Optional[str] = None,
	prefer_doc: Optional[str] = None,
	enforce_constraints: bool = False,
	neighbor_window: Optional[int] = None,
	max_defs: Optional[int] = None,
	explain: bool = False,
	per_heading_cap: int = 1,
) -> List[Dict[str, Any]]:
	"""Cosine ANN + weighted FTS for both clauses and section_blobs, with proper MUST/NOT for both."""
	_ensure_schema()
	_ensure_vector_index()
	_ensure_fts()
	_ensure_uniqueness_guard()
	_ensure_perf_indexes()
	_ensure_section_blobs_ready()

	qvec = _embed_query(query)
	qvec_lit = _vector_to_sql_literal(qvec)
	q_fts = _expand_for_fts(query) if expand_sparse else query
	probes = probes or IVFFLAT_PROBES
	rrf_k = rrf_k or RRF_K

	# Build dynamic MUST / MUST_NOT SQL & params
	dyn_params: List[Any] = []
	must_sql_c = _build_must_sql_on("c", "content", must_tokens, dyn_params)
	mustnot_sql_c = _build_must_not_sql_on("c", "content", must_not_tokens, dyn_params)
	must_sql_s = _build_must_sql_on("sb", "section_text", must_tokens, dyn_params)
	mustnot_sql_s = _build_must_not_sql_on("sb", "section_text", must_not_tokens, dyn_params)

	should_patterns = _expand_should_patterns(should_tokens)
	should_params: List[Any] = []
	should_score_c = _build_should_score_sql_on("c", "content", should_patterns, should_params)
	should_score_s = _build_should_score_sql_on("sb", "section_text", should_patterns, should_params)

	# law LIKE fallback
	law_like = None
	if filter_law:
		_l = filter_law.strip().lower()
		law_map = {"ny": "%New York%", "new york": "%New York%", "us-ny": "%New York%", "de": "%Delaware%", "delaware": "%Delaware%", "us-de": "%Delaware%", "ca": "%California%", "california": "%California%", "us-ca": "%California%"}
		law_like = law_map.get(_l)

	with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
		cur.execute(f"SET LOCAL ivfflat.probes = {int(probes)}")
		sql = f"""
			WITH params AS (
			  SELECT %s::vector AS qvec,
					 websearch_to_tsquery('english', %s) AS q,
					 plainto_tsquery('simple', %s) AS q_simple,
					 %s::int AS pool_n,
					 %s::int AS k,
					 %s::float AS sparse_w,
					 %s::float AS should_w,
					 %s::text AS heading_like,
					 %s::text AS law,
					 %s::text AS law_like,
					 %s::text AS industry,
					 %s::text AS doc_type,
					 %s::text AS seat,
					 %s::text AS prefer_doc,
					 %s::text AS filter_doc_id,
					 %s::boolean AS enforce_constraints
			),

			/* CLAUSES */
			dense_clauses AS (
			  SELECT c.id,
					 row_number() OVER (ORDER BY c.embedding <-> (SELECT qvec FROM params)) AS r_dense
			  FROM clauses c
			  JOIN documents d ON d.document_id = c.document_id
			  WHERE c.embedding IS NOT NULL
				AND ((SELECT heading_like FROM params) IS NULL
					 OR c.heading_number LIKE (SELECT heading_like FROM params)
					 OR c.clause_number  LIKE (SELECT heading_like FROM params))
				AND ((SELECT filter_doc_id FROM params) IS NULL OR c.document_id = (SELECT filter_doc_id FROM params))
				AND {must_sql_c}
				AND {mustnot_sql_c}
				AND (
				  (SELECT enforce_constraints FROM params) = FALSE OR (
					((SELECT law FROM params) IS NULL OR d.governing_law = (SELECT law FROM params)
					 OR (d.governing_law IS NULL AND (SELECT law_like FROM params) IS NOT NULL AND c.content ILIKE (SELECT law_like FROM params)))
					AND ((SELECT industry FROM params) IS NULL OR d.industry = (SELECT industry FROM params))
					AND ((SELECT doc_type FROM params) IS NULL OR d.document_type = (SELECT doc_type FROM params))
				  )
				)
			  ORDER BY c.embedding <-> (SELECT qvec FROM params)
			  LIMIT (SELECT pool_n FROM params)
			),
			sparse_clauses AS (
			  SELECT c.id,
					 row_number() OVER (
					   ORDER BY ( ts_rank_cd(c.fts, (SELECT q FROM params), 32)
								+ 0.2 * ts_rank_cd(c.fts_simple, (SELECT q_simple FROM params)) ) DESC
					 ) AS r_sparse
			  FROM clauses c
			  JOIN documents d ON d.document_id = c.document_id
			  WHERE c.fts @@ (SELECT q FROM params)
				AND ((SELECT heading_like FROM params) IS NULL
					 OR c.heading_number LIKE (SELECT heading_like FROM params)
					 OR c.clause_number  LIKE (SELECT heading_like FROM params))
				AND ((SELECT filter_doc_id FROM params) IS NULL OR c.document_id = (SELECT filter_doc_id FROM params))
				AND {must_sql_c}
				AND {mustnot_sql_c}
				AND (
				  (SELECT enforce_constraints FROM params) = FALSE OR (
					((SELECT law FROM params) IS NULL OR d.governing_law = (SELECT law FROM params)
					 OR (d.governing_law IS NULL AND (SELECT law_like FROM params) IS NOT NULL AND c.content ILIKE (SELECT law_like FROM params)))
					AND ((SELECT industry FROM params) IS NULL OR d.industry = (SELECT industry FROM params))
					AND ((SELECT doc_type FROM params) IS NULL OR d.document_type = (SELECT doc_type FROM params))
				  )
				)
			  ORDER BY ( ts_rank_cd(c.fts, (SELECT q FROM params), 32)
					   + 0.2 * ts_rank_cd(c.fts_simple, (SELECT q_simple FROM params)) ) DESC
			  LIMIT (SELECT pool_n FROM params)
			),
			union_clauses AS (
			  SELECT id, MIN(r_dense) AS r_dense, MIN(r_sparse) AS r_sparse
			  FROM (
				SELECT id, r_dense, NULL::int AS r_sparse FROM dense_clauses
				UNION ALL
				SELECT id, NULL::int, r_sparse FROM sparse_clauses
			  ) u
			  GROUP BY id
			),
			fused_clauses AS (
			  SELECT c.id,
					 c.document_id, c.section_id, c.title, c.content,
					 (SELECT sparse_w FROM params) * COALESCE(1.0/((SELECT k FROM params) + uc.r_sparse), 0.0)
				   + (1.0 - (SELECT sparse_w FROM params)) * COALESCE(1.0/((SELECT k FROM params) + uc.r_dense), 0.0)
				   + (SELECT should_w FROM params) * ({should_score_c})
				   + 0.10 * CASE WHEN (SELECT law FROM params) IS NOT NULL AND d.governing_law = (SELECT law FROM params) THEN 1 ELSE 0 END
				   + 0.10 * CASE WHEN (SELECT doc_type FROM params) IS NOT NULL AND d.document_type = (SELECT doc_type FROM params) THEN 1 ELSE 0 END
				   + 0.10 * CASE WHEN (SELECT industry FROM params) IS NOT NULL AND d.industry = (SELECT industry FROM params) THEN 1 ELSE 0 END
				   + 0.15 * CASE WHEN (SELECT prefer_doc FROM params) IS NOT NULL AND c.document_id = (SELECT prefer_doc FROM params) THEN 1 ELSE 0 END AS rrf,
				 uc.r_dense, uc.r_sparse,
				 c.heading_number, c.seq, c.clause_type, c.defined_terms,
				 d.document_type, d.governing_law, d.industry,
				 'C'::text AS src
			  FROM union_clauses uc
			  JOIN clauses c ON c.id = uc.id
			  JOIN documents d ON d.document_id = c.document_id
			),

			/* SECTIONS */
			dense_sections AS (
			  SELECT sb.id,
					 row_number() OVER (ORDER BY sb.embedding <-> (SELECT qvec FROM params)) AS r_dense
			  FROM section_blobs sb
			  JOIN documents d ON d.document_id = sb.document_id
			  WHERE sb.embedding IS NOT NULL
				AND ((SELECT heading_like FROM params) IS NULL
					 OR sb.heading_number LIKE (SELECT heading_like FROM params))
				AND ((SELECT filter_doc_id FROM params) IS NULL OR sb.document_id = (SELECT filter_doc_id FROM params))
				AND {must_sql_s}
				AND {mustnot_sql_s}
				AND (
				  (SELECT enforce_constraints FROM params) = FALSE OR (
					((SELECT law FROM params) IS NULL OR d.governing_law = (SELECT law FROM params))
					AND ((SELECT industry FROM params) IS NULL OR d.industry = (SELECT industry FROM params))
					AND ((SELECT doc_type FROM params) IS NULL OR d.document_type = (SELECT doc_type FROM params))
				  )
				)
			  ORDER BY sb.embedding <-> (SELECT qvec FROM params)
			  LIMIT (SELECT pool_n FROM params)
			),
			sparse_sections AS (
			  SELECT sb.id,
					 row_number() OVER (
					   ORDER BY ts_rank_cd(to_tsvector('english', sb.section_text), (SELECT q FROM params), 32) DESC
					 ) AS r_sparse
			  FROM section_blobs sb
			  JOIN documents d ON d.document_id = sb.document_id
			  WHERE to_tsvector('english', sb.section_text) @@ (SELECT q FROM params)
				AND ((SELECT heading_like FROM params) IS NULL
					 OR sb.heading_number LIKE (SELECT heading_like FROM params))
				AND ((SELECT filter_doc_id FROM params) IS NULL OR sb.document_id = (SELECT filter_doc_id FROM params))
				AND {must_sql_s}
				AND {mustnot_sql_s}
				AND (
				  (SELECT enforce_constraints FROM params) = FALSE OR (
					((SELECT law FROM params) IS NULL OR d.governing_law = (SELECT law FROM params))
					AND ((SELECT industry FROM params) IS NULL OR d.industry = (SELECT industry FROM params))
					AND ((SELECT doc_type FROM params) IS NULL OR d.document_type = (SELECT doc_type FROM params))
				  )
				)
			  ORDER BY ts_rank_cd(to_tsvector('english', sb.section_text), (SELECT q FROM params), 32) DESC
			  LIMIT (SELECT pool_n FROM params)
			),
			union_sections AS (
			  SELECT id, MIN(r_dense) AS r_dense, MIN(r_sparse) AS r_sparse
			  FROM (
				SELECT id, r_dense, NULL::int AS r_sparse FROM dense_sections
				UNION ALL
				SELECT id, NULL::int, r_sparse FROM sparse_sections
			  ) u
			  GROUP BY id
			),
			fused_sections AS (
			  SELECT sb.id,
					 sb.document_id,
					 ('H:' || sb.heading_number) as section_id,
					 sb.title,
					 sb.section_text AS content,
					 (SELECT sparse_w FROM params) * COALESCE(1.0/((SELECT k FROM params) + us.r_sparse), 0.0)
				   + (1.0 - (SELECT sparse_w FROM params)) * COALESCE(1.0/((SELECT k FROM params) + us.r_dense), 0.0)
				   + (SELECT should_w FROM params) * ({should_score_s})
				   + 0.02 AS rrf,
				 us.r_dense, us.r_sparse,
				 sb.heading_number,
				 NULL::int AS seq,
				 NULL::text AS clause_type,
				 NULL::text[] AS defined_terms,
				 d.document_type, d.governing_law, d.industry,
				 'S'::text AS src
			  FROM union_sections us
			  JOIN section_blobs sb ON sb.id = us.id
			  JOIN documents d ON d.document_id = sb.document_id
			),

			fused_all AS (
			  SELECT * FROM fused_clauses
			  UNION ALL
			  SELECT * FROM fused_sections
			),
			ranked AS (
			  SELECT *,
					 ROW_NUMBER() OVER (
					   PARTITION BY document_id, heading_number
					   ORDER BY rrf DESC
					 ) AS rn
			  FROM fused_all
			)
			SELECT id, document_id, section_id, title, content, rrf, r_dense, r_sparse,
			       heading_number, seq, clause_type, defined_terms,
			       document_type, governing_law, industry, src
			FROM ranked
			WHERE rn <= %s
			ORDER BY rrf DESC
			LIMIT %s
		"""

		# Assemble parameters in exact order
		base_params = [
			qvec_lit, q_fts, query, pool_n, rrf_k,
			float(sparse_weight), float(should_weight),
			heading_like, filter_law, law_like, filter_industry, filter_doc_type,
			filter_seat, prefer_doc, filter_doc_id, bool(enforce_constraints),
		]
		# Build dynamic param list deterministically, matching placeholder order:
		def expand_list(tokens: Optional[List[str]]) -> List[str]:
			return [f"%{v}%" for tok in (tokens or []) for v in _expand_token_variants(tok)]
		dyn_ordered: List[Any] = []
		# clauses dense (must + not)
		dyn_ordered += expand_list(must_tokens)
		dyn_ordered += expand_list(must_not_tokens)
		# clauses sparse (must + not)
		dyn_ordered += expand_list(must_tokens)
		dyn_ordered += expand_list(must_not_tokens)
		# sections dense (must + not)
		dyn_ordered += expand_list(must_tokens)
		dyn_ordered += expand_list(must_not_tokens)
		# sections sparse (must + not)
		dyn_ordered += expand_list(must_tokens)
		dyn_ordered += expand_list(must_not_tokens)
		# should score (clauses)
		dyn_ordered += [f"%{p}%" for p in should_patterns]
		# should score (sections)
		dyn_ordered += [f"%{p}%" for p in should_patterns]

		all_params = base_params + dyn_ordered + [int(per_heading_cap), top_k]
		cur.execute(sql, all_params)
		rows = cur.fetchall()

	# De-dupe & attach neighbors/defs for clause rows (sections left as-is)
	seen = set(); clause_rows: List[Dict[str, Any]] = []; section_rows: List[Dict[str, Any]] = []
	for r in rows:
		key = (r["document_id"], r["section_id"])
		if key in seen:
			continue
		seen.add(key)
		item = {
			"id": r.get("id"),
			"document_id": r["document_id"],
			"section_id": r["section_id"],
			"title": r["title"],
			"content": r["content"],
			"rrf": float(r["rrf"]) if r["rrf"] is not None else None,
			"r_dense": int(r["r_dense"]) if r.get("r_dense") is not None else None,
			"r_sparse": int(r["r_sparse"]) if r.get("r_sparse") is not None else None,
			"heading_number": r.get("heading_number"),
			"seq": r.get("seq"),
			"clause_type": r.get("clause_type"),
			"defined_terms": r.get("defined_terms"),
			"document_type": r.get("document_type"),
			"governing_law": r.get("governing_law"),
			"industry": r.get("industry"),
			"src": r.get("src"),
		}
		if r.get("src") == "C":
			clause_rows.append(item)
		else:
			section_rows.append(item)

	enriched_clauses = _attach_neighbors_and_defs(
		clause_rows, neighbor_window=neighbor_window, max_def_match=max_defs
	)
	# Also enrich sections so they can contribute definitions
	enriched_sections = _attach_neighbors_and_defs(
		section_rows, neighbor_window=neighbor_window, max_def_match=max_defs
	)
	return enriched_clauses + enriched_sections


# ---------------------------
# Dense-only search (cosine)
# ---------------------------
def search(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
	"""Dense-only (cosine) search over clauses; keeps prior API for tools/tests."""
	_ensure_schema()
	_ensure_vector_index()
	_ensure_fts()
	_ensure_perf_indexes()
	qvec = _embed_query(query)
	rows = _search_dense(qvec, limit=top_k)
	return _attach_neighbors_and_defs(rows)


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
	_ensure_perf_indexes()

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

	# Rebuild section blobs after ingest so section index stays in sync
	_rebuild_section_blobs(reembed=True)

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
	**kwargs: Any,
) -> List[Dict[str, Any]]:
	"""End-to-end retrieval: hybrid by default; dense-only if use_hybrid=False."""
	if use_hybrid:
		return search_hybrid(
			query=query, pool_n=pool_n, top_k=top_k, expand_sparse=expand_sparse, **kwargs
		)
	qvec = _embed_query(query)
	rows = _search_dense(qvec, limit=top_k)
	return _attach_neighbors_and_defs(rows, neighbor_window=kwargs.get("neighbor_window"), max_def_match=kwargs.get("max_defs"))


# ---------------------------
# Clause graph schema & population
# ---------------------------

def _ensure_graph_schema() -> None:
	with _connect() as conn, conn.cursor() as cur:
		# Add heading_number column for numeric section identifiers
		cur.execute(
			"""
			ALTER TABLE IF EXISTS clauses
			  ADD COLUMN IF NOT EXISTS heading_number text,
			  ADD COLUMN IF NOT EXISTS clause_number  text
			"""
		)
		# Pattern ops indexes for fast prefix LIKE
		cur.execute("CREATE INDEX IF NOT EXISTS clauses_heading_like_idx ON clauses (heading_number text_pattern_ops)")
		cur.execute("CREATE INDEX IF NOT EXISTS clauses_clause_like_idx  ON clauses (clause_number  text_pattern_ops)")
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
	"""Derive clause_number (dotted like 10.2.1) and top-level heading_number."""
	with _connect() as conn, conn.cursor() as cur:
		params: List[Any] = []
		where = ""
		if document_id:
			where = "WHERE document_id = %s"
			params.append(document_id)
		cur.execute(
			f"""
			WITH first_line AS (
			  SELECT id,
			         regexp_replace(
			           NULLIF(regexp_replace(split_part(regexp_replace(content, E'\\r', '', 'g'), E'\\n', 1), '^\\s+', '', 'g'), ''),
			           E'\\s+', ' ', 'g'
			         ) AS line1
			  FROM clauses {where}
			)
			UPDATE clauses c
			SET clause_number = COALESCE(
			        substring(c.title from '(^\\s*[0-9]+(?:\\.[0-9]+)*)'),
			        (SELECT substring(line1 from '(^\\s*[0-9]+(?:\\.[0-9]+)*)') FROM first_line f WHERE f.id=c.id),
			        NULLIF(regexp_replace(c.section_id, '[^0-9\\.]', '', 'g'), '')
			     ),
			    heading_number = COALESCE(
			        CASE
			          WHEN position('.' in COALESCE(
			             substring(c.title from '(^\\s*[0-9]+(?:\\.[0-9]+)*)'),
			             (SELECT substring(line1 from '(^\\s*[0-9]+(?:\\.[0-9]+)*)') FROM first_line f WHERE f.id=c.id),
			             NULLIF(regexp_replace(c.section_id, '[^0-9\\.]', '', 'g'), '')
			          )) > 0
			          THEN split_part(
			               COALESCE(
			                 substring(c.title from '(^\\s*[0-9]+(?:\\.[0-9]+)*)'),
			                 (SELECT substring(line1 from '(^\\s*[0-9]+(?:\\.[0-9]+)*)') FROM first_line f WHERE f.id=c.id),
			                 NULLIF(regexp_replace(c.section_id, '[^0-9\\.]', '', 'g'), '')
			               ), '.', 1)
			          ELSE NULLIF(regexp_replace(c.section_id, '[^0-9]', '', 'g'), '')
			        END
			     )
			WHERE TRUE
			  {"AND c.document_id = %s" if document_id else ""}
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
	"""Scan content for references like 'Section 6.2' or 'Clause 14.1' and link within same document.
	Requires a legal token (Section/Clause/Article/§); splits simple ranges like 20.1.12-20.1.15.
	"""
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
		# Strict legal-reference regex with optional range
		ref_re = re.compile(
			r"\b(?:Sections?|Clauses?|Articles?|§)\s*(\d+(?:\.\d+)*(?:\s*(?:–|-|to)\s*\d+(?:\.\d+)*)?)",
			flags=re.IGNORECASE,
		)
		cur.execute(f"SELECT id, document_id, content FROM clauses {where}", tuple(params))
		rows = cur.fetchall()
		for r in rows:
			text = r.get("content") or ""
			matches = [m.group(1) for m in ref_re.finditer(text)]
			if not matches:
				continue
			mapping = doc_clause_map.get(r["document_id"], {})
			nums: List[str] = []
			for m in matches:
				m = m.replace("–", "-")
				if "-" in m or " to " in m:
					parts = re.split(r"\s*(?:-|to)\s*", m)
					if len(parts) == 2:
						start, end = parts
						# expand only if same prefix except last component
						sp = start.split(".")
						ep = end.split(".")
						if len(sp) == len(ep) and sp[:-1] == ep[:-1] and sp[-1].isdigit() and ep[-1].isdigit():
							for i in range(int(sp[-1]), int(ep[-1]) + 1):
								nums.append(".".join(sp[:-1] + [str(i)]))
						else:
							nums.extend([start, end])
				else:
					nums.append(m)
			for num in dict.fromkeys(nums):
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
				terms += re.findall(r'"([^\\"]{2,40})"\s*(?:means|shall\s+mean)', text, flags=re.IGNORECASE)
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


def retrieve_recursive(query: str, k: int = 8, max_hops: int = 2, relations: Optional[List[str]] = None, neighbor_window: Optional[int] = None, max_defs: Optional[int] = None) -> List[Dict[str, Any]]:
	"""Hybrid search seeds + follow refers_to/defines edges up to max_hops. Supports relation filtering and neighbor/defs overrides."""
	_ensure_schema()
	_ensure_vector_index()
	_ensure_fts()
	_ensure_graph_schema()
	_ensure_perf_indexes()
	seeds = search_hybrid(query, pool_n=200, top_k=k, neighbor_window=neighbor_window, max_defs=max_defs)
	seen = {int(r["id"]) for r in seeds}
	frontier = [int(r["id"]) for r in seeds]
	hops = 0
	out = list(seeds)
	rel_tuple: Tuple[str, ...] = tuple((relations or ["refers_to","defines"]))
	while frontier and hops < max_hops:
		dsts = _fetch_edges(frontier, relation_in=rel_tuple)
		new_ids = [i for i in dsts if i not in seen]
		if not new_ids:
			break
		ctx = _fetch_clause_rows(new_ids)
		out.extend(ctx)
		for i in new_ids:
			seen.add(i)
		frontier = new_ids
		hops += 1
	# Attach neighbors/defs across the final set
	return _attach_neighbors_and_defs(out, neighbor_window=neighbor_window, max_def_match=max_defs) 


def rebuild_sections(reembed: bool = True) -> None:
	"""Public entrypoint to refresh section blobs."""
	_rebuild_section_blobs(reembed=reembed) 