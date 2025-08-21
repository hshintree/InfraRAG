import os
from pathlib import Path
from typing import List
import sys

# Load .env for local runs
try:
	from dotenv import load_dotenv, find_dotenv
	load_dotenv(find_dotenv(), override=False)
except Exception:
	pass

# Ensure project root is on sys.path when running as a script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from src.ingestion import DocumentIngestionPipeline
from src.indexing import PgIndexer


USE_MODAL_EMBED = os.getenv("USE_MODAL_EMBED", "0") in {"1", "true", "True"}


def ingest_all(data_dir: str = os.getenv("DATA_DIR", "./data")) -> None:
	pipeline = DocumentIngestionPipeline()
	indexer = PgIndexer()
	model = None
	if not USE_MODAL_EMBED:
		from sentence_transformers import SentenceTransformer
		model = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

	path = Path(data_dir)
	files: List[str] = []
	for ext in ("*.xml", "*.pdf"):
		files.extend([str(p) for p in path.glob(ext)])

	for f in files:
		print(f"Ingesting: {f}")
		doc = pipeline.ingest_document(f)
		chunks = pipeline.chunk_document(doc)
		texts = [c.content for c in chunks]
		if USE_MODAL_EMBED:
			from adapters.modal_embedding import embed_texts_remote
			embs = embed_texts_remote(texts)
		else:
			embs = model.encode(texts, normalize_embeddings=True).tolist()
		indexer.upsert_document(doc)
		indexer.index_chunks(doc, chunks, embs)
	print("Done.")


def backfill_missing_embeddings(batch_size: int = 1000) -> None:
	"""Embed and fill any clauses with NULL embeddings."""
	import psycopg
	from adapters.modal_embedding import embed_texts_remote

	host = os.getenv("DB_HOST", "localhost")
	port = int(os.getenv("DB_PORT", "5433"))
	name = os.getenv("DB_NAME", "infra_rag")
	user = os.getenv("DB_USER", "postgres")
	pw = os.getenv("DB_PASSWORD", "changeme_local_pw")

	with psycopg.connect(host=host, port=port, dbname=name, user=user, password=pw) as conn:
		while True:
			with conn.cursor() as cur:
				cur.execute(
					"SELECT id, content FROM clauses WHERE embedding IS NULL LIMIT %s",
					(batch_size,)
				)
				rows = cur.fetchall()
				if not rows:
					print("No missing embeddings to backfill.")
					break
				ids = [r[0] for r in rows]
				texts = [r[1] or "" for r in rows]
				if USE_MODAL_EMBED:
					vecs = embed_texts_remote(texts)
				else:
					from sentence_transformers import SentenceTransformer
					model = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
					vecs = model.encode(texts, normalize_embeddings=True).tolist()
				# Update in batches
				for _id, v in zip(ids, vecs):
					vec_lit = "[" + ",".join(f"{float(x):.8f}" for x in v) + "]"
					cur.execute("UPDATE clauses SET embedding = %s WHERE id = %s", (vec_lit, _id))
				conn.commit()
				print(f"Backfilled {len(ids)} embeddings...")


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Ingest all documents from a directory into Postgres")
	parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "./data"))
	parser.add_argument("--use-modal", action="store_true", help="Use Modal for embeddings (overrides env)")
	parser.add_argument("--backfill-missing", action="store_true", help="Embed rows with NULL embeddings")
	args = parser.parse_args()
	if args.use_modal:
		os.environ["USE_MODAL_EMBED"] = "1"
		globals()["USE_MODAL_EMBED"] = True
	if args.backfill_missing:
		backfill_missing_embeddings()
	else:
		ingest_all(args.data_dir) 