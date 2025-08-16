import os
from pathlib import Path
from typing import List
import sys

# Ensure project root is on sys.path when running as a script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from sentence_transformers import SentenceTransformer

from src.ingestion import DocumentIngestionPipeline
from src.indexing import PgIndexer


USE_MODAL_EMBED = os.getenv("USE_MODAL_EMBED", "0") in {"1", "true", "True"}

def ingest_all(data_dir: str = "./data") -> None:
	pipeline = DocumentIngestionPipeline()
	indexer = PgIndexer()
	model = None
	if not USE_MODAL_EMBED:
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


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Ingest all documents from a directory into Postgres")
	parser.add_argument("--data-dir", default="./data")
	args = parser.parse_args()
	ingest_all(args.data_dir) 