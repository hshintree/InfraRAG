import argparse
import os
import sys

# Ensure project root is on sys.path when running as a script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from adapters.retrieval_adapter import search, ingest_data_dir, search_hybrid


def main():
	parser = argparse.ArgumentParser(description="Quick test search against Postgres index")
	parser.add_argument("query", help="Search query")
	parser.add_argument("--ingest", action="store_true", help="Ingest data/ before searching")
	parser.add_argument("--top-k", type=int, default=5)
	parser.add_argument("--hybrid", action="store_true", help="Use hybrid (BM25 + vector)")
	parser.add_argument("--alpha", type=float, default=0.5, help="(Unused) Legacy weight; ignored for RRF hybrid")
	parser.add_argument("--pool-n", type=int, default=200, help="Hybrid pool size")
	args = parser.parse_args()

	if args.ingest:
		stats = ingest_data_dir("./data")
		print(f"Ingested: {stats}")

	if args.hybrid:
		results = search_hybrid(args.query, pool_n=args.pool_n, top_k=args.top_k)
	else:
		results = search(args.query, top_k=args.top_k)

	for i, r in enumerate(results, 1):
		# Prefer rrf (hybrid), else cosine similarity, else convert from distance
		score = r.get("rrf")
		if score is None:
			score = r.get("cos_sim")
		if score is None:
			d = r.get("dist")
			score = (1.0 / (1.0 + d)) if d is not None else 0.0
		print(f"{i}. [{score:.3f}] {r['title']} ({r['document_id']} §{r['section_id']})")
		print(r["content"][:200].replace("\n", " ") + ("..." if len(r["content"]) > 200 else ""))
		print()


if __name__ == "__main__":
	main() 