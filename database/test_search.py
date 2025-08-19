import argparse
import os
import sys

# Ensure project root is on sys.path when running as a script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from adapters.retrieval_adapter import search, ingest_data_dir, search_hybrid
from adapters.retrieval_adapter import populate_clause_graph, retrieve_recursive


def main():
	parser = argparse.ArgumentParser(description="Quick test search against Postgres index")
	parser.add_argument("query", nargs="?", help="Search query")
	parser.add_argument("--ingest", action="store_true", help="Ingest data/ before searching")
	parser.add_argument("--top-k", type=int, default=5)
	parser.add_argument("--hybrid", action="store_true", help="Use hybrid (BM25 + vector)")
	parser.add_argument("--alpha", type=float, default=0.5, help="(Unused) Legacy weight; ignored for RRF hybrid")
	parser.add_argument("--pool-n", type=int, default=200, help="Hybrid pool size")
	parser.add_argument("--populate-graph", action="store_true", help="Populate clause graph (adjacent/refers_to/defines)")
	parser.add_argument("--recursive", action="store_true", help="Run recursive retrieval (BFS over graph)")
	parser.add_argument("--hops", type=int, default=2, help="Max hops for recursive retrieval")
	args = parser.parse_args()

	if args.ingest:
		stats = ingest_data_dir("./data")
		print(f"Ingested: {stats}")

	if args.populate_graph:
		populate_clause_graph()
		print("Populated clause graph (adjacent/refers_to/defines).")

	if args.recursive:
		if not args.query:
			print("Query is required for recursive retrieval")
			return
			
		results = retrieve_recursive(args.query, k=args.top_k, max_hops=args.hops)
	else:
		if not args.query:
			print("Query is required for search")
			return
		if args.hybrid:
			results = search_hybrid(args.query, pool_n=args.pool_n, top_k=args.top_k)
		else:
			results = search(args.query, top_k=args.top_k)

	for i, r in enumerate(results, 1):
		score = r.get("rrf")
		if score is None:
			score = r.get("cos_sim")
		if score is None:
			d = r.get("dist")
			score = (1.0 / (1.0 + d)) if d is not None else 0.0
		title = r.get("title") or ""
		print(f"{i}. [{score:.3f}] {title} ({r['document_id']} §{r['section_id']})")
		print((r.get("content") or "")[:200].replace("\n", " ") + ("..." if len(r.get("content") or "") > 200 else ""))
		if r.get("neighbors"):
			print(f"   neighbors: {len(r['neighbors'])}")
		if r.get("definitions"):
			print(f"   definitions: {len(r['definitions'])}")
		print()


if __name__ == "__main__":
	main() 