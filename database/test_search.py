import argparse
import os
import sys

# Ensure project root is on sys.path when running as a script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from adapters.retrieval_adapter import search, ingest_data_dir


def main():
	parser = argparse.ArgumentParser(description="Quick test search against Postgres index")
	parser.add_argument("query", help="Search query")
	parser.add_argument("--ingest", action="store_true", help="Ingest data/ before searching")
	parser.add_argument("--top-k", type=int, default=5)
	args = parser.parse_args()

	if args.ingest:
		stats = ingest_data_dir("./data")
		print(f"Ingested: {stats}")

	results = search(args.query, top_k=args.top_k)
	for i, r in enumerate(results, 1):
		print(f"{i}. [{r['score']:.3f}] {r['title']} ({r['document_id']} §{r['section_id']})")
		print(r["content"][:200].replace("\n", " ") + ("..." if len(r["content"]) > 200 else ""))
		print()


if __name__ == "__main__":
	main() 