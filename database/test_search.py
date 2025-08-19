import argparse
import os
import sys
from collections import Counter
from itertools import groupby
from typing import Any, Dict, List

# Ensure project root is on sys.path when running as a script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from adapters.retrieval_adapter import search, ingest_data_dir, search_hybrid
from adapters.retrieval_adapter import populate_clause_graph, retrieve_recursive


def _parse_semicolon_list(val: str) -> List[str]:
	if not val:
		return []
	return [t.strip() for t in val.split(";") if t.strip()]


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	seen = set()
	out: List[Dict[str, Any]] = []
	for r in rows:
		key = (r.get("document_id"), r.get("section_id"))
		if key in seen:
			continue
		seen.add(key)
		out.append(r)
	return out


def _stitch(buf: List[Dict[str, Any]]) -> Dict[str, Any]:
	buf_sorted = sorted(buf, key=lambda r: (r.get("seq") or 0))
	first = buf_sorted[0]
	merged_content = "\n".join((r.get("content") or "").strip() for r in buf_sorted if r.get("content"))
	stitched = dict(first)
	stitched["content"] = merged_content
	stitched["merged_count"] = len(buf_sorted)
	return stitched


def _merge_adjacent(rows: List[Dict[str, Any]], cap: int = 2000) -> List[Dict[str, Any]]:
	if not rows:
		return rows
	rows_sorted = sorted(rows, key=lambda r: (r.get("document_id"), r.get("seq") or 0))
	out: List[Dict[str, Any]] = []
	for doc_id, group in groupby(rows_sorted, key=lambda r: r.get("document_id")):
		buf: List[Dict[str, Any]] = []
		for r in list(group):
			if not buf:
				buf = [r]
				continue
			prev = buf[-1]
			seq_ok = (r.get("seq") is not None and prev.get("seq") is not None and (int(r.get("seq")) - int(prev.get("seq"))) <= 1)
			length_ok = (len(" ".join(x.get("content") or "" for x in buf)) + len(r.get("content") or "")) <= cap
			if seq_ok and length_ok:
				buf.append(r)
			else:
				out.append(_stitch(buf))
				buf = [r]
		if buf:
			out.append(_stitch(buf))
	return out


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
	# Stage 1 additions
	parser.add_argument("--sparse-weight", type=float, default=0.6)
	parser.add_argument("--should-weight", type=float, default=0.05)
	parser.add_argument("--must", default="", help="semicolon-separated tokens that must appear")
	parser.add_argument("--must-not", default="", help="semicolon-separated tokens to exclude")
	parser.add_argument("--should", default="", help="semicolon-separated tokens to soft-boost")
	parser.add_argument("--heading-like", default=None, help="SQL LIKE pattern for heading_number (e.g. '10.%')")
	parser.add_argument("--filter-doc-type", dest="filter_doc_type", default=None)
	parser.add_argument("--filter-industry", dest="filter_industry", default=None)
	parser.add_argument("--filter-law", dest="filter_law", default=None)
	parser.add_argument("--filter-seat", dest="filter_seat", default=None)
	parser.add_argument("--prefer-doc", dest="prefer_doc", default=None)
	parser.add_argument("--doc-id", dest="doc_id", default=None, help="Hard filter to a single document")
	parser.add_argument("--enforce-constraints", action="store_true", help="Turn law/seat/etc into hard filters")
	parser.add_argument("--relations", default="refers_to,defines", help="Comma-separated relations for recursive retrieval")
	parser.add_argument("--neighbor-window", type=int, default=None, help="Override NEIGHBOR_WINDOW per run")
	parser.add_argument("--neighbors", type=int, default=None, help="Alias for neighbor-window")
	parser.add_argument("--max-defs", type=int, default=None, help="Max definition expansions")
	parser.add_argument("--scoped-rerun", action="store_true")
	parser.add_argument("--scoped-top-k", type=int, default=5)
	parser.add_argument("--merge-adjacent", action="store_true")
	parser.add_argument("--merge-cap-chars", type=int, default=2000)
	parser.add_argument("--json", action="store_true")
	parser.add_argument("--explain", action="store_true")
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
		relations = [s.strip() for s in (args.relations or "").split(",") if s.strip()]
		results = retrieve_recursive(
			args.query,
			k=args.top_k,
			max_hops=args.hops,
			relations=relations if relations else None,
			neighbor_window=(args.neighbor_window if args.neighbor_window is not None else args.neighbors),
			max_defs=args.max_defs,
		)
	else:
		if not args.query:
			print("Query is required for search")
			return
		if args.hybrid:
			must = _parse_semicolon_list(args.must)
			must_not = _parse_semicolon_list(args.must_not)
			should = _parse_semicolon_list(args.should)
			results = search_hybrid(
				args.query,
				pool_n=args.pool_n,
				top_k=args.top_k,
				sparse_weight=args.sparse_weight,
				should_weight=args.should_weight,
				must_tokens=must,
				must_not_tokens=must_not,
				should_tokens=should,
				heading_like=args.heading_like,
				filter_doc_type=args.filter_doc_type,
				filter_industry=args.filter_industry,
				filter_law=args.filter_law,
				filter_seat=args.filter_seat,
				filter_doc_id=args.doc_id,
				prefer_doc=args.prefer_doc,
				enforce_constraints=args.enforce_constraints,
				neighbor_window=(args.neighbor_window if args.neighbor_window is not None else args.neighbors),
				max_defs=args.max_defs,
				explain=args.explain,
			)
			# Scoped rerun within the top document
			if args.scoped_rerun and results:
				top_docs = [r.get("document_id") for r in results[: args.top_k] if r.get("document_id")]
				if top_docs:
					top_doc = Counter(top_docs).most_common(1)[0][0]
					scoped = search_hybrid(
						args.query,
						pool_n=args.pool_n,
						top_k=args.scoped_top_k,
						sparse_weight=args.sparse_weight,
						should_weight=args.should_weight,
						filter_doc_id=top_doc,
						heading_like=args.heading_like,
						neighbor_window=(args.neighbor_window if args.neighbor_window is not None else args.neighbors),
						max_defs=args.max_defs,
					)
					results = _dedupe_rows(results + scoped)
		else:
			results = search(args.query, top_k=args.top_k)

	# Optional merge
	if args.merge_adjacent:
		results = _merge_adjacent(results, cap=args.merge_cap_chars)

	if args.json:
		import json as _json
		print(_json.dumps(results, ensure_ascii=False, indent=2))
		return

	# Pretty print with optional explanation
	for i, r in enumerate(results, 1):
		score = r.get("rrf")
		if score is None:
			score = r.get("cos_sim")
		if score is None:
			d = r.get("dist")
			score = (1.0 / (1.0 + d)) if d is not None else 0.0
		title = r.get("title") or ""
		meta_bits: List[str] = []
		if args.explain:
			bonuses: List[str] = []
			if args.filter_law and r.get("governing_law") == args.filter_law:
				bonuses.append("law +0.1")
			if args.filter_doc_type and r.get("document_type") == args.filter_doc_type:
				bonuses.append("doc_type +0.1")
			if args.filter_industry and r.get("industry") == args.filter_industry:
				bonuses.append("industry +0.1")
			if args.prefer_doc and r.get("document_id") == args.prefer_doc:
				bonuses.append("prefer_doc +0.15")
			should_tokens = _parse_semicolon_list(args.should)
			if should_tokens and any((t.lower() in (r.get("content") or "").lower()) for t in should_tokens):
				bonuses.append("should +0.05")
			meta_bits.append(f"(r_dense={r.get('r_dense')} r_sparse={r.get('r_sparse')}{(' | ' + ', '.join(bonuses)) if bonuses else ''})")
		print(f"{i}. [{score:.3f}] {title} ({r['document_id']} §{r['section_id']})")
		print((r.get("content") or "")[:200].replace("\n", " ") + ("..." if len(r.get("content") or "") > 200 else ""))
		if r.get("neighbors"):
			print(f"   neighbors: {len(r['neighbors'])}")
		if r.get("definitions"):
			print(f"   definitions: {len(r['definitions'])}")
		if meta_bits:
			print(f"   {' '.join(meta_bits)}")
		print()


if __name__ == "__main__":
	main() 