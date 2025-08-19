from __future__ import annotations

import argparse, json, os, time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# project import
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from adapters.retrieval_adapter import search_hybrid  # you already expose this

# ------------ Slot configs (minimal but pragmatic) ------------
# Each slot: base queries + soft/hard tokens + typical scoping hints.
SLOT_HINTS: Dict[str, Dict[str, Any]] = {
    "Parties": {
        "queries": ["preamble parties this agreement is made by and between", "parties and definitions preamble"],
        "should": ["between", "Party", "Parties", "Seller", "Buyer"],
    },
    "Definitions": {
        "queries": ["definitions", "interpretation definitions"],
        "should": ["means", "shall mean", "definition"],
        "heading_like": "1%"  # often 1. or 1.x
    },
    "Purchase and Sale": {
        "queries": ["purchase and sale", "sale and purchase", "subject matter"],
        "should": ["purchase", "sale", "subject matter"],
    },
    "Price": {
        "queries": ["contract sales price", "purchase price", "price"],
        "should": ["price", "pricing", "formula", "index", "LIBOR", "SOFR"],
    },
    "Adjustments": {
        "queries": ["price adjustment", "true-up", "final reconciliation", "adjustments"],
        "should": ["true-up", "adjustment", "reconciliation"],
    },
    "Closing": {
        "queries": ["closing", "completion", "date of substantial completion", "date of full operations"],
        "should": ["closing", "completion"],
    },
    "CPs": {
        "queries": ["conditions precedent", "CPs", "conditions to closing"],
        "should": ["condition precedent", "CP", "satisfaction"],
    },
    "R&W - Seller": {
        "queries": ["representations and warranties of seller", "seller representations warranties"],
        "should": ["represents", "warrants", "authority", "no conflict", "enforceability"],
    },
    "R&W - Buyer": {
        "queries": ["representations and warranties of buyer", "buyer representations warranties"],
        "should": ["represents", "warrants", "authority", "no conflict", "enforceability"],
    },
    "Covenants": {
        "queries": ["covenants", "undertakings", "affirmative covenants", "negative covenants"],
        "should": ["shall", "covenant", "undertakes"],
    },
    "Indemnities": {
        "queries": ["indemnity", "indemnification", "liabilities and indemnification"],
        "should": ["indemnify", "defend", "hold harmless"],
    },
    "Limitations": {
        "queries": ["limitation of liability", "limitations on liability", "exclusion of damages"],
        "should": ["cap", "indirect", "consequential", "lost profits", "basket"],
    },
    "Governing Law": {
        "queries": ["governing law"],
        "must": ["govern"],  # “governed by…”
    },
    "Dispute Resolution": {
        "queries": ["dispute resolution", "arbitration", "venue seat rules"],
        "should": ["arbitration", "seat", "rules", "LCIA", "ICC", "UNCITRAL", "London"],
    },
    "Notices": {
        "queries": ["notices", "form of notice", "delivery of notice"],
        "should": ["notice", "address", "email", "copy to"],
    },
    "Termination": {
        "queries": ["termination", "termination events", "default and termination"],
        "should": ["terminate", "event of default"],
    },
    "Miscellaneous": {
        "queries": ["miscellaneous", "general", "boilerplate"],
        "should": ["entire agreement", "severability", "amendments", "waiver", "assignment"],
    },

    # Optional
    "Change of Control": {
        "queries": ["change of control", "assignment by buyer", "assignment by seller"],
        "should": ["change of control", "assignment", "acquisition", "transfer"],
    },
    "Performance Guarantee": {
        "queries": ["guarantee", "parent guarantee", "letter of credit", "credit support"],
        "should": ["guarantee", "guarantor", "credit support"],
    },
    "Tax Matters": {
        "queries": ["taxes", "withholding taxes", "transfer taxes", "tax gross-up"],
        "should": ["withholding", "gross-up", "VAT", "sales tax"],
    },
    "Environmental": {
        "queries": ["environmental", "compliance with laws", "environmental laws"],
        "should": ["environmental", "laws", "permit"],
    },
    "Insurance": {
        "queries": ["insurance", "liability insurance", "coverage", "limits"],
        "should": ["insurance", "coverage", "policy", "limits"],
    },
}

# ----------------- helpers: merge adjacent & pack -----------------
def _merge_adjacent(rows: List[Dict[str, Any]], cap_chars: int = 2000) -> List[Dict[str, Any]]:
    if not rows: return rows
    rows_sorted = sorted(rows, key=lambda r: (r.get("document_id"), r.get("seq") or 0))
    out: List[Dict[str, Any]] = []
    buf: List[Dict[str, Any]] = []
    cur_doc = None
    for r in rows_sorted:
        d = r.get("document_id")
        if cur_doc is None:
            cur_doc = d
        if d != cur_doc:
            if buf: out.append(_stitch(buf))
            buf, cur_doc = [r], d
            continue
        if not buf:
            buf = [r]; continue
        prev = buf[-1]
        seq_ok = (r.get("seq") is not None and prev.get("seq") is not None
                  and (int(r["seq"]) - int(prev["seq"])) <= 1)
        length_ok = (len("\n".join(x.get("content") or "" for x in buf)) + len(r.get("content") or "")) <= cap_chars
        if seq_ok and length_ok:
            buf.append(r)
        else:
            out.append(_stitch(buf)); buf=[r]
    if buf: out.append(_stitch(buf))
    return out

def _stitch(buf: List[Dict[str, Any]]) -> Dict[str, Any]:
    buf_sorted = sorted(buf, key=lambda r: (r.get("seq") or 0))
    first = buf_sorted[0]
    merged_content = "\n".join((r.get("content") or "").strip() for r in buf_sorted if r.get("content"))
    stitched = dict(first)
    stitched["content"] = merged_content
    stitched["merged_count"] = len(buf_sorted)
    stitched["sources"] = [{"doc_id": r["document_id"], "section_id": r["section_id"], "title": r.get("title")} for r in buf_sorted]
    return stitched

def _take_top(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return rows[:n] if len(rows) > n else rows

# ----------------- retrieval plan per slot -----------------
def _filters_from_spec(spec: Dict[str, Any], slot: str) -> Dict[str, Any]:
    """Translate spec into retrieval filters/boosts by slot."""
    filters: Dict[str, Any] = {
        "filter_doc_type": spec.get("doc_type"),
        "filter_industry": spec.get("industry"),
        # neighbor/defs tuned here; ok to override via CLI
        "neighbor_window": 1,
        "max_defs": 6,
        "expand_sparse": True,
    }
    constraints = (spec.get("constraints") or {})
    # Law is global constraint: enforce where it matters most
    if slot in ("Governing Law", "Dispute Resolution"):
        if constraints.get("law"):
            filters["filter_law"] = constraints["law"]
            filters["enforce_constraints"] = True
        # Arbitration seat appears in content, so use should/must at query level (below)
        # Do not constrain doc_type/industry for these slots, to avoid hard-filtering out valid clauses
        filters["filter_doc_type"] = None
        filters["filter_industry"] = None
    return filters

def _slot_query_and_tokens(slot: str, spec: Dict[str, Any]) -> Tuple[str, List[str], List[str], List[str], Optional[str]]:
    hints = SLOT_HINTS.get(slot, {"queries":[slot.lower()]})
    base_q = " ".join(hints.get("queries", [slot.lower()]))  # simple concat; works well with your RRF
    must = hints.get("must", [])
    should = hints.get("should", [])
    must_not: List[str] = []

    # Add constraint-guided tokens for key slots
    if slot == "Dispute Resolution":
        seat = (spec.get("constraints") or {}).get("arbitration_seat")
        if seat:
            should = list(dict.fromkeys(should + [seat]))
        law = (spec.get("constraints") or {}).get("law")
        if law and law.lower() in ("ny","new york","us-ny"):
            should = list(dict.fromkeys(should + ["New York"]))
        # Ensure "arbitration" strongly present
        must = list(dict.fromkeys(must + ["arbitr"]))
    if slot == "Governing Law":
        law = (spec.get("constraints") or {}).get("law")
        if law and law.lower() in ("ny","new york","us-ny"):
            should = list(dict.fromkeys(should + ["New York"]))
        must = list(dict.fromkeys(must + ["govern"]))  # “governed by”
    return base_q, must, must_not, should, hints.get("heading_like")

def retrieve_slot(slot: str, spec: Dict[str, Any], top_k: int = 6, pool_n: int = 300, scoped_top_k: int = 3, merge_adjacent: bool = True, merge_cap: int = 2200) -> Dict[str, Any]:
    base_q, must, must_not, should, heading_like = _slot_query_and_tokens(slot, spec)
    filters = _filters_from_spec(spec, slot)
    if heading_like: filters["heading_like"] = heading_like

    # Pass 1: broad, metadata-aware
    rows = search_hybrid(
        base_q,
        pool_n=pool_n,
        top_k=top_k,
        must_tokens=must or None,
        must_not_tokens=must_not or None,
        should_tokens=should or None,
        sparse_weight=0.6,
        should_weight=0.05,
        **filters
    )

    # Optional scoped rerun against most-common doc among top hits
    scoped = []
    if rows:
        top_docs = [r.get("document_id") for r in rows if r.get("document_id")]
        if top_docs:
            prefer_doc = Counter(top_docs).most_common(1)[0][0]
            scoped = search_hybrid(
                base_q,
                pool_n=pool_n,
                top_k=scoped_top_k,
                must_tokens=must or None,
                must_not_tokens=must_not or None,
                should_tokens=should or None,
                sparse_weight=0.6,
                should_weight=0.05,
                filter_doc_id=prefer_doc,
                neighbor_window=filters.get("neighbor_window"),
                max_defs=filters.get("max_defs"),
            )

    # Combine & dedupe by (doc_id, section_id)
    seen = set()
    combined: List[Dict[str, Any]] = []
    for r in (rows + scoped):
        key = (r.get("document_id"), r.get("section_id"))
        if key in seen: continue
        seen.add(key); combined.append(r)

    # Optionally merge near-adjacent hits within same doc
    stitched = _merge_adjacent(combined, cap_chars=merge_cap) if merge_adjacent else combined

    # Assemble slot package
    top_items = _take_top(stitched, n=min(3, top_k))  # keep a few; model can synthesize from these
    sources = []
    for r in top_items:
        srcs = r.get("sources") or [{"doc_id": r["document_id"], "section_id": r["section_id"], "title": r.get("title")}]
        sources.extend(srcs)

    # Build a small “definitions table” seed for this slot
    defs = []
    for r in top_items:
        for d in (r.get("definitions") or []):
            defs.append({"section_id": d["section_id"], "title": d["title"], "content": d["content"], "doc_id": r["document_id"]})

    package = {
        "slot": slot,
        "query": base_q,
        "must": must,
        "should": should,
        "filters": filters,
        "items": [
            {
                "doc_id": r["document_id"],
                "section_id": r["section_id"],
                "title": r.get("title"),
                "heading_number": r.get("heading_number"),
                "seq": r.get("seq"),
                "rrf": r.get("rrf"),
                "content": r.get("content"),
                "neighbors": r.get("neighbors") or [],
                "definitions": r.get("definitions") or [],
                "sources": r.get("sources") or [{"doc_id": r["document_id"], "section_id": r["section_id"], "title": r.get("title")}],
                "merged_count": r.get("merged_count"),
            }
            for r in top_items
        ],
        "sources": sources,
        "definitions": defs,
    }
    return package

def build_scp(spec: Dict[str, Any], include_optional: bool = True, top_k: int = 6, pool_n: int = 300) -> Dict[str, Any]:
    required = spec.get("required", [])
    optional = spec.get("optional", []) if include_optional else []
    # 1) Retrieve per-slot
    slots: Dict[str, Any] = {}
    for slot in required + optional:
        slots[slot] = retrieve_slot(slot, spec, top_k=top_k, pool_n=pool_n)

    # 2) Build a consolidated definitions table (term -> candidate snippets)
    # Note: You can add a real term extractor later; here we just carry the gathered defs.
    def_table: List[Dict[str, Any]] = []
    for s in slots.values():
        def_table.extend(s.get("definitions", []))

    return {
        "spec": spec,
        "created_at": int(time.time()),
        "slots": slots,
        "definitions_table": def_table,
        "notes": {
            "neighbor_window": 1,
            "max_defs": 6,
            "fusion": "RRF hybrid",
            "scoped_rerun": True,
            "merged_adjacent": True
        }
    }

def _default_spec() -> Dict[str, Any]:
    return {
        "doc_type": "Purchase Agreement",
        "jurisdiction": "CO",
        "industry": "Power",
        "required": ["Parties","Definitions","Purchase and Sale","Price","Adjustments","Closing","CPs",
                     "R&W - Seller","R&W - Buyer","Covenants","Indemnities","Limitations","Governing Law",
                     "Dispute Resolution","Notices","Termination","Miscellaneous"],
        "optional": ["Change of Control","Performance Guarantee","Tax Matters","Environmental","Insurance"],
        "constraints": {"arbitration_seat":"London","law":"NY"},
    }

def main():
    ap = argparse.ArgumentParser(description="Build Section Context Protocol (SCP) from retriever")
    ap.add_argument("--spec-file", default=None, help="JSON file with drafting spec; if omitted, uses default")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--pool-n", type=int, default=300)
    ap.add_argument("--no-optional", action="store_true")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    spec = _default_spec()
    if args.spec_file:
        with open(args.spec_file, "r") as f:
            spec = json.load(f)

    scp = build_scp(spec, include_optional=not args.no_optional, top_k=args.top_k, pool_n=args.pool_n)

    os.makedirs(args.out, exist_ok=True)
    fname = os.path.join(args.out, f"scp_{int(time.time())}.json")
    with open(fname, "w") as f:
        json.dump(scp, f, ensure_ascii=False, indent=2)

    # brief console summary
    req = len(spec.get("required", []))
    opt = len(spec.get("optional", []))
    built = len(scp["slots"])
    print(f"Built SCP with {built} slots (required={req} optional={opt}) → {fname}")
    for name, pkg in scp["slots"].items():
        print(f"- {name}: {len(pkg.get('items', []))} items, {len(pkg.get('definitions', []))} def-snippets")

if __name__ == "__main__":
    main()
