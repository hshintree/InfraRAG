from __future__ import annotations

import argparse, json, os, time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# project import
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load .env for local runs
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# Force local embeddings for this process before importing retriever (import reads env)
os.environ["USE_MODAL_EMBED"] = os.getenv("USE_MODAL_EMBED", "0")

from adapters.retrieval_adapter import search_hybrid  # you already expose this

PER_SLOT_MAX = int(os.getenv("SCP_PER_SLOT_MAX", "3"))
PER_SLOT_MIN = int(os.getenv("SCP_PER_SLOT_MIN", "2"))
LAW_SLOT_MIN = int(os.getenv("SCP_LAW_SLOT_MIN", "1"))

# LangDSPy orchestrator (optional, global)
USE_LANG_DSPY = os.getenv("USE_LANG_DSPY", "0").lower() in ("1", "true")
ORCH = None
if USE_LANG_DSPY:
    try:
        from langchain_openai import ChatOpenAI
        from adapters.langdspy_orchestrator import LangDSPyOrchestrator

        llm_model = os.getenv("LLM_MODEL", os.getenv("LANGCHAIN_MODEL", "gpt-4o-mini"))
        llm = ChatOpenAI(model=llm_model, temperature=0)
        ORCH = LangDSPyOrchestrator(llm, cfg=os.environ)
        print(f"[langdspy] enabled with model={llm_model}")
        if os.getenv("RERANKER_MODEL"):
            print(f"[reranker] using {os.getenv('RERANKER_MODEL')}")
    except Exception as e:
        print(f"[langdspy] DISABLED (init error): {e}")
        ORCH = None

# ------------ Slot configs (minimal but pragmatic) ------------
# Each slot: base queries + soft/hard tokens + typical scoping hints.
SLOT_HINTS: Dict[str, Dict[str, Any]] = {
    "Parties": {
        "queries": [
            "preamble parties this agreement is made by and between",
            "this agreement is made by and between the"
        ],
        "should": [
            "Parties","between","this Agreement",
            "Buyer","Seller","Vendor","Purchaser","Contractor","Employer",
            "Lender","Borrower","Concessionaire","Grantor","Authority"
        ],
    },
    "Definitions": {
        "queries": ["definitions", "interpretation definitions"],
        "should": ["means", "shall mean", "definition"],
        "heading_like": "1%"  # often 1. or 1.x
    },
    "Purchase and Sale": {
        "queries": ["purchase and sale", "sale and purchase", "subject matter"],
        # no hard MUSTs – allow supply/transfer style language in infra contracts
        "should": ["purchase", "sale", "subject matter", "transfer", "deliver"],
    },
    "Price": {
        "queries": ["contract sales price", "purchase price", "price"],
        "should": ["price", "pricing", "formula", "index", "LIBOR", "SOFR", "EURIBOR"],
    },
    "Adjustments": {
        "queries": ["price adjustment", "true-up", "final reconciliation", "adjustments"],
        "should": ["true-up", "adjustment", "reconciliation"],
    },
    "Closing": {
        # avoid EPC-specific phrases here; keep this generic for P&S and financing docs
        "queries": ["closing", "completion"],
        "must": ["closing"],
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
        "must": ["indemn"],
        "should": ["indemnify", "defend", "hold harmless"],
    },
    "Limitations": {
        "queries": ["limitation of liability", "limitations on liability", "exclusion of damages"],
        "should": ["cap", "indirect", "consequential", "lost profits", "basket"],
    },
    "Governing Law": {
        "queries": ["governing law"],
        "must": ["govern"],
        "should": ["governing law","laws of"],
    },
    "Dispute Resolution": {
        "queries": ["dispute resolution", "arbitration", "venue seat rules"],
        "must": ["arbitr"],
        # keep institutions broad; no cities/countries hard-coded
        "should": ["arbitration","seat","rules","LCIA","ICC","UNCITRAL","SIAC","HKIAC","SCC","AAA","JAMS"],
    },
    "Notices": {
        "queries": ["notices", "form of notice", "delivery of notice"],
        "should": ["notice", "address", "email"],
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
        "queries": [
            "change of control",
            "change in control",
            "transfer of control",
            "ownership change",
            "assignment novation (control)"
        ],
        "must": ["control"],
        "should": [
            "change of control", "change in control", "majority", "voting",
            "ownership", "assignment", "novation", "directly or indirectly"
        ],
        "must_not": ["tax", "transfer tax", "transfer taxes", "sales or use tax", "VAT"],
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

# Title-bias helpers
def _title_matches(slot: str, title: str) -> bool:
    if not title: return False
    t = title.lower()
    return (
        (slot == "Parties" and ("parties" in t or "preamble" in t)) or
        (slot == "Definitions" and ("definition" in t or "interpretation" in t)) or
        (slot == "Purchase and Sale" and (("purchase" in t and "sale" in t) or "sale and purchase" in t)) or
        (slot == "Governing Law" and "governing law" in t) or
        (slot == "Dispute Resolution" and ("dispute resolution" in t or "arbitration" in t)) or
        (slot == "Notices" and "notice" in t) or
        (slot == "Termination" and ("termination" in t or "default" in t))
    )

def _prefer_title_matches(slot: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items: return items
    good = [r for r in items if _title_matches(slot, (r.get("title") or ""))]
    bad  = [r for r in items if r not in good]
    return good + bad

# Drop too-short fragments
def _drop_too_short(rows: List[Dict[str, Any]], min_chars: int = 280) -> List[Dict[str, Any]]:
    out = [r for r in rows if len((r.get("content") or "").strip()) >= min_chars]
    return out or rows

# --- soft backfill ---
def _backfill_minimum(slot: str, primary: List[Dict[str, Any]], pool: List[Dict[str, Any]], fallback: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    """If we filtered too aggressively, top up with next best long-ish items from pool."""
    need = (LAW_SLOT_MIN if slot in ("Governing Law",) else PER_SLOT_MIN) - len(primary)
    if need <= 0:
        return primary
    seen = {(r.get("doc_id") or r.get("document_id"), r.get("section_id")) for r in primary}
    candidates = [r for r in pool if (r.get("doc_id") or r.get("document_id"), r.get("section_id")) not in seen]
    if len(candidates) < need and fallback:
        extra = [r for r in fallback if (r.get("doc_id") or r.get("document_id"), r.get("section_id")) not in seen]
        candidates += extra
    candidates = sorted(candidates, key=lambda r: (-len((r.get("content") or "")), -(float(r.get("rrf") or 0.0))))
    return primary + candidates[:need]

# Slot on-topic filters (generic scoring)
import re as _re

_SLOT_TOKENS = {
    "Parties": ["parties", "preamble"],
    "Definitions": ["definition", "interpretation", "means", "shall mean"],
    "Purchase and Sale": ["purchase", "sale", "subject matter", "shall sell", "shall purchase", "shall buy", "shall take", "transfer", "deliver"],
    "Price": ["price", "pricing", "formula", "index", "libor", "sofr", "euribor"],
    "Adjustments": ["adjust", "true-up", "reconcil"],
    "Closing": ["closing", "completion"],
    "CPs": ["condition precedent", "conditions to closing"],
    "R&W - Seller": ["represent", "warrant"],
    "R&W - Buyer": ["represent", "warrant"],
    "Covenants": ["covenant", "undertake", "shall"],
    "Indemnities": ["indemn", "hold harmless", "defend"],
    "Limitations": ["limitation", "limitations", "consequential", "lost profits", "cap"],
    "Governing Law": ["governing law", "governed by the laws"],
    "Dispute Resolution": ["arbitra","seat","rules","lcia","icc","uncitral","siac","hkiac","scc","aaa","jams"],
    "Notices": ["notice", "address", "email"],
    "Termination": ["terminate", "termination", "event of default"],
    "Miscellaneous": ["entire agreement", "severability", "amendment", "waiver", "assignment"],
}

_CLAUSE_TYPE_MAP = {
    "Indemnities": {"Indemnities", "Indemnity"},
    "Limitations": {"Limitations", "Liability"},
    "Purchase and Sale": {"Purchase", "Sale"},
    "Price": {"Price", "Payment", "Pricing"},
    "Notices": {"Notices"},
    "Termination": {"Termination", "Default"},
    "Covenants": {"Covenants"},
    "Governing Law": {"Governing Law"},
    "Dispute Resolution": {"Dispute Resolution", "Arbitration"},
}

_PAT = {
    "parties": _re.compile(r"\b(this agreement\s+is\s+made\s+by\s+and\s+between|by\s+and\s+between)\b", _re.I),
    "defs": _re.compile(r"(“[^”]{2,80}”|\"[^\"]{2,80}\")\s+(means|shall\s+mean)\b", _re.I),
    "defs_many": _re.compile(r"((“[^”]{2,80}”|\"[^\"]{2,80}\")\s+(means|shall\s+mean)\b.*?){3,}", _re.I|_re.S),
    "rw_phrase": _re.compile(r"\brepresent(s|ations)?\s+and\s+warrant(s|ies)?\b", _re.I),
    "ps_sell": _re.compile(r"\bshall\s+sell\b", _re.I),
    "ps_take": _re.compile(r"\b(shall\s+purchase|shall\s+buy|shall\s+take)\b", _re.I),
    "indemn": _re.compile(r"\b(indemnif\w*|hold harmless|defend)\b", _re.I),
    "limitation": _re.compile(r"\b(limitation of liability|consequential|lost profits|cap\b)\b", _re.I),
    "cp": _re.compile(r"\b(condition[s]?\s+precedent|conditions?\s+to\s+closing)\b", _re.I),
    "close": _re.compile(r"\b(closing (date)?|completion)\b", _re.I),
    "govlaw": _re.compile(r"\bgoverned by the laws?\b", _re.I),
    "govlaw_strict": _re.compile(r"\bshall\s+be\s+governed\b.{0,40}\blaws?\b", _re.I),
    "conflicts": _re.compile(r"without\s+regard\s+to\s+(the\s+)?conflict(s)?\s+of\s+law", _re.I),
    "arb": _re.compile(r"\barbitra\w+\b|\bLCIA\b|\bICC\b|\bUNCITRAL\b|\bseat of arbitration\b", _re.I),
    "notice": _re.compile(r"\bnotice\b", _re.I),
    "term": _re.compile(r"\b(terminate|termination|event of default)\b", _re.I),
    "misc": _re.compile(r"\b(entire agreement|severability|amendment|waiver|assignment)\b", _re.I),
}

def slot_gate(slot: str, text: str) -> bool:
    tl = (text or "").lower()
    if slot == "Governing Law":
        return bool(_PAT["govlaw_strict"].search(tl))
    if slot == "Dispute Resolution":
        return ("arbitra" in tl)
    if slot == "Parties":
        return bool(_PAT["parties"].search(tl)) or ("parties" in tl)
    if slot == "Definitions":
        return bool(_PAT["defs"].search(text or ""))
    if slot == "Purchase and Sale":
        return ("purchase and sale" in tl) or bool(_PAT["ps_sell"].search(tl) or _PAT["ps_take"].search(tl))
    return True

def _slot_score_generic(slot: str, item: Dict[str, Any], constraints: Dict[str, Any]) -> float:
    title = (item.get("title") or "").lower()
    text  = (item.get("content") or "")
    tlow  = text.lower()
    L = len(text)

    score = 0.0
    # soft title cue (first word of slot or key alias)
    if slot.lower().split()[0] in title:
        score += 1.0
    for tok in _SLOT_TOKENS.get(slot, []):
        if tok in title or tok in tlow:
            score += 0.6
    ctype = (item.get("clause_type") or "").lower()
    wanted = _CLAUSE_TYPE_MAP.get(slot)
    if wanted:
        score += 0.8 if any(w.lower() in ctype for w in wanted) else 0.0
    if L >= 900:   score += 0.8
    elif L >= 600: score += 0.5
    elif L >= 350: score += 0.2
    if slot == "Definitions":
        if _PAT["defs_many"].search(text):
            score += 0.8
        elif _PAT["defs"].search(text):
            score += 0.4
        else:
            score -= 0.6
    # light demotion for common noise near P&S (but allow real hits to win)
    off = ("tax", "withholding", "export", "sanction", "assignment", "novat")
    if slot == "Purchase and Sale" and any(w in tlow for w in off):
        score -= 0.35

    # slot-specific regex signals (lift ties; reduce off-topic)
    if slot == "Parties" and _PAT["parties"].search(tlow):
        score += 0.8
    if slot == "Definitions" and _PAT["defs"].search(item.get("content") or ""):
        score += 0.7
    if slot in ("R&W - Seller","R&W - Buyer"):
        if _PAT["rw_phrase"].search(tlow): score += 0.6
        role = "seller" if "seller" in slot.lower() else "buyer"
        if role in title: score += 0.5

    if slot == "Governing Law":
        if _PAT["govlaw_strict"].search(tlow):
            score += 0.9
        else:
            score -= 0.6
        if _PAT["conflicts"].search(tlow):
            score += 0.3
        if ("prohibited practices" in tlow) or ("anti-corruption" in tlow):
            score -= 0.4

    if slot == "Purchase and Sale":
        if ("purchase and sale" in title) or _PAT["ps_sell"].search(tlow) or _PAT["ps_take"].search(tlow):
            score += 0.5
        else:
            score -= 0.3

    if slot == "Dispute Resolution":
        seat = (constraints or {}).get("arbitration_seat", "")
        if seat and seat.lower() in tlow:
            score += 0.7
        if any(k in tlow for k in ("lcia","icc","uncitral","siac","hkiac","scc","aaa","jams")):
            score += 0.4
        # generic: if the clause is litigation-heavy and lacks arbitra* cues, nudge down a bit
        if ("court" in tlow or "litigation" in tlow) and ("arbitra" not in tlow):
            score -= 0.4

    # light token density bonus to break rrf ties
    tok_hits = sum(t in tlow for t in (x.lower() for x in _SLOT_TOKENS.get(slot, [])))
    score += min(0.25, 0.03 * tok_hits)
    return score

def _rank_and_filter(slot: str, items: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not items: return items
    coarse = []
    for r in items:
        title = (r.get("title") or "").lower()
        tlow  = (r.get("content") or "").lower()
        tokens = _SLOT_TOKENS.get(slot, [])
        if slot == "Parties":
            ok = ("parties" in title) or bool(_PAT["parties"].search(r.get("content") or ""))
        else:
            ok = any(tok in title for tok in tokens) or any(tok in tlow for tok in tokens)
        if not tokens:
            ok = True
        ctype = (r.get("clause_type") or "")
        want  = _CLAUSE_TYPE_MAP.get(slot)
        if want and any(w.lower() in ctype.lower() for w in want):
            ok = True
        if ok:
            coarse.append(r)
    pool = coarse if coarse else items
    def _tiebreak_key(r: Dict[str, Any]):
        s = _slot_score_generic(slot, r, constraints)
        tlow = (r.get("content") or "").lower()
        title = (r.get("title") or "").lower()
        gate_ok = slot_gate(slot, r.get("content") or "")
        title_exact = _title_matches(slot, title)
        length = len(r.get("content") or "")
        ctype = (r.get("clause_type") or "").lower()
        ctype_hit = 1 if any(w.lower() in ctype for w in (_CLAUSE_TYPE_MAP.get(slot) or [])) else 0
        return (gate_ok, title_exact, ctype_hit, s, float(r.get("rrf") or 0.0), length)
    ranked = sorted(pool, key=_tiebreak_key, reverse=True)
    return ranked

# ----------------- retrieval plan per slot -----------------
def _filters_from_spec(spec: Dict[str, Any], slot: str) -> Dict[str, Any]:
    """Translate spec into retrieval filters/boosts by slot."""
    fat = {"R&W - Seller","R&W - Buyer","Indemnities","Limitations","Notices","Covenants","Termination","Definitions"}
    filters: Dict[str, Any] = {
        "filter_doc_type": spec.get("doc_type"),
        "filter_industry": spec.get("industry"),
        "neighbor_window": 2 if slot in fat else 1,
        "max_defs": 6,
        "expand_sparse": True,
    }
    constraints = (spec.get("constraints") or {})
    # Law is global constraint: enforce where it matters most
    if slot in ("Governing Law", "Dispute Resolution"):
        if constraints.get("law"):
            filters["filter_law"] = constraints["law"]
            filters["enforce_constraints"] = True
        filters["filter_doc_type"] = None
        filters["filter_industry"] = None
    return filters


def _slot_query_and_tokens(slot: str, spec: Dict[str, Any]) -> Tuple[str, List[str], List[str], List[str], Optional[str]]:
    hints = SLOT_HINTS.get(slot, {"queries":[slot.lower()]})
    base_q = " ".join(hints.get("queries", [slot.lower()]))
    must = hints.get("must", [])
    should = hints.get("should", [])
    must_not: List[str] = hints.get("must_not", [])

    # Add constraint-guided tokens for key slots
    if slot == "Dispute Resolution":
        seat = (spec.get("constraints") or {}).get("arbitration_seat")
        if seat:
            should = list(dict.fromkeys(should + [seat]))
        must = list(dict.fromkeys(must + ["arbitr"]))
    if slot == "Governing Law":
        law = (spec.get("constraints") or {}).get("law")
        if law:
            should = list(dict.fromkeys(should + [str(law)]))
        must = list(dict.fromkeys(must + ["govern"]))
    return base_q, must, must_not, should, hints.get("heading_like")


def retrieve_slot(slot: str, spec: Dict[str, Any], top_k: int = 6, pool_n: int = 300, scoped_top_k: int = 3, merge_adjacent: bool = True, merge_cap: int = 2200) -> Dict[str, Any]:
    # Optional LangChain + DSPy orchestrator (global ORCH)
    if ORCH:
        try:
            from adapters.query_types import SlotSpec as _SlotSpec
            sp = _SlotSpec(
                slot=slot,
                doc_type=spec.get("doc_type"),
                industry=spec.get("industry"),
                constraints=spec.get("constraints") or {},
            )
            print(f"[langdspy] slot={slot} (auto queries)")
            fat = {"R&W - Seller","R&W - Buyer","Indemnities","Limitations","Notices","Covenants","Termination","Definitions"}
            per_heading_cap = 2 if slot in fat else 1
            pkg = ORCH.retrieve_slot(sp, pool_n=pool_n, top_k=top_k, per_heading_cap=per_heading_cap)
            # If empty, fall back to legacy path
            if pkg.items:
                # Rank + quality filter then length drop, title-bias, cap + floor
                ranked = _rank_and_filter(slot, pkg.items, spec.get("constraints") or {})
                ranked = _drop_too_short(ranked, min_chars=300)
                ranked = _prefer_title_matches(slot, ranked)
                prelim = ranked[:PER_SLOT_MAX]
                fallback_pool = pkg.items
                capped = _backfill_minimum(slot, prelim, ranked, fallback=fallback_pool)
                # Rebuild sources/defs reflecting new order
                _sources: List[Dict[str, Any]] = []
                _defs: List[Dict[str, Any]] = []
                for r in capped:
                    srcs = r.get("sources") or [{"doc_id": r["document_id"], "section_id": r["section_id"], "title": r.get("title")}]
                    _sources.extend(srcs)
                    for d in (r.get("definitions") or []):
                        _defs.append({"section_id": d["section_id"], "title": d["title"], "content": d["content"], "doc_id": r["document_id"]})
                return {
                    "slot": slot,
                    "query": "AUTO",
                    "must": pkg.debug.get("tokens", {}).get("must", []),
                    "should": pkg.debug.get("tokens", {}).get("should", []),
                    "filters": pkg.debug.get("filters", {}),
                    "items": capped,
                    "sources": _sources,
                    "definitions": _defs,
                    "debug": pkg.debug,
                }
            else:
                print(f"[langdspy] slot={slot} returned 0 items; falling back to legacy hints")
        except Exception as _e:
            print(f"[langdspy] slot={slot} DISABLED (run error): {_e}")
            # Fall through to legacy path

    base_q, must, must_not, should, heading_like = _slot_query_and_tokens(slot, spec)
    filters = _filters_from_spec(spec, slot)
    if heading_like: filters["heading_like"] = heading_like

    # per-heading cap per slot
    fat = {"R&W - Seller","R&W - Buyer","Indemnities","Limitations","Notices","Covenants","Termination","Definitions"}
    per_heading_cap = 2 if slot in fat else 1

    # Pass 1: broad, metadata-aware
    rows = search_hybrid(
        base_q,
        pool_n=pool_n,
        top_k=top_k,
        must_tokens=must or None,
        must_not_tokens=must_not or None,
        should_tokens=should or None,
        sparse_weight=0.58,
        should_weight=0.09,
        per_heading_cap=per_heading_cap,
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
                sparse_weight=0.58,
                should_weight=0.09,
                filter_doc_id=prefer_doc,
                neighbor_window=filters.get("neighbor_window"),
                max_defs=filters.get("max_defs"),
                per_heading_cap=per_heading_cap,
            )

    # Prefer clause hits if present (neighbors/seq), else fall back to sections
    preferred = [r for r in (rows + scoped) if r.get("src") == "C"] or (rows + scoped)
    # Length drop, rank, title-bias
    stitched = _merge_adjacent(preferred, cap_chars=merge_cap) if merge_adjacent else preferred
    stitched = _drop_too_short(stitched, min_chars=300)
    stitched = _rank_and_filter(slot, stitched, spec.get("constraints") or {})
    stitched = _prefer_title_matches(slot, stitched)

    # Post-filter then cap + floor
    prelim = _take_top(stitched, n=min(PER_SLOT_MAX, top_k))
    fallback_pool = sorted((rows + scoped), key=lambda r: (-(len((r.get("content") or ""))), -(float(r.get("rrf") or 0.0))))
    top_items = _backfill_minimum(slot, prelim, stitched, fallback=fallback_pool)
    sources = []
    for r in top_items:
        srcs = r.get("sources") or [{"doc_id": r["document_id"], "section_id": r["section_id"], "title": r.get("title")}]
        sources.extend(srcs)

    # Build a normalized definitions table seed for this slot
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


def _normalize_def_table(defs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract {term, content, doc_id, section_id} from raw definition snippets and de-dup."""
    out: List[Dict[str, Any]] = []
    seen = set()
    import re as _re
    patt = _re.compile(r'"([^\"]{2,80})"\s*(?:means|shall\s+mean)\s+(.+)', _re.IGNORECASE)
    for d in defs:
        text = d.get("content") or ""
        m = patt.search(text)
        term = None
        if m:
            term = m.group(1).strip()
        # fallback to title for term
        if not term:
            title = (d.get("title") or "").strip()
            if title and len(title) <= 80:
                term = title
        if not term:
            continue
        key = (term.lower(), d.get("doc_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "term": term,
            "content": text,
            "doc_id": d.get("doc_id"),
            "section_id": d.get("section_id"),
        })
    return out


def _build_doc_ranking(spec: Dict[str, Any], slots: Dict[str, Any], primary: str, secondaries: List[str]) -> Dict[str, Any]:
    """Produce explainable scores used for Primary/Secondary selection."""
    required = set(spec.get("required", []))
    # Collect per-slot best by doc
    per_slot_scores: Dict[str, Dict[str, float]] = {}
    docs = set()
    for slot, pkg in slots.items():
        scores: Dict[str, float] = {}
        for r in pkg.get("items", []):
            d = r.get("doc_id")
            if not d:
                continue
            docs.add(d)
            s = float(r.get("rrf") or 0.0)
            if s > scores.get(d, 0.0):
                scores[d] = s
        per_slot_scores[slot] = scores
    # Totals
    totals: Dict[str, float] = {d: 0.0 for d in docs}
    for slot, scores in per_slot_scores.items():
        w = 2.0 if slot in required else 1.0
        for d, s in scores.items():
            totals[d] = totals.get(d, 0.0) + w * s
    # Margins where a secondary beats primary
    margins: Dict[str, float] = {}
    for slot, scores in per_slot_scores.items():
        p = scores.get(primary, 0.0)
        best_sec = 0.0
        for d in secondaries:
            best_sec = max(best_sec, scores.get(d, 0.0))
        margins[slot] = round(best_sec - p, 4)
    return {"totals": totals, "per_slot": per_slot_scores, "margins_vs_primary": margins}


def build_scp(spec: Dict[str, Any], include_optional: bool = True, top_k: int = 6, pool_n: int = 300) -> Dict[str, Any]:
    required = spec.get("required", [])
    optional = spec.get("optional", []) if include_optional else []
    # 1) Retrieve per-slot
    slots: Dict[str, Any] = {}
    for slot in required + optional:
        slots[slot] = retrieve_slot(slot, spec, top_k=top_k, pool_n=pool_n)

    # 2) Build a consolidated definitions table (term -> candidate snippets)
    # Note: You can add a real term extractor later; here we just carry the gathered defs.
    def_table_raw: List[Dict[str, Any]] = []
    for s in slots.values():
        def_table_raw.extend(s.get("definitions", []))
    def_table = _normalize_def_table(def_table_raw)

    # 3) Stage-3: choose primary/secondaries and slot decisions
    primary, secondaries, slot_decisions = _compute_primary_and_secondaries(spec, slots)
    doc_ranking = _build_doc_ranking(spec, slots, primary, secondaries)

    # 4) Retrieval params / runtime metadata
    retrieval_params = {
        "per_heading_cap_default": 1,
        "fat_slots_per_heading_cap": {"R&W - Seller": 2, "R&W - Buyer": 2, "Indemnities": 2, "Limitations": 2},
    }
    retrieval_params.update({
        "sparse_weight": 0.58,
        "should_weight": 0.09,
        "pool_n": pool_n,
        "top_k": top_k,
        "use_section_blobs": True,
    })

    return {
        "spec": spec,
        "created_at": int(time.time()),
        "slots": slots,
        "definitions_table": def_table,
        "primary_doc": primary,
        "secondary_docs": secondaries,
        "slot_decisions": slot_decisions,
        "doc_ranking": doc_ranking,
        "retrieval_params": retrieval_params,
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
        "doc_type": None,
        "jurisdiction": None,
        "industry": None,
        "required": ["Parties","Definitions","Purchase and Sale","Price","Adjustments","Closing","CPs",
                     "R&W - Seller","R&W - Buyer","Covenants","Indemnities","Limitations","Governing Law",
                     "Dispute Resolution","Notices","Termination","Miscellaneous"],
        "optional": ["Change of Control","Performance Guarantee","Tax Matters","Environmental","Insurance"],
        "constraints": {},
    }

def _slot_best_by_doc(slot_pkg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return best item per document for a slot (by rrf)."""
    best: Dict[str, Dict[str, Any]] = {}
    for r in slot_pkg.get("items", []):
        doc = r.get("doc_id") or r.get("document_id")
        if not doc:
            continue
        cur = best.get(doc)
        if cur is None or (r.get("rrf") or 0.0) > (cur.get("rrf") or 0.0):
            best[doc] = r
    return best


def _compute_primary_and_secondaries(spec: Dict[str, Any], slots: Dict[str, Any], max_secondaries: int = 2, improve_margin: float = 0.0) -> Tuple[str, List[str], Dict[str, Dict[str, Any]]]:
    """Compute Primary doc, Secondary docs (greedy set-cover), and slot_decisions."""
    required = set(spec.get("required", []))
    optional = set(spec.get("optional", []))

    # Gather candidate docs
    docs = set()
    slot_best: Dict[str, Dict[str, Any]] = {}
    slot_best_by_doc: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for slot, pkg in slots.items():
        best = _slot_best_by_doc(pkg)
        slot_best_by_doc[slot] = best
        for d in best.keys():
            docs.add(d)
    docs = list(docs)

    # Per-doc score across slots
    doc_total: Dict[str, float] = {d: 0.0 for d in docs}
    for slot, best_map in slot_best_by_doc.items():
        w = 2.0 if slot in required else 1.0
        for d, item in best_map.items():
            doc_total[d] += w * float(item.get("rrf") or 0.0)

    if not doc_total:
        return "", [], {}

    primary = max(doc_total.items(), key=lambda x: x[1])[0]

    # Greedy pick secondaries by number of required slots where they beat primary
    secondaries: List[str] = []
    remaining = [d for d in docs if d != primary]
    while len(secondaries) < max_secondaries and remaining:
        best_doc = None
        best_gain = 0
        for d in remaining:
            gain = 0
            for slot in required:
                bm = slot_best_by_doc.get(slot, {})
                d_rrf = float((bm.get(d) or {}).get("rrf") or 0.0)
                p_rrf = float((bm.get(primary) or {}).get("rrf") or 0.0)
                if d_rrf > p_rrf + improve_margin:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_doc = d
        if best_doc is None or best_gain <= 0:
            break
        secondaries.append(best_doc)
        remaining = [d for d in remaining if d != best_doc]

    # Slot-level decisions: prefer primary unless a secondary beats by margin or primary missing
    slot_decisions: Dict[str, Dict[str, Any]] = {}
    override_margin = 0.03
    pool = [primary] + secondaries
    for slot, best_map in slot_best_by_doc.items():
        choice_doc = primary
        reason = None
        p_item = best_map.get(primary)
        p_rrf = float((p_item or {}).get("rrf") or 0.0)
        best_sec = None
        best_sec_rrf = p_rrf
        for d in secondaries:
            rrfd = float((best_map.get(d) or {}).get("rrf") or 0.0)
            if rrfd > best_sec_rrf:
                best_sec_rrf = rrfd
                best_sec = d
        if p_item is None and best_sec is not None:
            choice_doc = best_sec
            reason = f"primary missing; took secondary with rrf={best_sec_rrf:.3f}"
        elif best_sec is not None and (best_sec_rrf - p_rrf) >= override_margin:
            choice_doc = best_sec
            reason = f"beats primary by Δrrf={best_sec_rrf - p_rrf:.2f}"
        sel_item = best_map.get(choice_doc)
        if sel_item:
            slot_decisions[slot] = {
                "doc": choice_doc,
                "section_id": sel_item.get("section_id"),
                "title": sel_item.get("title"),
                "rrf": sel_item.get("rrf"),
            }
            if reason:
                slot_decisions[slot]["reason"] = reason
        else:
            slot_decisions[slot] = {"doc": None, "section_id": None, "reason": "no candidates"}

    return primary, secondaries, slot_decisions


def main():
    ap = argparse.ArgumentParser(description="Build Section Context Protocol (SCP) from retriever")
    ap.add_argument("--spec-file", default=None, help="JSON file with drafting spec; if omitted, uses default")
    ap.add_argument("--top-k", type=int, default=int(os.getenv("SCP_TOP_K", "6")))
    ap.add_argument("--pool-n", type=int, default=int(os.getenv("SCP_POOL_N", "300")))
    ap.add_argument("--no-optional", action="store_true")
    ap.add_argument("--out", default=os.getenv("ARTIFACTS_DIR", "artifacts"))
    ap.add_argument("--parallel", type=int, default=int(os.getenv("SCP_PARALLEL", "0")), help="Parallel slot retrieval (threads); 0=off")
    args = ap.parse_args()

    spec = _default_spec()
    if args.spec_file:
        with open(args.spec_file, "r") as f:
            spec = json.load(f)

    # Optionally run slot retrievals in parallel
    if args.parallel and args.parallel > 0:
        req = spec.get("required", [])
        opt = [] if args.no_optional else spec.get("optional", [])
        slot_list = req + opt
        results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(retrieve_slot, name, spec, args.top_k, args.pool_n): name for name in slot_list}
            for fut in as_completed(futs):
                name = futs[fut]
                results[name] = fut.result()
        scp = {
            "spec": spec,
            "created_at": int(time.time()),
            "slots": results,
            "definitions_table": [d for s in results.values() for d in (s.get("definitions", []))],
            "notes": {
                "neighbor_window": 1,
                "max_defs": 6,
                "fusion": "RRF hybrid",
                "scoped_rerun": True,
                "merged_adjacent": True,
                "parallel": args.parallel,
            }
        }
    else:
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
