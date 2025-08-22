from __future__ import annotations

import os
from typing import Any, Dict, List
from collections import Counter
import re

from adapters.query_types import SlotSpec, QueryBundle, RetrievalPackage
from adapters.lc_blocks import make_multiquery_chain, make_hyde_chain, make_selfquery_chain
from adapters.lc_pg_retriever import retrieve_with_pg
from adapters.rerank import CrossEncoderReranker

KEY_LAW_SLOTS = {"Governing Law", "Dispute Resolution"}

GUARDRAILS: Dict[str, Dict[str, List[str]]] = {
    "Parties": {
        "should": [
            "Parties","this Agreement",
            "Buyer","Seller","Vendor","Purchaser","Contractor","Employer",
            "Lender","Borrower","Concessionaire","Grantor","Authority"
        ],
    },
    "Purchase and Sale": {
        "should": ["subject matter","purchase","sale","transfer","deliver"],
    },
    "Governing Law": {
        "should": ["governing law","laws of"],
    },
    "Dispute Resolution": {
        "should": ["seat","rules","LCIA","ICC","UNCITRAL","SIAC","HKIAC","SCC","AAA","JAMS"],
    },
    "Definitions": {
        "should": ["means","shall mean","definition"],
    },
}

RERANKER_QUERY: Dict[str, str] = {
    "Parties": "contract clause titled Parties preamble made by and between",
    "Purchase and Sale": "clause titled Purchase and Sale subject matter of the agreement",
    "Governing Law": "clause titled Governing Law governed by the laws of",
    "Dispute Resolution": "arbitration clause dispute resolution seat rules",
}


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, dict):
        return True if x else False
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(x)


class LangDSPyOrchestrator:
    def __init__(self, llm, cfg: Dict[str, Any]):
        self.llm = llm
        self.n = int(cfg.get("MULTIQUERY_N", 4))
        self.use_hyde = bool(int(cfg.get("HYDE_ON", 0)))
        self.use_selfquery = bool(int(cfg.get("SELFQUERY_ON", 1)))
        self.hyde_mode = os.getenv("HYDE_MODE", "").lower()  # "backstop" to enable JIT HyDE
        self.reranker = None
        self._last_selfquery_raw: str | None = None
        self._last_selfquery_obj: Dict[str, Any] | None = None

        # Normalize HF auth/cache for CrossEncoder
        _hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        if _hf_token:
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", _hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _hf_token)
            os.environ.setdefault("HF_TOKEN", _hf_token)
        _hf_home = os.getenv("HF_HOME")
        if _hf_home:
            os.environ["HF_HOME"] = os.path.expanduser(_hf_home)
        _hf_transfer = os.getenv("HF_HUB_ENABLE_HF_TRANSFER")
        if _hf_transfer:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = _hf_transfer

        reranker_model = os.getenv("RERANKER_MODEL")
        if reranker_model:
            try:
                # Encourage offline cache usage if available
                os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
                os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
                self.reranker = CrossEncoderReranker(reranker_model)
                # Optional warmup to ensure model is downloaded and usable
                if os.getenv("RERANKER_WARMUP", "0").lower() in ("1","true","yes"):
                    _ = self.reranker.rerank("warmup", [{"content": "warmup"}], top_k=1)
                print(f"[reranker] ready: {reranker_model}")
            except Exception as e:
                print(f"[reranker] DISABLED (load error): {e}")
                self.reranker = None

        self.mq = make_multiquery_chain(llm, self.n)
        self.hyde = make_hyde_chain(llm)
        self.sq = make_selfquery_chain(llm)

    def _sanitize_queries(self, qs: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen: set[str] = set()
        for q in qs or []:
            q2 = (q or "").strip()
            if q2.startswith(("- ", "* ")):
                q2 = q2[2:].strip()
            q2 = q2.strip("\"' ").rstrip(".,;:")
            if len(q2) >= 3 and q2.lower() not in seen:
                seen.add(q2.lower())
                cleaned.append(q2)
        return cleaned

    def _top_terms(self, txt: str, k: int = 12) -> List[str]:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", txt or "")
        stop = {"the","and","for","with","that","shall","hereby","herein","agreement","party","parties","of","to","in","by","as","on"}
        terms = [w.lower() for w in words if w.lower() not in stop]
        common = [w for w, _ in Counter(terms).most_common(k)]
        return common

    def _default_heading_like(self, slot: str) -> str | None:
        mapping = {
            "Definitions": "1%",
        }
        return mapping.get(slot)

    def _validate_like(self, val: Any) -> str | None:
        if not isinstance(val, str):
            return None
        s = val.strip()
        if re.match(r"^\d+(?:\.\d+)*%?$", s):
            return s if s.endswith('%') or '.' in s else (s + '%')
        return None

    def _valid_law_code(self, val: Any) -> str | None:
        if not val:
            return None
        s = str(val).strip().upper()
        import re as _re
        return s if _re.match(r"^[A-Z]{2}(-[A-Z]{2})?$", s) else None

    def _make_hyde_queries(self, hyde_terms: List[str], n: int = 2, width: int = 4) -> List[str]:
        out: List[str] = []
        if not hyde_terms:
            return out
        for i in range(0, min(len(hyde_terms), n * width), width):
            chunk = hyde_terms[i:i+width]
            if chunk:
                out.append(" ".join(chunk))
        return out

    def synthesize(self, spec: SlotSpec) -> QueryBundle:
        base = self.mq(spec.slot, spec.__dict__)
        hyde_txt = self.hyde(spec.slot, spec.__dict__) if self.use_hyde else None
        hyde_terms: List[str] = self._top_terms(hyde_txt, k=12) if hyde_txt else []

        sq_out: Dict[str, Any] = {"filters": {}, "must_tokens": [], "should_tokens": [], "must_not_tokens": []}
        if self.use_selfquery:
            try:
                sq_out = self.sq(spec.slot, spec.__dict__) or sq_out
            except Exception:
                pass
        self._last_selfquery_raw = sq_out.get("_raw") if isinstance(sq_out, dict) else None
        if isinstance(sq_out, dict):
            self._last_selfquery_obj = {k: v for k, v in sq_out.items() if k != "_raw"}
        else:
            self._last_selfquery_obj = None

        sq_filters = (sq_out.get("filters") or {}).copy()
        if "enforce_constraints" in sq_filters:
            sq_filters["enforce_constraints"] = _as_bool(sq_filters["enforce_constraints"])

        allow_law = spec.slot in KEY_LAW_SLOTS
        filters: Dict[str, Any] = {
            "filter_doc_type": None if allow_law else spec.doc_type,
            "filter_industry": None if allow_law else spec.industry,
        }
        if allow_law:
            law = self._valid_law_code(sq_filters.get("filter_law"))
            if law:
                filters["filter_law"] = law
                filters["enforce_constraints"] = _as_bool(sq_filters.get("enforce_constraints", True))

        heading_like = sq_filters.get("heading_like") or spec.slot
        hl = self._validate_like(heading_like)
        if not hl:
            hl = self._default_heading_like(spec.slot)
        filters["heading_like"] = hl

        base = self._sanitize_queries(base)
        base_queries = [q for q in base if q]
        base_queries = list(dict.fromkeys(base_queries))

        should_combined: List[str] = []
        seen_tok: set[str] = set()
        guard = GUARDRAILS.get(spec.slot, {})
        combined_should = (sq_out.get("should_tokens", []) or []) + hyde_terms + guard.get("should", [])
        for t in combined_should:
            tt = (t or "").strip()
            if not tt:
                continue
            low = tt.lower()
            if low in seen_tok:
                continue
            seen_tok.add(low)
            should_combined.append(tt)
        must_combined: List[str] = []
        seen_m: set[str] = set()
        combined_must = (sq_out.get("must_tokens", []) or []) + guard.get("must", [])
        for t in combined_must:
            tt = (t or "").strip()
            if not tt:
                continue
            low = tt.lower()
            if low in seen_m:
                continue
            seen_m.add(low)
            must_combined.append(tt)

        return QueryBundle(
            base_queries=base_queries,
            must_tokens=must_combined,
            should_tokens=should_combined,
            must_not_tokens=sq_out.get("must_not_tokens", []),
            heading_like=filters.get("heading_like"),
            filters=filters,
        )

    def retrieve_slot(self, spec: SlotSpec, pool_n: int = 300, top_k: int = 6, per_heading_cap: int = 1) -> RetrievalPackage:
        qb = self.synthesize(spec)
        pooled: List[Dict[str, Any]] = []
        tried: List[str] = []
        for q in qb.base_queries:
            rows = retrieve_with_pg(
                query=q,
                filters={**(qb.filters or {}), "heading_like": qb.heading_like},
                tokens={
                    "must_tokens": qb.must_tokens,
                    "should_tokens": qb.should_tokens,
                    "must_not_tokens": qb.must_not_tokens,
                },
                pool_n=pool_n,
                top_k=top_k * 3,
                per_heading_cap=per_heading_cap,
            )
            pooled.extend(rows)
            tried.append(q)

        seen = set(); dedup: List[Dict[str, Any]] = []
        for r in pooled:
            k = (r.get("document_id"), r.get("section_id"))
            if k in seen:
                continue
            seen.add(k); dedup.append(r)

        if (self.hyde_mode == "backstop" or os.getenv("HYDE_BACKSTOP", "0") in {"1","true","True"}) and len(dedup) < 2:
            hyde_txt = self.hyde(spec.slot, spec.__dict__)
            hyde_terms = self._top_terms(hyde_txt, k=12)
            hyde_queries = self._make_hyde_queries(hyde_terms, n=3, width=4)
            for q in hyde_queries:
                rows = retrieve_with_pg(
                    query=q,
                    filters={**(qb.filters or {}), "heading_like": qb.heading_like},
                    tokens={
                        "must_tokens": qb.must_tokens,
                        "should_tokens": list(dict.fromkeys((qb.should_tokens or []) + hyde_terms)),
                        "must_not_tokens": qb.must_not_tokens,
                    },
                    pool_n=pool_n,
                    top_k=top_k * 3,
                    per_heading_cap=per_heading_cap,
                )
                pooled.extend(rows); tried.append(q)
            seen = set(); dedup = []
            for r in pooled:
                k = (r.get("document_id"), r.get("section_id"))
                if k in seen: continue
                seen.add(k); dedup.append(r)

        ranked = dedup
        if self.reranker and dedup:
            rr_query = RERANKER_QUERY.get(spec.slot, spec.slot)
            ranked = self.reranker.rerank(rr_query, dedup, top_k=top_k)

        items = ranked[:top_k]
        sources: List[Dict[str, Any]] = []
        defs: List[Dict[str, Any]] = []
        for r in items:
            srcs = r.get("sources") or [{"doc_id": r["document_id"], "section_id": r["section_id"], "title": r.get("title")}]
            sources.extend(srcs)
            for d in (r.get("definitions") or []):
                defs.append({"section_id": d["section_id"], "title": d["title"], "content": d["content"], "doc_id": r["document_id"]})

        dbg: Dict[str, Any] = {
            "queries_tried": tried,
            "tokens": {"must": qb.must_tokens, "should": qb.should_tokens, "must_not": qb.must_not_tokens},
            "filters": qb.filters,
        }
        if self._last_selfquery_obj is not None:
            dbg["selfquery"] = self._last_selfquery_obj

        return RetrievalPackage(
            slot=spec.slot,
            items=items,
            sources=sources,
            definitions=defs,
            debug=dbg,
        ) 