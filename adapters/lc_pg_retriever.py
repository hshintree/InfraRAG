from __future__ import annotations

from typing import List, Dict, Any, Optional
from adapters.retrieval_adapter import search_hybrid


def retrieve_with_pg(
    query: str,
    filters: Dict[str, Any],
    tokens: Dict[str, List[str]],
    pool_n: int = 300,
    top_k: int = 6,
    per_heading_cap: int = 1,
) -> List[Dict[str, Any]]:
    return search_hybrid(
        query=query,
        pool_n=pool_n,
        top_k=top_k,
        must_tokens=tokens.get("must_tokens"),
        must_not_tokens=tokens.get("must_not_tokens"),
        should_tokens=tokens.get("should_tokens"),
        heading_like=filters.get("heading_like"),
        filter_doc_type=filters.get("filter_doc_type"),
        filter_industry=filters.get("filter_industry"),
        filter_law=filters.get("filter_law"),
        enforce_constraints=bool(filters.get("enforce_constraints")),
        per_heading_cap=per_heading_cap,
        sparse_weight=0.55,
        should_weight=0.12,
    ) 