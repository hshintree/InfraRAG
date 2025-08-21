from __future__ import annotations

from typing import List, Dict, Any

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # optional dependency


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers not installed; cannot use CrossEncoder reranker")
        self.model = CrossEncoder(model_name, trust_remote_code=True)

    def rerank(self, query: str, rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not rows:
            return rows
        pairs = [(query, (r.get("content") or "")) for r in rows]
        scores = self.model.predict(pairs)
        for r, s in zip(rows, scores):
            r["_ce_score"] = float(s)
        ranked = sorted(rows, key=lambda x: x.get("_ce_score", 0.0), reverse=True)
        return ranked[:top_k] 