from __future__ import annotations

from typing import List, Dict, Any, Callable
from langchain_core.language_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate
import re
import json as _json


def make_multiquery_chain(llm: BaseChatModel, n: int) -> Callable[[str, Dict[str, Any]], List[str]]:
    prompt = ChatPromptTemplate.from_template(
        "Generate {n} short retrieval queries for contract clause mining.\n"
        "Slot: {slot}\nSpec: {spec}\n"
        "- Prefer noun-phrase probes, not questions.\n"
        "- Include synonyms and typical heading phrasings.\n"
        "- 3-6 words each, no punctuation, one per line."
    )

    def run(slot: str, spec: Dict[str, Any]) -> List[str]:
        msgs = prompt.format_messages(slot=slot, spec=spec, n=n)
        out = llm.invoke(msgs)
        text = getattr(out, "content", str(out))
        qs = [q.strip() for q in (text or "").split("\n") if q.strip()]
        return qs[:n]

    return run


def make_hyde_chain(llm: BaseChatModel) -> Callable[[str, Dict[str, Any]], str]:
    prompt = ChatPromptTemplate.from_template(
        "Given the slot and constraints, write a short ideal answer/text we wish to retrieve.\n"
        "Slot: {slot}\nSpec: {spec}\nText:"
    )

    def run(slot: str, spec: Dict[str, Any]) -> str:
        msgs = prompt.format_messages(slot=slot, spec=spec)
        out = llm.invoke(msgs)
        return getattr(out, "content", "").strip()

    return run


def _extract_json(raw: str) -> str:
    if not raw:
        return "{}"
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"(\{.*\})", raw, re.DOTALL)
    return m.group(1) if m else "{}"


def make_selfquery_chain(llm: BaseChatModel) -> Callable[[str, Dict[str, Any]], Dict[str, Any]]:
    prompt = ChatPromptTemplate.from_template(
        "From the slot+spec, infer structured filters and token hints.\n"
        "Return ONLY JSON with keys:\n"
        "  filters (filter_law, filter_industry, filter_doc_type, enforce_constraints, heading_like),\n"
        "  must_tokens, should_tokens, must_not_tokens.\n"
        "Rules: heading_like must be a SQL LIKE string (e.g., '1%', '10.%') or null; never an array.\n"
        "If unknown, use empty arrays/objects; do not add extra keys.\n"
        "Slot: {slot}\nSpec: {spec}\nJSON only:"
    )

    def run(slot: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        msgs = prompt.format_messages(slot=slot, spec=spec)
        out = llm.invoke(msgs)
        raw = getattr(out, "content", "{}")
        try:
            data = _json.loads(_extract_json(raw))
            if not isinstance(data, dict):
                return {}
            # Type coercions and guards
            f = data.get("filters") or {}
            if not isinstance(f, dict):
                f = {}
            hl = f.get("heading_like")
            if not isinstance(hl, str) or not hl.strip():
                f["heading_like"] = None
            f["enforce_constraints"] = bool(f.get("enforce_constraints"))
            data["filters"] = f
            data["_raw"] = raw
            return data
        except Exception:
            return {"_raw": raw}

    return run 