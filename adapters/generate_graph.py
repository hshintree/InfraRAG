from __future__ import annotations

"""
Generation graph with lightweight per-slot rubric scoring.

Usage:
  python adapters/generate_graph.py artifacts/scp_<timestamp>.json --out artifacts --contract-file --min-score 75

Env:
  LLM_MODEL=gpt-4o-mini (default)
  TEMPERATURE=0.0 (default)
  USE_DSPY=0/1  (drafting via DSPy teleprompted program)
  RUBRIC_LLM=0/1 (optional tiny LLM critique appended in scorecard)

Inputs:
  - SCP JSON produced by adapters/build_scp.py
Outputs:
  - artifacts/draft_<ts>.json (per-slot drafts + scorecards)
  - artifacts/contract_<ts>.txt (assembled contract text) when --contract-file

Notes:
  - Keeps your earlier LangChain/DSPy toggles.
  - Adds rubric scoring node after validate, with a single retry when below threshold or hard issues.
"""

import argparse, json, os, time, re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

# Load .env so OPENAI_API_KEY and other tokens are available
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# --- LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Optional DSPy (teleprompted), guarded by env
USE_DSPY = os.getenv("USE_DSPY", "0").lower() in ("1","true")
if USE_DSPY:
    try:
        import dspy
    except Exception:
        USE_DSPY = False

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
RUBRIC_LLM = os.getenv("RUBRIC_LLM", "0").lower() in ("1","true")

# Optional external rubric config
try:
    import yaml  # type: ignore
    def _load_rubric_cfg() -> Dict[str, Any]:
        path = os.getenv("RUBRIC_CFG", "configs/rubric.yaml")
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    RUBRIC_CFG = _load_rubric_cfg()
except Exception:
    RUBRIC_CFG = {}

# ---------- Rubric config (lightweight, heuristic) ----------
MANDATORY_CUES: Dict[str, List[str]] = RUBRIC_CFG.get("mandatory_cues") or {
    "Parties": ["by and between", "Parties"],
    "Definitions": ["means"],
    "Purchase and Sale": ["sell", "purchase"],
    "Price": ["price"],
    "Adjustments": ["adjust"],
    "Closing": ["Closing"],
    "CPs": ["conditions precedent"],
    "R&W - Seller": ["represents", "warrants"],
    "R&W - Buyer": ["represents", "warrants"],
    "Covenants": ["shall"],
    "Indemnities": ["indemnify", "hold harmless"],
    "Limitations": ["limitation of liability", "consequential"],
    "Governing Law": ["governed by the laws"],
    "Dispute Resolution": ["arbitration"],
    "Notices": ["notice"],
    "Termination": ["terminate"],
    "Miscellaneous": ["entire agreement"],
}

INSTITUTIONS = RUBRIC_CFG.get("arbitration_institutions") or ["LCIA","ICC","UNCITRAL","SIAC","HKIAC","SCC","AAA","ICDR","JAMS"]
STOP = {
    "the","and","for","with","that","shall","hereby","herein","agreement","party","parties",
    "of","to","in","by","as","on","or","any","all","be","is","are","this","section",
}

# ---------- Graph state ----------
@dataclass
class SlotDraft:
    title: str
    text: str
    issues: List[str] = field(default_factory=list)
    scorecard: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenState:
    scp: Dict[str, Any]
    slot: str
    evidence: List[str] = field(default_factory=list)
    allowed_terms: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    draft: SlotDraft | None = None
    retry_count: int = 0
    min_score: int = 75
    style: Dict[str, str] = field(default_factory=dict)

# ---------- Utilities ----------

def _collect_evidence(scp: Dict[str,Any], slot: str) -> Tuple[List[str], List[str]]:
    slot_pkg = scp.get("slots", {}).get(slot, {})
    items = slot_pkg.get("items", [])
    lock = set(scp.get("doc_lock_ids") or [])
    if lock:
        locked = [r for r in items if (r.get("doc_id") or r.get("document_id")) in lock]
        items = locked or items
    texts = [(r.get("title") or "") + "\n" + (r.get("content") or "") for r in items]
    defs_table = scp.get("definitions_table") or []
    terms = []
    for d in defs_table:
        t = (d.get("term") or "").strip()
        if t and t not in terms:
            terms.append(t)
    return texts, terms

def _evidence_vocab(evidence: List[str], max_terms: int = 60) -> List[str]:
    joined = "\n".join(evidence).lower()
    words = re.findall(r"[a-z][a-z\-]{2,}", joined)
    freq: Dict[str, int] = {}
    for w in words:
        if w in STOP:
            continue
        freq[w] = freq.get(w,0)+1
    return [w for w,_ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))][:max_terms]

def _style_profile(evidence: List[str]) -> Dict[str, str]:
    joined = "\n".join(evidence)
    caps = bool(re.search(r"\n[A-Z0-9 ,&\-]{6,}\n", joined))
    has_1 = bool(re.search(r"\n\s*1\.\s+[A-Za-z]", joined))
    has_1_1 = bool(re.search(r"\n\s*1\.\d+\s+[A-Za-z]", joined))
    pattern = "1." if has_1 and not has_1_1 else ("1.1" if has_1_1 else "none")
    return {"caps_headings": ("true" if caps else "false"), "numbering": pattern}

# ---------- Drafting ----------

DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You draft precise, conservative clauses. Adhere strictly to the provided slot, constraints, and evidence.\n"
     "* Use only concepts that are reasonably supported by the evidence; avoid domain drift.\n"
     "* Prefer compact, single-heading output: Title on first line, then body.\n"
     "* Match the source style: caps_headings={caps_headings}, numbering={numbering}.\n"
     "* No placeholders like [TBD] or 'as defined in Section X' unless present in evidence.\n"
     "* Keep defined terms consistent and capitalized.\n"
     "* No duplicated words (e.g., 'Purchase Purchase').\n"
     ),
    ("human",
     "Slot: {slot}\nConstraints: {constraints}\nAllowed terms (soft): {allowed_terms}\nEvidence (snippets):\n---\n{evidence}\n---\nPlease draft the clause. Start with a single Title line, then the clause body. Do not include citations or footnotes.")
])

REV_PROMPT = ChatPromptTemplate.from_messages([
    ("system","Revise the clause to address reviewer issues while staying faithful to the evidence."),
    ("human",
     "Slot: {slot}\nConstraints: {constraints}\nReviewer issues:\n{issues}\nEvidence (snippets):\n---\n{evidence}\n---\nReturn the corrected clause with a single Title line and body. Avoid placeholders and domain drift.")
])

# Optional DSPy teleprompted program
class _DSPyDraft:
    def __init__(self):
        # Minimal teleprompting: define signature with fields we care about
        class DraftSig(dspy.Signature):
            """Draft a contract clause from slot, constraints, and evidence."""
            slot = dspy.InputField()
            constraints = dspy.InputField()
            allowed_terms = dspy.InputField()
            evidence = dspy.InputField()
            out = dspy.OutputField(desc="Title on first line, then the body")
        self.program = dspy.ChainOfThought(DraftSig)

    def __call__(self, slot: str, constraints: Dict[str,Any], allowed_terms: List[str], evidence: str) -> str:
        rsp = self.program(slot=slot, constraints=json.dumps(constraints),
                           allowed_terms=", ".join(allowed_terms), evidence=evidence)
        return rsp.out

# ---------- Validator ----------

def validate(slot: str, text: str, constraints: Dict[str,Any]) -> List[str]:
    issues: List[str] = []
    if re.compile(r"^(?P<t>.+?)\n\s*\1\s*$", re.S).match(text.strip()):
        issues.append("Title duplicated above body; keep a single title line")
    if re.compile(r"\[(?:TBD|XX|YY|INSERT|FILL IN)[^\]]*\]|\?\?\?", re.I).search(text):
        issues.append("Contains placeholder(s) like [TBD]/??? — replace with concrete terms")
    if re.compile(r"\b(\w+)\s+\1\b", re.I).search(text):
        issues.append("Duplicated consecutive word detected (e.g., 'Purchase Purchase')")
    low = text.lower()
    if slot == "Governing Law":
        if "governed" not in low or "laws" not in low:
            issues.append("Governing Law should contain 'governed by the laws' formulation")
    if slot == "Dispute Resolution":
        if "arbitra" not in low:
            issues.append("Dispute Resolution should specify arbitration explicitly")
        if not any(inst.lower() in low for inst in [i.lower() for i in INSTITUTIONS]):
            issues.append("Arbitration institution not recognized (e.g., LCIA, ICC, UNCITRAL, SIAC, HKIAC, SCC, AAA/ICDR, JAMS)")
    seat = (constraints or {}).get("arbitration_seat")
    if seat and slot == "Dispute Resolution" and seat.lower() not in low:
        issues.append(f"Seat '{seat}' not referenced")
    return issues

# ---------- Rubric scorer ----------

def _align_pct(text: str, evidence: List[str]) -> float:
    if not text.strip():
        return 0.0
    ev = "\n".join(evidence).lower()
    ev_tokens = ev.split()
    ev_bi = set(zip(ev_tokens, ev_tokens[1:])) if len(ev_tokens) > 1 else set()
    sents = [s.strip() for s in re.split(r"[\.\n];?\s+", text) if len(s.strip()) > 0]
    if not sents:
        return 0.0
    good = 0
    for s in sents:
        low = s.lower()
        toks = low.split()
        bi = set(zip(toks, toks[1:])) if len(toks) > 1 else set()
        denom = max(1, len(bi))
        inter = len(bi & ev_bi)
        if (inter / denom) >= 0.15:
            good += 1
    return good / max(1, len(sents))

def rubric_score(slot: str, text: str, evidence: List[str], constraints: Dict[str,Any], industry: str | None) -> Dict[str, Any]:
    scores: Dict[str, float] = {}
    notes: List[str] = []
    low = text.lower()

    cues = MANDATORY_CUES.get(slot, [])
    hit = sum(1 for c in cues if c.lower() in low)
    scores["slot_fit"] = 20.0 * (hit / max(1, len(cues)))

    vocab = _evidence_vocab(evidence)
    ev_set = set(vocab)
    draft_terms = set(re.findall(r"[a-z][a-z\-]{2,}", low))
    overlap = len(draft_terms & ev_set)
    denom = max(8, len(ev_set))
    scores["evidence_overlap"] = 20.0 * (overlap / denom)
    if scores["evidence_overlap"] < 8:
        notes.append("Low evidence-term overlap")

    ph = len(re.findall(r"\[(?:TBD|XX|YY|INSERT|FILL IN)[^\]]*\]|\?\?\?", text, flags=re.I))
    ref_bad = len(re.findall(r"as defined in section\s+\d", low))
    scores["placeholders"] = max(0.0, 15.0 - 7.5*ph - 5.0*ref_bad)

    dup = 1 if re.search(r"\b(\w+)\s+\1\b", text, flags=re.I) else 0
    title_dup = 1 if re.match(r"^(?P<t>.+?)\n\s*\1\s*$", text.strip(), flags=re.S) else 0
    scores["format"] = max(0.0, 10.0 - 5.0*dup - 5.0*title_dup)

    cons = 0.0
    if slot == "Governing Law":
        if "governed" in low and "laws" in low:
            cons += 10
        if constraints.get("law") and str(constraints["law"]).lower() in low:
            cons += 5
    elif slot == "Dispute Resolution":
        if "arbitra" in low:
            cons += 6
        if any(inst.lower() in low for inst in [i.lower() for i in INSTITUTIONS]):
            cons += 6
        seat = (constraints or {}).get("arbitration_seat")
        if seat and seat.lower() in low:
            cons += 3
    else:
        cons += 12
    scores["constraints"] = min(15.0, cons)

    # Dynamic drift: penalize tokens not in evidence vocab
    ev_vocab = set(_evidence_vocab(evidence, max_terms=120))
    draft_tokens = [w for w in re.findall(r"[a-z][a-z\-]{2,}", low) if w not in STOP]
    ood_hits = sum(1 for w in draft_tokens if (w not in ev_vocab))
    scores["domain_drift"] = max(0.0, 10.0 - 0.05 * ood_hits)
    if ood_hits > 0:
        notes.append("Possible domain drift relative to evidence")

    # Source alignment score (extractive overlap proxy)
    align = _align_pct(text, evidence)
    scores["source_alignment"] = round(20.0 * align, 1)
    if align < 0.5:
        notes.append("Many sentences weakly supported by evidence")

    n_chars = len(text.strip())
    scores["length"] = 5.0 if 200 <= n_chars <= 4000 else 3.0

    total = round(sum(scores.values()), 1)

    critique = None
    if RUBRIC_LLM:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        prompt = ChatPromptTemplate.from_template(
            "Draft (slot={slot}) to critique:\n---\n{text}\n---\nIn 3 bullets, list the most material legal/consistency issues and any domain drift.")
        critique = llm.invoke(prompt.format_messages(slot=slot, text=text)).content

    return {
        "subscores": {k: round(v,1) for k,v in scores.items()},
        "total": total,
        "notes": notes,
        "critique": critique,
    }

# ---------- Graph nodes ----------

def node_prepare(state: GenState) -> GenState:
    ev, terms = _collect_evidence(state.scp, state.slot)
    state.evidence = ev
    state.allowed_terms = list(dict.fromkeys(terms + _evidence_vocab(ev)))[:80]
    state.constraints = (state.scp.get("spec", {}) or {}).get("constraints", {})
    state.style = _style_profile(ev)
    return state

_llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
_dspy_drafter = _DSPyDraft() if USE_DSPY else None

def node_draft(state: GenState) -> GenState:
    # Extract-then-compose for Definitions
    if state.slot == "Definitions":
        joined = "\n".join(state.evidence)
        pat = re.compile(r"([“\"][^”\"]{2,80}[”\"])\s+(means|shall\s+mean)\s+.+?(?:\.\s|\n)", re.I)
        uniq: List[str] = []
        seen: set[str] = set()
        for m in pat.finditer(joined):
            frag = m.group(0).strip()
            key = frag.lower().rstrip('.')
            if key not in seen:
                uniq.append(frag.rstrip(".") + ".")
                seen.add(key)
        if uniq:
            body = "\n".join(uniq[:100])
            title = "Definitions"
            state.draft = SlotDraft(title=title, text=f"{title}\n\n{body}")
            return state

    ev = "\n---\n".join(state.evidence[:3])
    if _dspy_drafter:
        text = _dspy_drafter(state.slot, state.constraints, state.allowed_terms, ev)
    else:
        msgs = DRAFT_PROMPT.format_messages(
            slot=state.slot,
            constraints=json.dumps(state.constraints) if state.constraints else "{}",
            allowed_terms=", ".join(state.allowed_terms[:40]),
            evidence=ev,
            **(state.style or {}),
        )
        text = _llm.invoke(msgs).content
    lines = [l.rstrip() for l in (text or "").strip().splitlines() if l.strip()]
    if not lines:
        title, body = state.slot, "[Draft empty]"
    else:
        title = lines[0]
        body = "\n".join(lines[1:]).strip()
        if not body:
            body = "[Draft empty]"
    state.draft = SlotDraft(title=title, text=f"{title}\n\n{body}")
    return state

def node_validate(state: GenState) -> GenState:
    issues = validate(state.slot, state.draft.text, state.constraints)
    state.draft.issues = issues
    return state

def node_score(state: GenState) -> GenState:
    industry = (state.scp.get("spec", {}) or {}).get("industry")
    sc = rubric_score(state.slot, state.draft.text, state.evidence, state.constraints, industry)
    state.draft.scorecard = sc
    return state

def should_retry(state: GenState) -> str:
    sc = state.draft.scorecard or {}
    hard = any(state.draft.issues)
    low_total = sc.get("total", 0) < state.min_score
    weak_ground = (sc.get("subscores", {}).get("source_alignment", 0) < 10.0)
    drift = (sc.get("subscores", {}).get("domain_drift", 10) < 6.0)
    low = low_total or weak_ground or drift
    return "revise" if (state.retry_count == 0 and (hard or low)) else "ok"


def node_revise(state: GenState) -> GenState:
    ev = "\n---\n".join(state.evidence[:3])
    msgs = REV_PROMPT.format_messages(
        slot=state.slot,
        constraints=json.dumps(state.constraints) if state.constraints else "{}",
        issues="\n- " + "\n- ".join(state.draft.issues + (state.draft.scorecard.get("notes") or [])),
        evidence=ev,
    )
    text = _llm.invoke(msgs).content
    lines = [l.rstrip() for l in (text or "").strip().splitlines() if l.strip()]
    if not lines:
        title, body = state.slot, "[Draft empty]"
    else:
        title = lines[0]
        body = "\n".join(lines[1:]).strip()
        if not body:
            body = "[Draft empty]"
    state.draft = SlotDraft(title=title, text=f"{title}\n\n{body}")
    state.retry_count += 1
    return state

# ---------- Runner ----------

def run_for_slot(scp: Dict[str,Any], slot: str, min_score: int) -> Dict[str,Any]:
    graph = StateGraph(GenState)
    graph.add_node("prepare", node_prepare)
    graph.add_node("draft", node_draft)
    graph.add_node("validate", node_validate)
    graph.add_node("score", node_score)
    graph.add_node("revise", node_revise)

    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "draft")
    graph.add_edge("draft", "validate")
    graph.add_edge("validate", "score")
    graph.add_conditional_edges("score", should_retry, {"revise": "revise", "ok": END})
    graph.add_edge("revise", "validate")

    app = graph.compile()
    init = GenState(scp=scp, slot=slot, min_score=min_score)
    # LangGraph returns a plain dict state; handle both dict and dataclass
    def _get(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
    out = app.invoke(init)
    draft_obj = _get(out, "draft")
    title = _get(draft_obj, "title", slot) if draft_obj else slot
    text = _get(draft_obj, "text", "") if draft_obj else ""
    issues = _get(draft_obj, "issues", []) if draft_obj else []
    scorecard = _get(draft_obj, "scorecard", {}) if draft_obj else {}
    retries = _get(out, "retry_count", 0)
    return {
        "title": title,
        "text": text,
        "issues": issues,
        "scorecard": scorecard,
        "retries": retries,
    }


def assemble_contract(drafts: Dict[str,Any]) -> str:
    order = list(drafts.keys())
    parts: List[str] = []
    for slot in order:
        parts.append(drafts[slot]["text"].rstrip())
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scp_file")
    ap.add_argument("--out", default=os.getenv("ARTIFACTS_DIR", "artifacts"))
    ap.add_argument("--contract-file", action="store_true")
    ap.add_argument("--min-score", type=int, default=int(os.getenv("MIN_SCORE", "75")))
    args = ap.parse_args()

    with open(args.scp_file, "r") as f:
        scp = json.load(f)

    req = scp.get("spec", {}).get("required", [])
    opt = scp.get("spec", {}).get("optional", [])
    slots = req + opt

    drafts: Dict[str,Any] = {}
    for slot in slots:
        drafts[slot] = run_for_slot(scp, slot, args.min_score)

    ts = int(time.time())
    os.makedirs(args.out, exist_ok=True)
    def _slug(s: Any) -> str:
        return re.sub(r"[^A-Za-z0-9]+","-", str(s or "")).strip("-").lower()
    spec = scp.get("spec", {})
    tag = "-".join(filter(None, [_slug(spec.get("doc_type")), _slug(spec.get("jurisdiction")), _slug(spec.get("industry"))])) or "generic"
    jpath = os.path.join(args.out, f"draft_{tag}_{ts}.json")
    with open(jpath, "w") as f:
        json.dump({
            "from_scp": os.path.basename(args.scp_file),
            "model": LLM_MODEL,
            "temperature": TEMPERATURE,
            "use_dspy": USE_DSPY,
            "drafts": drafts,
        }, f, ensure_ascii=False, indent=2)

    print(f"Wrote {jpath}")

    if args.contract_file:
        txt = assemble_contract(drafts)
        tpath = os.path.join(args.out, f"contract_{tag}_{ts}.txt")
        with open(tpath, "w") as f:
            f.write(txt)
        print(f"Wrote {tpath}")


if __name__ == "__main__":
    # Quiet tokenizer fork warnings if CrossEncoder warmed earlier
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
