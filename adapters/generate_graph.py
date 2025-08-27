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

# Per-section target length buckets (chars)
CORE_SLOTS = {
    "Definitions","Purchase and Sale","Price","Indemnities","Dispute Resolution","Limitations",
}
SECONDARY_SLOTS = {
    "CPs","R&W - Seller","R&W - Buyer","Covenants","Notices","Termination","Governing Law","Closing","Adjustments","Miscellaneous"
}
OPTIONAL_SLOTS = {
    "Change of Control","Performance Guarantee","Tax Matters","Environmental","Insurance"
}

def _target_chars_for_slot(slot: str) -> int:
    # Allow explicit override
    ov = os.getenv("GEN_TARGET_CHARS")
    if ov and ov.isdigit():
        return int(ov)
    # Bucketed defaults; allow broader env overrides per bucket
    core_default = int(os.getenv("GEN_TARGET_CHARS_CORE", "5000"))
    sec_default = int(os.getenv("GEN_TARGET_CHARS_SECONDARY", "3000"))
    opt_default = int(os.getenv("GEN_TARGET_CHARS_OPTIONAL", "2000"))
    if slot in CORE_SLOTS:
        return core_default
    if slot in SECONDARY_SLOTS:
        return sec_default
    if slot in OPTIONAL_SLOTS:
        return opt_default
    return int(os.getenv("GEN_TARGET_CHARS_FALLBACK", "2500"))

# ---------- Rubric config (lightweight, heuristic) ----------
MANDATORY_CUES: Dict[str, List[str]] = RUBRIC_CFG.get("mandatory_cues") or {
    "Parties": ["REGEX:this\s+agreement\s+is\s+made\s+by\s+and\s+(?:between|among)"],
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
    "Governing Law": ["REGEX:governed\s+.*\slaws"],
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

# Banned terms (evidence-agnostic, domain-specific); can be overridden via GEN_BANNED_TERMS
DEFAULT_BANNED = [
    "lng","tanker","driftwood","laytime","vetting","jkm","qatar","shipping",
]

def _banned_terms_for_spec(spec: Dict[str, Any]) -> List[str]:
    env = os.getenv("GEN_BANNED_TERMS")
    if env:
        return [t.strip().lower() for t in env.split(",") if t.strip()]
    # Example: for power industry, ban common LNG/maritime tokens
    if (spec.get("industry") or "").lower() == "power":
        return DEFAULT_BANNED
    return []

# Optional checklists per slot for Power/CO (extendable)
CHECKLISTS_POWER_CO: Dict[str, List[str]] = {
    "Price": ["demand charge","energy charge","market index (hub/LMP)","caps/floors","true-up","change-in-law"],
    "Covenants": ["scheduling","curtailment","outages","balancing","telemetry","transmission and losses"],
    "Indemnities": ["security","ratings triggers","cure periods"],
    "Miscellaneous": ["PUC approvals","FERC/WECC coordination"],
    "Environmental": ["RECs/green attributes ownership or N/A statement"],
    "Governing Law": ["Colorado law","venue","severability","waiver","assignment limits"],
}

def _checklist_for_spec(slot: str, spec: Dict[str, Any]) -> List[str]:
    ind = (spec.get("industry") or "").lower()
    jur = (spec.get("jurisdiction") or "").lower()
    if ind == "power" and jur in {"us-co","co","colorado"}:
        return CHECKLISTS_POWER_CO.get(slot, [])
    return []

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
    target_chars: int = 2500
    banned_terms: List[str] = field(default_factory=list)
    checklist: List[str] = field(default_factory=list)
    parties: Dict[str, Any] = field(default_factory=dict)

# ---------- Utilities ----------

def _collect_evidence(scp: Dict[str,Any], slot: str) -> Tuple[List[str], List[str]]:
    slot_pkg = scp.get("slots", {}).get(slot, {})
    # Prefer richer pool for drafting
    items = slot_pkg.get("gen_items") or slot_pkg.get("items", [])

    # 1) Prefer per-slot winning doc
    chosen_doc = (scp.get("slot_decisions", {}).get(slot) or {}).get("doc")
    if chosen_doc:
        filtered = [r for r in items if (r.get("doc_id") or r.get("document_id")) == chosen_doc]
        if filtered:
            items = filtered

    # 2) Then respect global doc lock (primary + secondaries)
    lock = set(scp.get("doc_lock_ids") or [])
    if lock:
        locked = [r for r in items if (r.get("doc_id") or r.get("document_id")) in lock]
        if locked:
            items = locked

    # Slot-aware gating at generation time (defensive against retrieval noise)
    def _slot_ok(s: str) -> bool:
        s_low = (s or "").lower()
        if slot == "CPs": return ("condition precedent" in s_low) or ("conditions to closing" in s_low)
        if slot == "Governing Law": return ("governed" in s_low and "law" in s_low)
        if slot == "Dispute Resolution": return ("arbitra" in s_low) or ("dispute resolution" in s_low)
        if slot == "Parties":
            return (bool(re.search(r"\bthis agreement\s+is\s+made\s+by\s+and\s+(between|among)\b", s_low)) and "third part" not in s_low)
        return True
    items = [r for r in items if _slot_ok(r.get("title","") + "\n" + (r.get("content") or ""))] or items

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
     "* Produce at least {target_chars} characters for this section; if below target, keep expanding without repetition.\n"
     "* Do not mention these terms unless present in the evidence: {banned_terms}.\n"
     "* Keep defined terms consistent and capitalized.\n"
     "* No duplicated words (e.g., 'Purchase Purchase').\n"
     ),
    ("human",
     "Slot: {slot}\nConstraints: {constraints}\nParties (if detected): {parties}\nChecklist (cover these where applicable): {checklist}\nAllowed terms (soft): {allowed_terms}\nEvidence (snippets):\n---\n{evidence}\n---\nDraft the full section with numbered subsections and sufficient detail.")
])

REV_PROMPT = ChatPromptTemplate.from_messages([
    ("system","Revise and expand to meet the target length while staying faithful to the evidence and constraints."),
    ("human",
     "Slot: {slot}\nConstraints: {constraints}\nTarget length (chars): {target_chars}\nReviewer issues:\n{issues}\nChecklist: {checklist}\nDo not mention: {banned_terms}\nEvidence (snippets):\n---\n{evidence}\n---\nReturn the corrected clause with a single Title line and a fully expanded body.")
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

def rubric_score(slot: str, text: str, evidence: List[str], constraints: Dict[str,Any], industry: str | None, target_chars: int) -> Dict[str, Any]:
    scores: Dict[str, float] = {}
    notes: List[str] = []
    low = text.lower()

    cues = MANDATORY_CUES.get(slot, [])
    hit = 0
    for c in cues:
        if isinstance(c, str) and c.startswith("REGEX:"):
            if re.search(c[6:], low):
                hit += 1
        elif isinstance(c, str) and c.lower() in low:
            hit += 1
    scores["slot_fit"] = 30.0 * (hit / max(1, len(cues)))

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
        if re.search(r"governed\s+.*\slaws", low):
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

    ev_vocab = set(_evidence_vocab(evidence, max_terms=120))
    draft_tokens = [w for w in re.findall(r"[a-z][a-z\-]{2,}", low) if w not in STOP]
    ood_hits = sum(1 for w in draft_tokens if (w not in ev_vocab))
    scores["domain_drift"] = max(0.0, 10.0 - 0.05 * ood_hits)
    if ood_hits > 0:
        notes.append("Possible domain drift relative to evidence")

    align = _align_pct(text, evidence)
    scores["source_alignment"] = round(20.0 * align, 1)
    if align < 0.5:
        notes.append("Many sentences weakly supported by evidence")

    n_chars = len(text.strip())
    if n_chars >= target_chars:
        length_score = 5.0
    elif n_chars >= int(0.8 * target_chars):
        length_score = 2.5
    else:
        length_score = 0.0
    scores["length"] = length_score

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
    # Allow both definition terms and high-frequency evidence terms
    state.allowed_terms = list(dict.fromkeys(terms + _evidence_vocab(ev)))[:80]
    spec = state.scp.get("spec", {}) or {}
    state.constraints = spec.get("constraints", {})
    state.style = _style_profile(ev)
    state.target_chars = _target_chars_for_slot(state.slot)
    state.banned_terms = _banned_terms_for_spec(spec)
    state.checklist = _checklist_for_spec(state.slot, spec)
    # Parties extraction context
    joined = "\n".join(ev)
    m = re.search(r"this agreement\s+is\s+made\s+by\s+and\s+(?:between|among)\s+(.+?)\s+and\s+(.+?)\.", joined, re.I|re.S)
    if m:
        state.parties = {"party_a": m.group(1).strip(), "party_b": m.group(2).strip()}
    return state

_llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
_dspy_drafter = _DSPyDraft() if USE_DSPY else None

def _greedy_compose(evidence: List[str], target_chars: int = 1200) -> str:
    import itertools
    ev = "\n".join(evidence).lower()
    picked: List[str] = []
    used: set[str] = set()
    sents = list(itertools.chain.from_iterable(
        re.split(r"(?<=[.;:])\s+", t) for t in evidence))
    for s in sents:
        key = re.sub(r"\s+"," ", s.strip().lower())
        if len(key) < 40 or key in used:
            continue
        # simple containment or 2-gram overlap proxy
        toks = key.split()
        bi = set(zip(toks, toks[1:])) if len(toks) > 1 else set()
        ev_tokens = ev.split()
        ev_bi = set(zip(ev_tokens, ev_tokens[1:])) if len(ev_tokens) > 1 else set()
        inter = len(bi & ev_bi)
        denom = max(1, len(bi))
        if (key in ev) or ((inter / denom) >= 0.25):
            picked.append(s.strip())
            used.add(key)
        if sum(len(x) for x in picked) > target_chars:
            break
    return "\n".join(picked)

def node_draft(state: GenState) -> GenState:
    # Extract-then-compose for Definitions
    if state.slot == "Definitions":
        joined = "\n".join(state.evidence)
        pat = re.compile(r"([“\"][^”\"]{2,80}[”\"])\s+(means|shall\s+mean)\s+.+?(?:\.\s|\n)", re.I)
        uniq: List[str] = []
        seen: set[str] = set()
        for m in pat.finditer(joined):
            frag = m.group(0).strip()
            key = re.sub(r"\s+", " ", frag.lower().rstrip('.'))
            if key not in seen:
                uniq.append(frag.rstrip(".") + ".")
                seen.add(key)
        if uniq:
            body = "\n".join(uniq[:200])
            title = "Definitions"
            state.draft = SlotDraft(title=title, text=f"{title}\n\n{body}")
            return state

    use_n = int(os.getenv("GEN_EVID_N", "25"))
    ev = "\n---\n".join(state.evidence[:use_n])
    extractive = _greedy_compose(state.evidence[:use_n], target_chars=state.target_chars)
    if extractive and len(extractive) > max(400, int(0.2 * state.target_chars)):
        priming = f"[BEGIN EXTRACTIVE BASIS]\n{extractive}\n[END EXTRACTIVE BASIS]"
        ev = priming + "\n\n" + ev
    if _dspy_drafter:
        text = _dspy_drafter(state.slot, state.constraints, state.allowed_terms, ev)
    else:
        msgs = DRAFT_PROMPT.format_messages(
            slot=state.slot,
            constraints=json.dumps(state.constraints) if state.constraints else "{}",
            parties=json.dumps(getattr(state, "parties", {})) if getattr(state, "parties", None) else "{}",
            allowed_terms=", ".join(state.allowed_terms[:60]),
            evidence=ev,
            target_chars=str(state.target_chars),
            banned_terms=", ".join(state.banned_terms) if state.banned_terms else "(none)",
            checklist=", ".join(state.checklist) if state.checklist else "(none)",
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
    sc = rubric_score(state.slot, state.draft.text, state.evidence, state.constraints, industry, state.target_chars)
    state.draft.scorecard = sc
    return state

def should_retry(state: GenState) -> str:
    sc = state.draft.scorecard or {}
    hard = any(state.draft.issues)
    low_total = sc.get("total", 0) < state.min_score
    weak_ground = (sc.get("subscores", {}).get("source_alignment", 0) < 14.0)
    slot_bad = (sc.get("subscores", {}).get("slot_fit", 0) < 15.0)
    drift = (sc.get("subscores", {}).get("domain_drift", 10) < 7.0)
    # Length gate: require >= 80% of target
    meets_len = len((state.draft.text or "").strip()) >= int(0.8 * state.target_chars)
    low = low_total or weak_ground or slot_bad or drift or (not meets_len)
    return "revise" if (state.retry_count == 0 and (hard or low)) else "ok"


def node_revise(state: GenState) -> GenState:
    # Rebuild a small, slot-gated evidence pack for revision
    gated: List[str] = []
    for t in state.evidence:
        tl = t.lower()
        if state.slot == "CPs" and ("condition precedent" in tl or "conditions to closing" in tl):
            gated.append(t)
        elif state.slot == "Parties" and re.search(r"\bthis agreement\s+is\s+made\s+by\s+and\s+(between|among)\b", tl):
            gated.append(t)
        elif state.slot == "Governing Law" and re.search(r"governed.*laws", tl):
            gated.append(t)
        elif state.slot == "Dispute Resolution" and ("arbitra" in tl or "dispute resolution" in tl):
            gated.append(t)
    use_n = int(os.getenv("GEN_EVID_N", "25"))
    ev = "\n---\n".join((gated or state.evidence)[:use_n])
    msgs = REV_PROMPT.format_messages(
        slot=state.slot,
        constraints=json.dumps(state.constraints) if state.constraints else "{}",
        target_chars=str(state.target_chars),
        issues="\n- " + "\n- ".join(state.draft.issues + (state.draft.scorecard.get("notes") or [])),
        checklist=", ".join(state.checklist) if state.checklist else "(none)",
        banned_terms=", ".join(state.banned_terms) if state.banned_terms else "(none)",
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
    out_state = app.invoke(init)
    draft_obj = _get(out_state, "draft")
    title = _get(draft_obj, "title", slot) if draft_obj else slot
    text = _get(draft_obj, "text", "") if draft_obj else ""
    issues = _get(draft_obj, "issues", []) if draft_obj else []
    scorecard = _get(draft_obj, "scorecard", {}) if draft_obj else {}
    retries = _get(out_state, "retry_count", 0)
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
