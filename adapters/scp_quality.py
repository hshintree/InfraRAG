import json, sys, re
from collections import defaultdict, Counter
from pathlib import Path

FAT_SLOTS = {"R&W - Seller","R&W - Buyer","Indemnities","Limitations","Dispute Resolution","Change of Control"}

# Slot heuristics (very lightweight, adjust freely)
SLOT_RULES = {
    "Parties": {
        "title_should": ["parties","preamble","between"],
        "content_must_any": ["between","party","parties","buyer","seller"],
    },
    "Definitions": {
        "title_should": ["definition"],
        "content_must_any": ["means","shall mean"],
    },
    "Purchase and Sale": {
        "title_should": ["sale","purchase","subject matter","amount of sale"],
        "content_must_any": ["sell","purchase","sale and purchase","subject matter"],
        "must_not_content": ["gas-up","cool-down","tanker","berth"],
    },
    "Governing Law": {
        "title_should": ["governing law"],
        "content_must_any": ["governed by","law"],
        "content_should_any": ["new york"],  # your spec uses NY
    },
    "Dispute Resolution": {
        "title_should": ["dispute","arbitration"],
        "content_must_any": ["arbitration","dispute"],
        "content_should_any": ["seat","lcia","icc","uncitral","london"],
    },
    "Change of Control": {
        "title_should": ["change of control","change in control","assignment"],
        "content_must_any": ["control","assignment","novation","ownership","voting"],
        "must_not_content": ["transfer tax","sales or use tax","vat"],
    },
    # add more as needed…
}

OCR_NOISE = [
    r"[A-Za-z]mTerminationed",   # amended
    r"intTerminations",          # intends
    r"LTerminationers",          # Lenders
    r"sTermination-?out",        # send-out
    r"Terminationing",           # ending
]

def load(path):
    with open(path,"r") as f:
        return json.load(f)

def txt(s): return (s or "").lower()

def contains_any(hay, needles):
    h = txt(hay)
    return any(n in h for n in (needles or []))

def qc_slot(name, pkg, spec):
    errs, warns = [], []
    items = pkg.get("items", [])
    defs  = pkg.get("definitions", [])
    # 1) count gate
    min_items = 2 if name in FAT_SLOTS else 1
    if len(items) < min_items:
        errs.append(f"too few items ({len(items)}/{min_items})")
    # 2) dedupe
    seen = set()
    for it in items:
        key = (it.get("doc_id"), it.get("section_id"))
        if key in seen:
            warns.append(f"duplicate entry {key}")
        seen.add(key)
    # 3) topical heuristics
    rules = SLOT_RULES.get(name, {})
    bad_offtopic = 0
    for it in items:
        title = it.get("title") or ""
        content = it.get("content") or ""
        t_ok = not rules.get("title_should") or contains_any(title, rules.get("title_should"))
        c_ok = not rules.get("content_must_any") or contains_any(content, rules.get("content_must_any"))
        c_bad = rules.get("must_not_content") and contains_any(content, rules.get("must_not_content"))
        if not (t_ok and c_ok) or c_bad:
            bad_offtopic += 1
    if bad_offtopic:
        warns.append(f"{bad_offtopic} potential off-topic items")
    # 4) constraint checks (NY law, London seat hints)
    if name == "Governing Law":
        # must mention governed by + (New York) somewhere in top item content if spec constraint says NY
        cons = (spec.get("constraints") or {})
        if txt(cons.get("law","")) in {"ny","new york","us-ny"}:
            c = txt(items[0]["content"]) if items else ""
            if "governed by" not in c or "new york" not in c:
                warns.append("NY governing law not clearly present in top item")
    if name == "Dispute Resolution":
        cons = (spec.get("constraints") or {})
        if txt(cons.get("arbitration_seat","")) == "london":
            c = txt(items[0]["content"]) if items else ""
            if not any(k in c for k in ["arbitration","lcia","icc","uncitral"]) or "london" not in c:
                warns.append("London/Arbitration cues not clearly present in top item")
    # 5) rrf sanity & content length
    rrfs = [float(it.get("rrf") or 0) for it in items]
    if rrfs:
        if rrfs[0] < 0.02:
            warns.append(f"low top rrf ({rrfs[0]:.3f})")
        if len(rrfs) >= 2 and abs(rrfs[0]-rrfs[1]) < 1e-3:
            warns.append("rrf #1 and #2 nearly tied (weak separation)")
    short = sum(1 for it in items if len((it.get("content") or "")) < 200)
    if short:
        warns.append(f"{short} item(s) have very short content (<200 chars)")
    # 6) defs presence
    if name not in {"Miscellaneous"} and len(defs) == 0:
        warns.append("no definitions collected")
    return errs, warns

def qc_ocr(scp):
    corpus = []
    for sname, pkg in scp.get("slots", {}).items():
        for it in pkg.get("items", []):
            corpus.append(it.get("content") or "")
    text = "\n".join(corpus).lower()
    hits = {}
    for pat in OCR_NOISE:
        cnt = len(re.findall(pat, text))
        if cnt:
            hits[pat] = cnt
    return hits

def doc_score_fallback(scp):
    # For parallel builds (no doc_ranking), synthesize a simple total by summing each slot’s top rrf per doc
    totals = Counter()
    for sname, pkg in scp.get("slots", {}).items():
        best_per_doc = {}
        for it in pkg.get("items", []):
            d = it.get("doc_id"); r = float(it.get("rrf") or 0)
            if d and r > best_per_doc.get(d, 0):
                best_per_doc[d] = r
        for d, r in best_per_doc.items():
            w = 2.0 if sname in scp.get("spec",{}).get("required", []) else 1.0
            totals[d] += w * r
    return totals.most_common(5)

def main():
    if len(sys.argv) != 2:
        print("Usage: python adapters/qc_scp.py <scp.json>")
        sys.exit(2)
    scp = load(sys.argv[1])
    spec = scp.get("spec", {})
    required = spec.get("required", list(scp.get("slots", {}).keys()))
    slots = scp.get("slots", {})
    failures = []
    print(f"SCP QC: {Path(sys.argv[1]).name}")
    print(f"- slots: {len(slots)}; required: {len(required)}")
    # Slot checks
    for name in required:
        pkg = slots.get(name)
        if not pkg:
            failures.append(f"missing required slot: {name}")
            continue
        errs, warns = qc_slot(name, pkg, spec)
        status = "OK"
        if errs: status = "FAIL"
        elif warns: status = "WARN"
        print(f"  · {name:<20} {status}  items={len(pkg.get('items',[]))} defs={len(pkg.get('definitions',[]))}")
        dbg = pkg.get("debug", {})
        qt = dbg.get("queries_tried")
        if qt:
            print(f"      · retriever=LangDSPy, queries_tried={len(qt)}")
        else:
            print(f"      · retriever=legacy/fallback")
        for e in errs:  print(f"      ! {e}")
        for w in warns: print(f"      - {w}")
    # OCR noise
    ocr_hits = qc_ocr(scp)
    if ocr_hits:
        print("  · OCR artifacts detected:")
        for k,v in ocr_hits.items():
            print(f"      - {k}: {v}")
    # Primary doc guess (fallback scorer)
    top_docs = doc_score_fallback(scp)
    if top_docs:
        print("  · Top docs (fallback scoring):")
        for d,score in top_docs:
            print(f"      - {d}: {score:.3f}")
    if failures:
        sys.exit(1)

if __name__ == "__main__":
    main()