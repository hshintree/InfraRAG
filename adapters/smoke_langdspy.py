from __future__ import annotations

import os, json, sys

# Ensure project root on sys.path when running as a script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load .env if present
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

from langchain_openai import ChatOpenAI
from adapters.langdspy_orchestrator import LangDSPyOrchestrator
from adapters.query_types import SlotSpec

if __name__ == "__main__":
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", os.getenv("LANGCHAIN_MODEL", "gpt-4o-mini")), temperature=0)
    orch = LangDSPyOrchestrator(llm, os.environ)

    spec = SlotSpec(
        slot="Governing Law",
        doc_type="Purchase Agreement",
        industry="Power",
        constraints={"filter_law": "NY", "enforce_constraints": True},
    )

    pkg = orch.retrieve_slot(spec, top_k=3)
    print(json.dumps(pkg.debug, indent=2)) 