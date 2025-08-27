# InfraRAG

InfraRAG is a retrieval-augmented drafting system for commercial agreements. It ingests source contracts, builds a hybrid retrieval index over sections/clauses, synthesizes and fuses queries with LangChain/DSPy, and generates contract sections with evidence-grounded drafting using LangGraph. Outputs preserve provenance (per-clause citations), quality scorecards, and configurable constraints (law/seat/industry).

## Overview

- Ingestion and indexing: parse PDFs/XML, normalize, chunk to clauses/sections; store in PostgreSQL with pgvector. Build FTS and vector indexes. Enrich with heading numbers, clause type, defined terms.
- Retrieval: hybrid search (ANN + FTS with RRF) plus token/regex gates; optional reranking via CrossEncoder. Law/seat scoping and doc locks to keep evidence coherent.
- Query synthesis: LangChain blocks (MultiQuery, HyDE, SelfQuery) orchestrated by `LangDSPyOrchestrator`, with guardrails and robust JSON parsing.
- Generation: LangGraph state machine produces per-slot sections from an SCP artifact; extractive-first for Definitions and Parties; evidence-locked drafting with per-section length targets; rubric scoring and one revise pass when required.
- Provenance and QC: SCP package captures items, sources, definitions, queries, and filters; generation produces scorecards per slot.

### System Design Diagram

- Mermaid source: `docs/system_overview.mmd`
- Export with Mermaid CLI:

```bash
npm i -g @mermaid-js/mermaid-cli
mmdc -i docs/system_overview.mmd -o docs/system_overview.png
mmdc -i docs/system_overview.mmd -o docs/system_overview.svg
```

## Repository structure

- `src/` parsers, ingestion, and schema
  - `parsers/pdf_parser.py`, `parsers/xml_parser.py`
  - `ingestion.py`, `indexing.py`, `schema.py`
- `database/` Postgres utilities and CLI tools
  - `database_storage.py`, `database_status.py`, `ingest_to_database.py`, `test_search.py`
- `adapters/` orchestration and retrieval/generation code
  - `retrieval_adapter.py` hybrid search; `lc_pg_retriever.py` bridge
  - `lc_blocks.py` LangChain prompts; `langdspy_orchestrator.py` orchestrator
  - `rerank.py` CrossEncoder reranker
  - `build_scp.py` SCP builder (retrieval packaging)
  - `scp_quality.py` quality checks
  - `generate_graph.py` LangGraph generator with rubric scoring
- `setup/` environment setup
- `notebooks/` quickstart notebooks

## Environment and dependencies

Create a Python environment and install requirements (conda/venv supported). Ensure PostgreSQL with pgvector is available (local Docker compose provided).

Environment variables (via `.env` recommended):

- Database: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`
- Embeddings: `USE_MODAL_EMBED=0` (local default); optional Modal remote embeddings when set to 1
- Hugging Face: `HUGGINGFACEHUB_API_TOKEN`, `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER=1`, optional `HF_HUB_OFFLINE=1`
- Reranker: `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`, optional `RERANKER_WARMUP=1`
- LLM: `LLM_MODEL=gpt-4o-mini`, `LLM_DRAFT_MODEL` (generator), `OPENAI_API_KEY`
- Orchestrator: `USE_LANG_DSPY=1`, `MULTIQUERY_N`, `HYDE_ON`, `SELFQUERY_ON`
- SCP controls: `SCP_PER_SLOT_MAX`, `SCP_PER_SLOT_MIN`, `GEN_PER_SLOT_MAX`
- Generator: `GEN_EVID_N` (evidence items), `GEN_TARGET_CHARS[_CORE|_SECONDARY|_OPTIONAL]`, `MIN_SCORE`, `GEN_BANNED_TERMS`

InfraRAG normalizes HF envs automatically in `build_scp.py` and `langdspy_orchestrator.py`.

## Database setup

- Local Postgres with pgvector: see `docker-compose.local.yml` and `local/sql/init.sql`.
- Verify status:

```bash
python database/database_status.py
```

## Ingestion and indexing

Ingest PDFs/XML into the database. The ingestion pipeline parses, normalizes, extracts headings/defined terms, chunks into clauses, and indexes vectors.

```bash
# Ensure .env configured
python database/ingest_to_database.py --data-dir ./data
```

Key capabilities:
- Adds optional columns (`seq`, `clause_type`, `defined_terms`, `content_hash`, `heading_number`).
- Builds FTS, trigram, and vector indexes.
- Idempotent upserts via content hash.

## Retrieval

Core hybrid search lives in `adapters/retrieval_adapter.py` and supports:
- ANN cosine + FTS weighted fusion (RRF), with token gates (`must`, `should`, `must_not`), per-heading caps, neighbor expansion, definition attachment.
- Law/seat filters applied only to relevant slots (GL/DR). Doc-type/industry scoping is preserved unless an explicit doc lock is present.
- Optional reranking via `CrossEncoder`.

Test utilities:

```bash
python database/test_search.py --query "governing law" --top-k 8
```

## Orchestration (LangChain/DSPy)

`LangDSPyOrchestrator` composes MultiQuery, HyDE (as terms), and SelfQuery into a `QueryBundle`, merges guardrails, validates `heading_like`, and calls the pg retriever. It collects pooled results, deduplicates, and optionally reranks.

Highlights:
- Guardrails tuned for neutrality (no venue bias); regex-normalized SelfQuery output.
- Backstop HyDE only when sparse.
- Keys law slots: GL and DR are gated appropriately, with enforceable constraints when provided.

## SCP builder

`adapters/build_scp.py` builds the Section Context Protocol (SCP), the generator’s input. It supports two paths: orchestrated (LangDSPy) and legacy static hints (fallback).

Features:
- Per-slot ranking using a generic scorer, strict filters for Parties/Definitions, and demotion of off-topic headings (e.g., “Third Parties”).
- Caps and floors per slot; per-heading caps; neighbor window tuning for “fat” slots.
- `gen_items`: extended ranked pool (default up to 10–12) for generation while keeping UI `items` small.
- `doc_lock_ids`: always included (primary + secondaries) and `slot_decisions` (per-slot winning doc) for generation doc-lock.
- Retrieval weights aligned (`sparse_weight=0.58`, `should_weight=0.09`).

Build SCP:

```bash
USE_MODAL_EMBED=0 python adapters/build_scp.py --spec-file specs/<your_spec>.json --parallel 3 --out artifacts
```

Outputs an `artifacts/scp_<ts>.json` with slots, items, gen_items, sources, definitions, doc_lock_ids, slot_decisions, and retrieval metadata.

## Generation (LangGraph)

`adapters/generate_graph.py` compiles a per-slot state graph: prepare → draft → validate → score → (optional revise). It loads the SCP, locks evidence to the chosen doc and `doc_lock_ids`, and produces per-section drafts with rubric scoring.

Key behaviors:
- Evidence selection: uses `gen_items` first, then applies per-slot doc lock and global doc lock; generation-time slot gating (CPs/GL/DR/Parties) to suppress bleed.
- Extractive-first: Definitions copied from evidence; Parties copies the preamble extractively.
- Style profile: detects caps headings and numbering, injects into prompt; optionally includes detected parties.
- Per-section length targets: configurable by bucket (core/secondary/optional) or global; enforced by rubric and a revise pass.
- Rubric: slot-fit (regex cues), evidence overlap, placeholders/refs, format, constraint consistency, dynamic domain drift (evidence-conditioned), source alignment, length vs target. Optional micro-critique.
- Retry: one revise pass triggers if alignment/slot-fit/length are below thresholds or issues present.

Run generation:

```bash
# Example targets for “contract-like” length and breadth
GEN_EVID_N=30 \
GEN_TARGET_CHARS=4500 \
MIN_SCORE=85 \
python adapters/generate_graph.py artifacts/scp_<ts>.json --out artifacts --contract-file
```

Outputs:
- `artifacts/draft_<spec-tag>_<ts>.json` with per-slot drafts, issues, and scorecards
- `artifacts/contract_<spec-tag>_<ts>.txt` assembled text

## Configuration guide

- Evidence pool size: `GEN_EVID_N` (suggest 25–40 for long-form drafting)
- Section length: `GEN_TARGET_CHARS` (global) or buckets `GEN_TARGET_CHARS_CORE`, `GEN_TARGET_CHARS_SECONDARY`, `GEN_TARGET_CHARS_OPTIONAL`
- Banned terms: `GEN_BANNED_TERMS` comma-separated (e.g., LNG-specific tokens for power deals)
- Checklist injection: extend `_checklist_for_spec` or provide a YAML via `RUBRIC_CFG` to feed per-slot expectations
- Reranker: set `RERANKER_MODEL`, ensure HF token; optional `RERANKER_WARMUP=1`

## Quality, scoring, and locking

- Doc locks: `doc_lock_ids` (primary + secondaries) and per-slot `slot_decisions` ensure evidence stays within selected sources.
- GL/DR scoping: doc_type/industry preserved unless locked; law constraints applied only to GL/DR.
- Parties/CPs gates: preamble regex, “Third Parties” demotion, CP phrase enforcement.

## Troubleshooting

- CrossEncoder loading errors (HF 404/429): verify `RERANKER_MODEL` includes the `cross-encoder/` prefix and `HUGGINGFACEHUB_API_TOKEN`. Use `RERANKER_WARMUP=1` and `HF_HOME`. If cached, set `HF_HUB_OFFLINE=1`.
- Modal embedding errors with parallel runs: set `USE_MODAL_EMBED=0`. To avoid fallback to Modal on local error, set `DISABLE_MODAL_FALLBACK=1`.
- Tokenizers fork warning in threaded runs: set `TOKENIZERS_PARALLELISM=false`.
- Off-domain bleed: ensure spec `industry`/`jurisdiction` align with target; use banned terms and doc locks.
- Too short sections: increase `GEN_EVID_N` and section targets; raise `MIN_SCORE` to force revision.

## Roadmap

- Offline DSPy optimization for query synthesis (few-shot exemplars, token seeds)
- Improved extractors for defined terms and parties
- Additional rubric configs per industry and contract family

## Quickstart

1. Ingest contracts

```bash
python database/ingest_to_database.py --data-dir ./data
```

2. Build SCP

```bash
USE_MODAL_EMBED=0 python adapters/build_scp.py --spec-file specs/<spec>.json --parallel 3 --out artifacts
```

3. Generate with targets

```bash
GEN_EVID_N=30 GEN_TARGET_CHARS=4500 MIN_SCORE=85 \
python adapters/generate_graph.py artifacts/scp_<ts>.json --out artifacts --contract-file
```

4. Inspect outputs
- SCP: `artifacts/scp_<ts>.json`
- Drafts: `artifacts/draft_<spec-tag>_<ts>.json`
- Contract: `artifacts/contract_<spec-tag>_<ts>.txt`

If you have any questions feel free to reach out! hshindy@stanford.edu
