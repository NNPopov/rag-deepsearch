# rag-deepsearch

A **RAG pipeline** that turns a book (PDF) into a populated Postgres/ParadeDB index and answers
questions over it with a hybrid-search deep-search loop. Built as a uv workspace following a
strict pipes-and-filters architecture (each filter reads files, writes files, holds no shared
state). Test corpus: *Designing Data-Intensive Applications* and *Acing the System Design Interview*.

> Working rules and the authoritative design live in `CLAUDE.md` and the `docs/Rag_*.md` docs (see below).
> Originally bootstrapped from the `book-to-skill` skill; that scaffold has been removed.

## Architecture

Two halves over shared libraries, plus a transport layer:

```
ingest:  PDF → Extract → Structure → (Clean) → Chunk → Embed → Persist → ParadeDB
rag:     question → PLAN → SEARCH(vector+BM25 → RRF → MMR → small2big) → REFLECT → … → SYNTH
serve:   FastAPI REST + SSE  |  A2A (a2a-sdk)   — async edge over the sync rag core
```

- **Pipes are files on disk**, keyed by a `guid`; a failed stage is fixed and resumed without
  re-running the whole pipeline.
- **Contracts at every boundary** — Pydantic models (`Section`, `Chunk`, `EmbeddedChunk`), one per
  `jsonl` line.
- **Dependency injection** — config is read in one place (the composition root) and passed down
  through constructors. Shared libs (`db`, `llm_gateway`) never read config themselves.
- **One LLM gateway** — all completion/embedding calls go through `llm_gateway` (LiteLLM).

## Packages

| Package | Role |
|---|---|
| `packages/ingest` | Ingestion app: Extract → Structure → Clean → Chunk → Embed → Persist + orchestrator/CLI |
| `packages/rag` | Query side: searchers, RRF/MMR, small2big, the deep-search loop, CLI |
| `packages/serve` | Transport: FastAPI REST + SSE and A2A (web/SDK deps quarantined here) |
| `packages/shared/db` | DB schema + access (pgvector HNSW + pg_search BM25 + ltree); `DIMENSION` contract |
| `packages/shared/llm_gateway` | The single LiteLLM gateway |

## Environment

- **Windows host**, Python 3.12 via **uv**, `.venv` at repo root. Run Python as
  `& .venv\Scripts\python.exe ...`.
- **Database runs in Docker, code runs on the host.** Start only the DB service:
  `docker compose -f .devcontainer/docker-compose.yml up -d db`. Connect via **`localhost:55432`**
  (ParadeDB = PostgreSQL 18 + pgvector + pg_search). `rag` = working DB, `rag_test` = tests only.
- `uv sync --inexact` to install (preserves the pinned `torch 2.7.0+cpu`).

## Run

Ingest a book end-to-end (secrets in `.devcontainer/.env`, export first — Dynaconf won't auto-find them):

```bash
set -a; . .devcontainer/.env; set +a
export DATABASE_URL="postgresql://rag:rag@localhost:55432/rag"
& .venv\Scripts\python.exe -m ingest --source Designing_Data.pdf --external-id ddia --guid <guid> --profile ddia --clean
```

Re-running the same `guid` skips completed stages. Then query:

```bash
& .venv\Scripts\python.exe -m rag --question "How does single-leader replication handle failover?"
```

Serve over HTTP/A2A: `& .venv\Scripts\python.exe -m serve` (uvicorn).

## Test

```bash
& .venv\Scripts\python.exe -m pytest -q     # tests/pipeline (TDD, red-first)
```

## Documents (source of truth)

- `CLAUDE.md` — project constitution / working rules.
- `docs/Rag_pipeline_architecture.md` — ingest architecture, contracts, orchestration, gateway, config.
- `docs/Rag_query_architecture.md` — query side: the deep-search loop, source-aware reflect, MMR, small2big.
- `docs/Rag_schema_sections.md` — DB schema (pgvector HNSW + BM25 + ltree) and query examples.
- `docs/Rag_implementation_steps.md` — checklist S0…S15 + testing rules and the R1…R13 map.
- `docs/RESUME_RAG_POC.md` — current session state / how to resume.
