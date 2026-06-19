# Golden set — retrieval-level evaluation on Pro Git

A **publicly reproducible** quality check for the RAG pipeline's retrieval, built on the
freely-licensed **Pro Git** corpus (CC BY-NC-SA 3.0; see `data/progit/NOTICE.md`). Unlike the
copyrighted DDIA/Acing books, this corpus and its golden questions can live in a public repo.

## What it checks

- **Level:** *retrieval* — deterministic, no LLM judge, no API cost beyond query embedding.
  For each question, the expected **chapter** must appear within the hybrid (vector+BM25→RRF)
  top-`min_recall_k`. This exercises retrieval + RRF + cross-lingual search.
- **Not checked here:** final answer correctness (answer-level / LLM-judge) — a separate,
  non-deterministic concern.
- **Granularity:** chapter-level (profile `progit` — section headings in Pro Git aren't
  reliably detectable; expected target is the chapter `path` = ltree ordinal `1..10`).
- **Cross-lingual:** RU questions target the **same English corpus** (PLAN/REFLECT search in
  the corpus language, answer in the question language; §5). Both RU questions resolve to the
  correct chapter at rank @1.

Data: `progit_golden.yaml` (corpus metadata + 12 verified questions, 10 EN + 2 RU).

## Seed the corpus (once)

Needs the ParadeDB container up and an OpenAI key (`.devcontainer/.env`). Embeddings cost
~$0.01. Creates a dedicated `rag_public` DB — never touches the working `rag`.

```bash
docker compose -f .devcontainer/docker-compose.yml up -d db
docker compose -f .devcontainer/docker-compose.yml exec -T db \
    psql -U rag -d postgres -c "CREATE DATABASE rag_public OWNER rag;"
& .venv/Scripts/python.exe -c "import psycopg, db.schema; \
    c=psycopg.connect('postgresql://rag:rag@localhost:55432/rag_public'); \
    db.schema.apply_schema(c); c.commit()"
# ingest (see data/progit/NOTICE.md for the full command) + set the document title.
```

## Run the harness

```bash
set -a; . .devcontainer/.env; set +a            # OpenAI key for query embedding
GOLDEN_DSN="postgresql://rag:rag@localhost:55432/rag_public" \
    & .venv/Scripts/python.exe tests/golden/run_golden.py
```

Exit `0` = all questions pass; `1` = misses listed. The query embedding model
(`text-embedding-3-large` @ 1536) **must** match the model the corpus was embedded with.

## Why not in the pytest gate

Like the other live e2e drivers, this needs a seeded DB + network (query embedding), so it is
a standalone runner, kept out of `pytest -q` (which stays hermetic). The `progit` **structure
profile** itself is covered by unit tests in `tests/pipeline/test_structure_progit.py`.
