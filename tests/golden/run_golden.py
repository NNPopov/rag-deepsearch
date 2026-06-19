"""Golden set runner: hybrid retrieval (vector+bm25→RRF) against the seeded Pro Git corpus.

Reads progit_golden.yaml, and for each question checks that the expected chapter landed in
the hybrid result's top-k (the `min_recall_k` threshold). Deterministic retrieval level,
no LLM loop. The query embedding goes through the same gateway/model as the corpus (§4/§6).

Run (needs a seeded corpus + an OpenAI key; see README.md):
    set -a; . .devcontainer/.env; set +a
    GOLDEN_DSN="postgresql://rag:rag@localhost:55432/rag_public" \
        .venv/Scripts/python.exe tests/golden/run_golden.py

Exit 0 — all questions passed; 1 — there are misses (for CI/gate).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import yaml

from rag.app import build_gateway
from rag.config import load_settings
from rag.search import BM25Searcher, VectorSearcher, rrf

GOLDEN = Path(__file__).with_name("progit_golden.yaml")
DSN = os.environ.get("GOLDEN_DSN", "postgresql://rag:rag@localhost:55432/rag_public")


def _chapter_of(chunk) -> str:
    """RetrievedChunk.source = 'Title › path' → chapter path (small2big reference)."""
    return chunk.source.split("›")[-1].strip()


def main() -> int:
    spec = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    chapters = spec["chapters"]
    default_k = int(spec["default_min_recall_k"])

    settings = load_settings()
    gateway = build_gateway(settings)
    connect = lambda: psycopg.connect(DSN)  # noqa: E731
    top_k = max(default_k, int(settings.search.top_k))
    vec = VectorSearcher(connect=connect, gateway=gateway, top_k=top_k)
    bm = BM25Searcher(connect=connect, top_k=top_k)

    failures: list[str] = []
    print(f"golden set: {GOLDEN.name}  | DSN: {DSN}")
    print(f"{'id':<22}{'lang':<5}{'expect':<8}{'rank':<6}result")
    print("-" * 72)
    for q in spec["questions"]:
        k = int(q.get("min_recall_k", default_k))
        fused = rrf([vec.search(q["question"]), bm.search(q["question"])])
        paths = [_chapter_of(c) for c in fused]
        expect = set(q["expect_chapters"])
        rank = next((i + 1 for i, p in enumerate(paths) if p in expect), None)
        passed = rank is not None and rank <= k
        mark = "ok" if passed else "MISS"
        exp_str = "/".join(sorted(expect))
        rank_str = f"@{rank}" if rank else "-"
        top = " ".join(f"{p}:{chapters.get(p, '?')[:8]}" for p in paths[:k])
        print(f"{q['id']:<22}{q['language']:<5}{exp_str:<8}{rank_str:<6}[{mark}] {top}")
        if not passed:
            failures.append(q["id"])

    print("-" * 72)
    total = len(spec["questions"])
    print(f"recall@k: {total - len(failures)}/{total} passed (threshold per-question)")
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
