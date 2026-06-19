"""Transport composition root (Q13) — the single place that assembles the app + reads config.

Mirrors `rag/app.py` and `rag/__main__.py`: config (`load_settings`) is read ONCE, the
`DeepSearch` core is built via `rag.app.build_deep_search` (DI), then wrapped in FastAPI
(`serve.api.create_app`). Web dependencies live only here and in `api.py`/`a2a.py` — the `rag`
core knows nothing about transport.

`load`/`build`/`deep_search` are injected: tests substitute them (to verify assembly without network/DB),
and a `deep_search` passed directly bypasses config reading entirely. Assembly is lazy — no network, no
DB at app build time (the core hits them only on a real request).
"""
from __future__ import annotations

from fastapi import FastAPI

from rag.app import build_deep_search
from rag.config import load_settings
from serve.api import create_app


def build_serve_app(
    *,
    deep_search=None,
    load=load_settings,
    build=build_deep_search,
    with_a2a: bool = True,
) -> FastAPI:
    """FastAPI app with REST+SSE and (optionally) A2A on the SAME app. If `deep_search` is not passed —
    reads config and builds the core. `with_a2a` mounts the Agent Card + JSON-RPC on top (§7)."""
    if deep_search is None:
        settings = load()                 # the single config read (composition root)
        deep_search = build(settings)
    app = create_app(deep_search=deep_search)
    if with_a2a:
        from serve.a2a import add_a2a_routes   # lazy web/SDK import — do not pull a2a unless needed

        add_a2a_routes(app, deep_search)
    return app
