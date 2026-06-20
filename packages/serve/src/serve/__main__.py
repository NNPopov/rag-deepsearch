"""Transport CLI entry point (Q13): `python -m serve` — bring up uvicorn over `build_serve_app()`.

Thin wrapper: parse host/port, build the app via the composition root (`serve.app`) and hand it
to uvicorn. There is no search logic here — only binding the ASGI app to a socket.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from serve.app import build_serve_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="serve",
        description="RAG transport: FastAPI REST + SSE over the deep-search core.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="listen address")
    parser.add_argument("--port", type=int, default=8000, help="port")
    args = parser.parse_args(argv)

    import uvicorn

    # ADR-0002: the core queries Postgres via psycopg's AsyncConnection, which on Windows cannot run
    # on a ProactorEventLoop. uvicorn's own loop factory hardcodes ProactorEventLoop on win32 (ignoring
    # the policy), so we take loop ownership: loop="none" tells uvicorn not to set one up, and we drive
    # server.serve() inside our own asyncio.run over a SelectorEventLoop.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        config = uvicorn.Config(build_serve_app(), host=args.host, port=args.port, loop="none")
        asyncio.run(uvicorn.Server(config).serve())
    else:
        uvicorn.run(build_serve_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
