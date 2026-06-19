"""Transport CLI entry point (Q13): `python -m serve` — bring up uvicorn over `build_serve_app()`.

Thin wrapper: parse host/port, build the app via the composition root (`serve.app`) and hand it
to uvicorn. There is no search logic here — only binding the ASGI app to a socket.
"""
from __future__ import annotations

import argparse
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

    uvicorn.run(build_serve_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
