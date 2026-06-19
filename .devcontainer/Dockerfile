# Application dev container (Linux). Moving away from Windows fragility (torch/docling/poppler).
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# poppler-utils -> provides pdftotext (available on Linux; layout quality better than PyPDF2)
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# uv — fast package manager (the same one used on the host)
# TODO: pin the uv version once we lock the toolchain
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# IMPORTANT: keep the Linux venv OUTSIDE /workspace (bind-mount) to avoid conflicting
# with the host .venv (Windows binaries). uv will create the environment here.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# /opt is owned by root, but the container runs as vscode — create the venv folder ahead of time
# and hand it to vscode, otherwise `uv sync` will fail with permission denied.
RUN mkdir -p /opt/venv && chown vscode:vscode /opt/venv
