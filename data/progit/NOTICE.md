# Pro Git — test corpus (NOTICE & attribution)

`progit_en.pdf` is the **Pro Git** book, 2nd Edition, by **Scott Chacon and Ben Straub**,
published by Apress. It is included here **unmodified** as a freely-redistributable test
corpus for the RAG pipeline's golden set.

- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)
- **Source:** https://github.com/progit/progit2 — release **2.1.450**
  (direct asset: `releases/download/2.1.450/progit.pdf`)
- **Online:** https://git-scm.com/book
- **Russian edition** (for true parallel-corpus / cross-lingual testing, not yet ingested):
  https://github.com/progit/progit2-ru — release 2.1.123

Per CC BY-NC-SA 3.0, this material is reused with attribution, for non-commercial testing,
and any derivative shared under the same license. The PDF is unmodified; only derived
artifacts (extracted text, chunks, embeddings) are produced by the pipeline at ingest time.

## Why this book

Unlike *Designing Data-Intensive Applications* and *Acing the System Design Interview*
(both copyrighted, kept out of git), Pro Git can be committed to a public repository, so the
golden set built on it is **publicly reproducible**. It also ships official translations,
enabling honest cross-lingual evaluation (RU question → EN corpus).

## How the corpus is produced

```bash
set -a; . .devcontainer/.env; set +a
export DATABASE_URL="postgresql://rag:rag@localhost:55432/rag_public"
& .venv/Scripts/python.exe -m ingest \
    --source data/progit/progit_en.pdf --external-id progit \
    --guid progit-public --profile progit --clean
# затем проставить заголовок документа (иначе цитаты 'None › N'):
#   UPDATE documents SET title='Pro Git' WHERE external_id='progit';
```

Profile `progit` (see `packages/ingest/.../structure.py`): Pro Git's Asciidoctor-PDF layout
has no `Chapter N` running header and no numbered body headings — chapters are detected by the
**form-feed page-break opener** (`\f<Chapter Title>`) against a curated title set; structure is
**chapter-level** (10 chapters; section headings are bare lines, not reliably detectable —
TOC-driven subsections are a future step). Result: 1 document / 10 sections / 245 chunks.
