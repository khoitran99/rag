# Local RAG (ChatPDF)

A fully-offline, scaled-down reimplementation of the RAG architecture from the lecture
*06 - Retrieval-Augmented Generation*. It answers natural-language questions over the PDFs in
the `documents/` folder, with inline citations back to the source document and page.

See [`CONTEXT.md`](CONTEXT.md) for the domain glossary and [`docs/adr/`](docs/adr/) for the
key architectural decisions.

## Slice 1 — walking skeleton

The thinnest end-to-end path: parse (PyMuPDF) → chunk → embed → FAISS flat index +
SQLite metadata → retrieve → prompt (context + citation instruction) → generate → answer
with cited sources, shown in a Streamlit chat. Heavier, more faithful implementations of each
box arrive in later slices. The CLIP image-encoder / image-index branch is a documented stub.

Chunking is **structure-aware** (Slice 2): it groups a heading with its body, keeps tables
intact, never merges across pages (page-accurate citations), and splits oversized content to a
token budget (tiktoken) with overlap. Until the AI-based parser supplies real headings/tables,
the budget defaults small (`chunk_max_tokens=128`) to keep retrieval precise — bigger chunks
dilute the signal when every block is just a page of body text.

## Slice 3 — AI-based parser (opt-in)

The lecture's AI-based parsing branch (layout detection → OCR → structured blocks) is realized
with **Docling** (offline, CPU) instead of LayoutParser/Detectron2, which has no Apple-Silicon
wheels — see [ADR-0002](docs/adr/0002-full-ai-based-layoutparser-parsing.md). Detection sits
behind a `LayoutDetector` seam; the deep, unit-tested `AiLayoutParser` maps raw regions onto
`LayoutBlock`s (label→kind, column-aware reading order, source stamping) and feeds the same
Chunker unchanged. It is opt-in because Docling is a heavy dependency a human installs:

```bash
.venv/bin/pip install -e ".[ai]"                        # installs Docling (heavy)
OMP_NUM_THREADS=1 RAG_PARSER=ai .venv/bin/rag ingest     # layout detection + OCR
```

**`OMP_NUM_THREADS=1` is required for AI ingest.** Docling's `torch` and `faiss` each bundle
their own OpenMP runtime; with both live in one process they race and intermittently segfault
(exit 139). Serializing OpenMP avoids it. Query time doesn't load `torch`, so plain
`rag query` needs no such flag. Expect AI ingest to be slow (~3–4 min/PDF on CPU, plus a
one-time model download on first run).

With real headings/tables you can raise `RAG_CHUNK_MAX_TOKENS` toward 256–512 for more coherent,
section-level chunks. The CLIP image-encoder branch remains a documented stub.

## Slice 4 — persistent, incremental ingestion

`rag ingest` is **incremental**: each PDF is keyed by a hash of its bytes, so re-running ingest
skips unchanged files, re-indexes only new or modified ones, and drops the chunks of files
removed from `documents/`. The FAISS index and SQLite metadata (chunks + per-document hashes)
persist to disk and reload across restarts, so the cost of parsing/embedding is paid once per
new or changed document rather than on every run — which matters most with the slow AI parser.

```bash
.venv/bin/rag ingest      # first run indexes everything
.venv/bin/rag ingest      # second run does nothing if documents/ is unchanged
```

Chunk ids are derived from `(document, position)`, so re-indexing one file never disturbs
another's ids or vectors.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[runtime,dev]"
```

This is fully offline at runtime, but needs [Ollama](https://ollama.com) with two models:

```bash
ollama pull nomic-embed-text      # text encoder (768-dim)
ollama pull qwen2.5:7b-instruct   # generation
```

## Run

```bash
# 1. Put PDFs in documents/ (the book chapters are already there)
.venv/bin/rag ingest                 # parse, chunk, embed, persist the index

.venv/bin/rag query "How does RAG retrieval work?"   # one-off CLI query
.venv/bin/streamlit run rag/app.py                   # chat UI
```

Configuration (models, top-k, chunk size, paths) is overridable via `RAG_*` environment
variables — see [`rag/config.py`](rag/config.py).

## Tests

```bash
.venv/bin/python -m pytest
```

The pipeline is tested offline with deterministic fakes injected for the embedder and
generator (the Ollama adapters are thin and verified manually). No network or models needed.
