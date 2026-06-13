"""One-command AI-parser ingest for manual verification (Slice 3).

Avoids the shell-paste pitfalls (split lines, unexported env vars) by setting everything
in-process. Ingests a documents folder through the Docling-backed AiLayoutParser into a
SEPARATE store (vector_store_ai/), leaving the default quick-path store untouched.

OMP_NUM_THREADS must be set BEFORE faiss/torch import, so it is set here at the very top.

Run:  .venv/bin/python scripts/ingest_ai.py [documents_dir]
      (defaults to documents/; pass a smaller folder for a fast single-PDF smoke test)
"""

import os

# Serialize OpenMP before any heavy import — prevents the faiss/torch segfault (exit 139).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import Config
from rag.factory import run_ingestion

documents_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("documents")

config = Config(
    parser="ai",
    documents_dir=documents_dir,
    index_path=Path("vector_store_ai/index.faiss"),
    db_path=Path("vector_store_ai/metadata.sqlite"),
)

print(f"AI ingest (Docling) of {documents_dir}/ into vector_store_ai/ — ~3-4 min/PDF, OCR logs scroll...")
count = run_ingestion(config)
print(f"\nDone: indexed {count} chunks via the AI parser into vector_store_ai/")
