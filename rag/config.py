"""Runtime configuration.

Everything that should be swappable without touching pipeline logic lives here: the
embedding model, the LLM, retrieval depth, chunk sizing, and where artifacts are stored.
Values can be overridden via environment variables (RAG_* prefix).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    documents_dir: Path = Path("documents")
    index_path: Path = Path("vector_store/index.faiss")
    db_path: Path = Path("vector_store/metadata.sqlite")

    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768  # nomic-embed-text
    llm_model: str = "qwen2.5:7b-instruct"

    # "quick" = rule-based PyMuPDF (instant, no deps); "ai" = AI-based Docling layout+OCR
    # (faithful to the lecture's Figure 6 branch, see ADR-0002). The AI path is opt-in because
    # Docling is a heavy optional dependency that a human installs.
    parser: str = "quick"

    top_k: int = 5
    # Small chunks keep retrieval precise while the quick-path parser emits one body block
    # per page (no real structure). Once the AI-based parser (Slice 8) supplies real headings
    # and tables, raise this toward 256-512 so chunks become coherent sections.
    chunk_max_tokens: int = 128
    chunk_overlap_tokens: int = 16

    ollama_host: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "Config":
        def _get(name: str, default):
            return os.environ.get(name, default)

        return cls(
            documents_dir=Path(_get("RAG_DOCUMENTS_DIR", cls.documents_dir)),
            index_path=Path(_get("RAG_INDEX_PATH", cls.index_path)),
            db_path=Path(_get("RAG_DB_PATH", cls.db_path)),
            embed_model=_get("RAG_EMBED_MODEL", cls.embed_model),
            embed_dim=int(_get("RAG_EMBED_DIM", cls.embed_dim)),
            llm_model=_get("RAG_LLM_MODEL", cls.llm_model),
            parser=_get("RAG_PARSER", cls.parser),
            top_k=int(_get("RAG_TOP_K", cls.top_k)),
            chunk_max_tokens=int(_get("RAG_CHUNK_MAX_TOKENS", cls.chunk_max_tokens)),
            chunk_overlap_tokens=int(_get("RAG_CHUNK_OVERLAP_TOKENS", cls.chunk_overlap_tokens)),
            ollama_host=_get("RAG_OLLAMA_HOST", cls.ollama_host),
        )
