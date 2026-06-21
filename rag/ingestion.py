"""Ingestion pipeline: parse -> chunk -> embed -> index, over a documents folder.

Incremental and hash-based (Slice 4): each PDF is processed independently and keyed by a hash
of its bytes. On re-ingest, unchanged files are skipped, new/modified files are re-indexed in
place, and files removed from the folder have their chunks dropped from both the index and the
metadata store. Chunk ids are derived from (document, position) so re-indexing one file never
disturbs another's ids.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag.knowledge_base import KnowledgeBase
from rag.models import Chunk


def _content_hash(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def _chunk_id(document: str, position: int) -> int:
    """A stable, globally-unique chunk id from the document name and the chunk's position.

    Same file + position always yields the same id (idempotent re-ingest); distinct documents
    never collide. 60 bits keeps it a positive int64 for FAISS.
    """
    digest = hashlib.sha256(f"{document}#{position}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


class IngestionPipeline:
    def __init__(
        self,
        *,
        parser,
        chunker,
        embedder,
        knowledge_base: KnowledgeBase | None = None,
        index=None,
        store=None,
    ):
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._index = index
        self._store = store
        if knowledge_base is None:
            if index is None or store is None:
                raise TypeError("knowledge_base or both index and store are required")
            knowledge_base = KnowledgeBase(index=index, store=store)
        self._knowledge_base = knowledge_base

    def ingest(self, folder: str | Path) -> int:
        """Index new/changed PDFs in `folder`, drop removed ones. Returns chunks (re)indexed."""
        folder = Path(folder)
        present = sorted(folder.glob("*.pdf"))
        present_names = {p.name for p in present}

        indexed = 0
        for pdf_path in present:
            content_hash = _content_hash(pdf_path)
            if self._knowledge_base.is_current(pdf_path.name, content_hash):
                continue  # unchanged -> skip entirely (no parse, no embed)
            indexed += self._reindex(pdf_path, content_hash)

        self._knowledge_base.remove_documents_except(present_names)

        return indexed

    def _reindex(self, pdf_path: Path, content_hash: str) -> int:
        document = pdf_path.name

        blocks = self._parser.parse(pdf_path)
        chunks = [
            Chunk(
                chunk_id=_chunk_id(document, draft.position),
                document=document,
                page=draft.page,
                text=draft.text,
            )
            for draft in self._chunker.chunk(blocks)
        ]
        vectors = self._embedder.embed([c.text for c in chunks]) if chunks else []
        return self._knowledge_base.replace_document(
            document=document,
            chunks=chunks,
            vectors=vectors,
            content_hash=content_hash,
        )
