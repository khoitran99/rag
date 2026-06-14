"""Ingestion pipeline: parse -> chunk -> embed -> index, over a documents folder.

Incremental and hash-based (Slice 4): each PDF is processed independently and keyed by a hash
of its bytes. On re-ingest, unchanged files are skipped, new/modified files are re-indexed in
place, and files removed from the folder have their chunks dropped from both the index and the
metadata store. Chunk ids are derived from (document, position) so re-indexing one file never
disturbs another's ids.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path


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
    def __init__(self, *, parser, chunker, embedder, index, store):
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._index = index
        self._store = store

    def ingest(self, folder: str | Path) -> int:
        """Index new/changed PDFs in `folder`, drop removed ones. Returns chunks (re)indexed."""
        folder = Path(folder)
        present = sorted(folder.glob("*.pdf"))
        present_names = {p.name for p in present}

        indexed = 0
        for pdf_path in present:
            content_hash = _content_hash(pdf_path)
            if self._store.hash_of(pdf_path.name) == content_hash:
                continue  # unchanged -> skip entirely (no parse, no embed)
            indexed += self._reindex(pdf_path, content_hash)

        # Files that vanished from the folder: forget their chunks and vectors.
        for document in set(self._store.documents()) - present_names:
            self._drop(document)

        return indexed

    def _reindex(self, pdf_path: Path, content_hash: str) -> int:
        document = pdf_path.name
        self._drop(document)  # clear any prior version before re-adding

        blocks = self._parser.parse(pdf_path)
        chunks = [
            replace(c, chunk_id=_chunk_id(document, c.chunk_id))
            for c in self._chunker.chunk(blocks)
        ]
        if chunks:
            vectors = self._embedder.embed([c.text for c in chunks])
            self._index.add(ids=[c.chunk_id for c in chunks], vectors=vectors)
            self._store.add(chunks)
        self._store.set_hash(document, content_hash)
        return len(chunks)

    def _drop(self, document: str) -> None:
        ids = self._store.chunk_ids_for(document)
        if ids:
            self._index.remove(ids)
        self._store.delete_by_document(document)
