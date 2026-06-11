"""Ingestion pipeline: parse -> chunk -> embed -> index, over a documents folder.

Slice 1 does a full pass over every PDF. Hash-based incremental ingestion (skip unchanged,
re-index changed, delete removed) arrives in Slice 4.
"""

from __future__ import annotations

from pathlib import Path

from rag.models import LayoutBlock


class IngestionPipeline:
    def __init__(self, *, parser, chunker, embedder, index, store):
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._index = index
        self._store = store

    def ingest(self, folder: str | Path) -> int:
        """Index every PDF in `folder`. Returns the number of chunks indexed."""
        folder = Path(folder)
        blocks: list[LayoutBlock] = []
        for pdf_path in sorted(folder.glob("*.pdf")):
            blocks.extend(self._parser.parse(pdf_path))

        chunks = self._chunker.chunk(blocks)
        if not chunks:
            return 0

        vectors = self._embedder.embed([c.text for c in chunks])
        self._index.add(ids=[c.chunk_id for c in chunks], vectors=vectors)
        self._store.add(chunks)
        return len(chunks)
