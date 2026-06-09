"""Ingestion pipeline: parse -> chunk -> embed -> index, over a documents folder.

Slice 1 does a full pass over every PDF. Hash-based incremental ingestion (skip unchanged,
re-index changed, delete removed) arrives in Slice 4.
"""

from __future__ import annotations

from pathlib import Path

from rag.models import Chunk


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
        chunks: list[Chunk] = []
        next_id = 0
        for pdf_path in sorted(folder.glob("*.pdf")):
            for page in self._parser.parse(pdf_path):
                for piece in self._chunker.split(page.text):
                    chunks.append(
                        Chunk(
                            chunk_id=next_id,
                            document=pdf_path.name,
                            page=page.page,
                            text=piece,
                        )
                    )
                    next_id += 1

        if not chunks:
            return 0

        vectors = self._embedder.embed([c.text for c in chunks])
        self._index.add(ids=[c.chunk_id for c in chunks], vectors=vectors)
        self._store.add(chunks)
        return len(chunks)
