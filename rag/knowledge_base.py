"""Knowledge base write/read seam over vectors and metadata.

The vector index and metadata store have to move in lockstep when a document changes or
disappears. This module keeps that coordination local so ingestion can stay focused on
finding, parsing, chunking, and embedding documents.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rag.models import Chunk


class KnowledgeBase:
    def __init__(self, *, index, store, trust_hashes: bool = True):
        self._index = index
        self._store = store
        self._trust_hashes = trust_hashes

    def is_current(self, document: str, content_hash: str) -> bool:
        return (
            self._trust_hashes
            and self._store.hash_of(document) == content_hash
            and bool(self._store.chunk_ids_for(document))
        )

    def replace_document(
        self,
        *,
        document: str,
        chunks: list[Chunk],
        vectors: np.ndarray,
        content_hash: str,
    ) -> int:
        """Replace one document's indexed state and record its latest content hash."""
        self.remove_document(document)
        if chunks:
            self._index.add(ids=[chunk.chunk_id for chunk in chunks], vectors=vectors)
            self._store.add(chunks)
        self._store.set_hash(document, content_hash)
        return len(chunks)

    def remove_document(self, document: str) -> None:
        ids = self._store.chunk_ids_for(document)
        if ids:
            self._index.remove(ids)
        self._store.delete_by_document(document)

    def remove_documents_except(self, documents: set[str]) -> None:
        for document in set(self._store.documents()) - documents:
            self.remove_document(document)

    def search(self, query: np.ndarray, k: int) -> list[Chunk]:
        chunks: list[Chunk] = []
        for chunk_id in self._index.search(query, k=k):
            try:
                chunks.append(self._store.get(chunk_id))
            except KeyError:
                continue
        return chunks

    def persist(self, path: str | Path) -> None:
        self._index.persist(path)

    def close(self) -> None:
        self._store.close()
