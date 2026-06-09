"""Retrieval: turn a query into the most relevant chunks.

Slice 1 embeds the query and does an exact nearest-neighbour search. Safety filtering and
query expansion are inserted ahead of embedding in Slice 6; this seam stays the same.
"""

from __future__ import annotations

from rag.models import Chunk


class Retriever:
    def __init__(self, *, embedder, index, store, top_k: int = 5):
        self._embedder = embedder
        self._index = index
        self._store = store
        self._top_k = top_k

    def retrieve(self, query: str) -> list[Chunk]:
        query_vector = self._embedder.embed([query])
        ids = self._index.search(query_vector, k=self._top_k)
        return [self._store.get(i) for i in ids]
