"""Retrieval: turn a query into the most relevant chunks.

Slice 1 embeds the query and does an exact nearest-neighbour search. Safety filtering and
query expansion are inserted ahead of embedding in Slice 6; this seam stays the same.
"""

from __future__ import annotations

from rag.knowledge_base import KnowledgeBase
from rag.models import Chunk


class Retriever:
    def __init__(
        self,
        *,
        embedder,
        knowledge_base: KnowledgeBase | None = None,
        index=None,
        store=None,
        top_k: int = 5,
    ):
        self._embedder = embedder
        if knowledge_base is None:
            if index is None or store is None:
                raise TypeError("knowledge_base or both index and store are required")
            knowledge_base = KnowledgeBase(index=index, store=store)
        self._knowledge_base = knowledge_base
        self._top_k = top_k

    def retrieve(self, query: str) -> list[Chunk]:
        query_vector = self._embedder.embed([query])
        return self._knowledge_base.search(query_vector, k=self._top_k)
