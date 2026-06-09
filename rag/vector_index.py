"""Vector index over chunk embeddings.

Slice 1 provides the exact (flat) nearest-neighbour path. The clustering-based ANN
(FAISS IVF) path is added in a later slice; the interface here is designed to absorb it.
"""

from __future__ import annotations

import faiss
import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class VectorIndex:
    """Exact nearest-neighbour search over chunk embeddings.

    Cosine similarity is used (vectors are L2-normalised, then inner-product searched).
    Vectors are keyed by an integer chunk id so results map back to the metadata store.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        self._index.add_with_ids(_normalize(vectors), np.asarray(ids, dtype="int64"))

    def search(self, query: np.ndarray, k: int) -> list[int]:
        _, ids = self._index.search(_normalize(query), k)
        return [int(i) for i in ids[0] if i != -1]

    def persist(self, path: str | Path) -> None:
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        raw = faiss.read_index(str(path))
        index = cls.__new__(cls)
        index.dim = raw.d
        index._index = raw
        return index
