import numpy as np

from rag.vector_index import VectorIndex


def test_search_returns_nearest_chunk_ids_by_similarity():
    index = VectorIndex(dim=3)
    index.add(
        ids=[10, 20, 30],
        vectors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )

    # Query points mostly along the third axis -> chunk 30 is nearest, then 20.
    results = index.search(np.array([0.1, 0.3, 0.9]), k=2)

    assert results == [30, 20]
