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


def test_removed_ids_no_longer_appear_in_search_results():
    index = VectorIndex(dim=3)
    index.add(
        ids=[10, 20, 30],
        vectors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )

    index.remove([30])

    # The query still points along the third axis, but 30 is gone -> 20 is now nearest.
    results = index.search(np.array([0.1, 0.3, 0.9]), k=3)

    assert 30 not in results
    assert results[0] == 20


def test_persisted_index_reproduces_search_after_reload(tmp_path):
    path = tmp_path / "index.faiss"
    index = VectorIndex(dim=3)
    index.add(
        ids=[10, 20, 30],
        vectors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    index.persist(path)

    # Simulate a process restart: load from disk into a fresh instance.
    reloaded = VectorIndex.load(path)
    results = reloaded.search(np.array([0.1, 0.3, 0.9]), k=2)

    assert results == [30, 20]
