import pytest

from rag.models import Chunk
from rag.metadata_store import MetadataStore


def test_metadata_round_trips_and_survives_reload(tmp_path):
    db_path = tmp_path / "meta.sqlite"

    store = MetadataStore(db_path)
    store.add(
        [
            Chunk(chunk_id=1, document="06 - RAG.pdf", page=3, text="Retrieval finds context."),
            Chunk(chunk_id=2, document="06 - RAG.pdf", page=4, text="Generation writes the answer."),
        ]
    )
    store.close()

    # Reopen the same file -> data must still be there.
    reopened = MetadataStore(db_path)
    chunk = reopened.get(2)

    assert chunk == Chunk(
        chunk_id=2, document="06 - RAG.pdf", page=4, text="Generation writes the answer."
    )


def test_delete_by_document_removes_only_that_documents_chunks(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite")
    store.add(
        [
            Chunk(chunk_id=1, document="a.pdf", page=1, text="alpha"),
            Chunk(chunk_id=2, document="a.pdf", page=2, text="beta"),
            Chunk(chunk_id=3, document="b.pdf", page=1, text="gamma"),
        ]
    )

    store.delete_by_document("a.pdf")

    # a.pdf's chunks are gone; b.pdf's chunk survives.
    with pytest.raises(KeyError):
        store.get(1)
    assert store.get(3).text == "gamma"


def test_chunk_ids_for_and_documents_expose_what_is_stored(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite")
    store.add(
        [
            Chunk(chunk_id=11, document="a.pdf", page=1, text="alpha"),
            Chunk(chunk_id=12, document="a.pdf", page=2, text="beta"),
            Chunk(chunk_id=13, document="b.pdf", page=1, text="gamma"),
        ]
    )

    assert set(store.documents()) == {"a.pdf", "b.pdf"}
    assert set(store.chunk_ids_for("a.pdf")) == {11, 12}
    assert store.chunk_ids_for("missing.pdf") == []


def test_document_hash_round_trips_and_survives_reload(tmp_path):
    db_path = tmp_path / "meta.sqlite"

    store = MetadataStore(db_path)
    assert store.hash_of("a.pdf") is None  # nothing recorded yet
    store.set_hash("a.pdf", "deadbeef")
    store.set_hash("a.pdf", "cafef00d")  # latest write wins
    store.close()

    reopened = MetadataStore(db_path)
    assert reopened.hash_of("a.pdf") == "cafef00d"
    assert reopened.hash_of("never-seen.pdf") is None
