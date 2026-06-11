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
