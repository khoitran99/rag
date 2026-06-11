from rag.models import Chunk
from rag.vector_index import VectorIndex
from rag.metadata_store import MetadataStore
from rag.retriever import Retriever

from tests.fakes import FakeEmbedder


def test_retrieve_returns_most_relevant_chunk_with_its_source(tmp_path):
    embedder = FakeEmbedder(dim=16)
    chunks = [
        Chunk(chunk_id=0, document="alpha.pdf", page=1, text="how to submit an expense report"),
        Chunk(chunk_id=1, document="beta.pdf", page=7, text="annual leave policy details"),
    ]
    index = VectorIndex(dim=embedder.dim)
    index.add(ids=[c.chunk_id for c in chunks], vectors=embedder.embed([c.text for c in chunks]))
    store = MetadataStore(tmp_path / "meta.sqlite")
    store.add(chunks)

    retriever = Retriever(embedder=embedder, index=index, store=store, top_k=1)
    results = retriever.retrieve("how to submit an expense report")

    assert len(results) == 1
    assert results[0].document == "alpha.pdf"
    assert results[0].page == 1
    assert results[0].text == "how to submit an expense report"
