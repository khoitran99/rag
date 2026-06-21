"""Incremental, hash-based ingestion (Slice 4).

Uses a controlled temp folder of fake '.pdf' files plus a stub parser and a counting embedder,
so we can assert exactly which files get (re)processed without touching pipeline internals.
"""

from rag.chunker import Chunker
from rag.ingestion import IngestionPipeline
from rag.metadata_store import MetadataStore
from rag.models import LayoutBlock
from rag.vector_index import VectorIndex

from tests.fakes import CountingEmbedder


class StubParser:
    """Reads a fake .pdf (plain text on disk) and emits one body block per file."""

    def parse(self, pdf_path):
        text = open(pdf_path, encoding="utf-8").read()
        return [LayoutBlock(document=pdf_path.name, page=1, kind="body", text=text)]


def _write(folder, name, text):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")


def _pipeline(tmp_path, embedder):
    store = MetadataStore(tmp_path / "meta.sqlite")
    index = VectorIndex(dim=embedder.dim)
    pipeline = IngestionPipeline(
        parser=StubParser(),
        chunker=Chunker(max_tokens=512, overlap_tokens=64),
        embedder=embedder,
        index=index,
        store=store,
    )
    return pipeline, store, index


def test_ingestion_assigns_stable_indexed_ids_from_document_and_draft_position(tmp_path):
    docs = tmp_path / "documents"
    _write(docs, "a.pdf", "Original expense policy.")

    embedder = CountingEmbedder(dim=16)
    store = MetadataStore(tmp_path / "meta.sqlite")
    index = VectorIndex(dim=embedder.dim)
    pipeline = IngestionPipeline(
        parser=StubParser(),
        chunker=Chunker(max_tokens=512, overlap_tokens=64),
        embedder=embedder,
        index=index,
        store=store,
    )

    pipeline.ingest(docs)
    original_a_ids = store.chunk_ids_for("a.pdf")

    _write(docs, "a.pdf", "Updated expense policy.")
    pipeline.ingest(docs)
    updated_a_ids = store.chunk_ids_for("a.pdf")

    _write(docs, "b.pdf", "Updated expense policy.")
    pipeline.ingest(docs)

    assert updated_a_ids == original_a_ids
    assert [store.get(cid).text for cid in updated_a_ids] == ["Updated expense policy."]
    assert store.chunk_ids_for("b.pdf") != updated_a_ids


def test_reingesting_unchanged_folder_embeds_nothing(tmp_path):
    docs = tmp_path / "documents"
    _write(docs, "a.pdf", "Expense reports go in the portal.")

    embedder = CountingEmbedder(dim=16)
    pipeline, _store, _index = _pipeline(tmp_path, embedder)

    first = pipeline.ingest(docs)
    assert first >= 1
    embedded_after_first = embedder.embedded_count
    assert embedded_after_first >= 1

    # Nothing changed on disk -> second pass must skip the file entirely.
    second = pipeline.ingest(docs)
    assert second == 0
    assert embedder.embedded_count == embedded_after_first


def test_adding_a_new_pdf_indexes_only_that_file(tmp_path):
    docs = tmp_path / "documents"
    _write(docs, "a.pdf", "Expense reports go in the portal.")

    embedder = CountingEmbedder(dim=16)
    pipeline, store, _index = _pipeline(tmp_path, embedder)

    pipeline.ingest(docs)
    embedded_after_first = embedder.embedded_count

    # Add a second file; only it should be parsed/embedded, and both end up indexed.
    _write(docs, "b.pdf", "Travel is reimbursed within fourteen days.")
    added = pipeline.ingest(docs)

    assert added == 1  # only b.pdf produced a chunk this run
    assert embedder.embedded_count == embedded_after_first + 1
    assert set(store.documents()) == {"a.pdf", "b.pdf"}


def test_modifying_a_pdf_reindexes_only_that_file(tmp_path):
    docs = tmp_path / "documents"
    _write(docs, "a.pdf", "Old expense policy.")
    _write(docs, "b.pdf", "Travel is reimbursed.")

    embedder = CountingEmbedder(dim=16)
    pipeline, store, _index = _pipeline(tmp_path, embedder)

    pipeline.ingest(docs)
    embedded_after_first = embedder.embedded_count

    # Change only a.pdf's content.
    _write(docs, "a.pdf", "New expense policy with updated limits.")
    reindexed = pipeline.ingest(docs)

    assert reindexed == 1  # only a.pdf re-processed
    assert embedder.embedded_count == embedded_after_first + 1  # b.pdf not re-embedded
    # a.pdf now holds the new text, and none of its chunks carry the old text.
    a_texts = [store.get(cid).text for cid in store.chunk_ids_for("a.pdf")]
    assert any("updated limits" in t for t in a_texts)
    assert all("Old expense policy" not in t for t in a_texts)


def test_removing_a_pdf_deletes_its_chunks_from_index_and_metadata(tmp_path):
    docs = tmp_path / "documents"
    _write(docs, "a.pdf", "Expense reports go in the portal.")
    _write(docs, "b.pdf", "Travel is reimbursed.")

    embedder = CountingEmbedder(dim=16)
    pipeline, store, index = _pipeline(tmp_path, embedder)

    pipeline.ingest(docs)
    b_ids = store.chunk_ids_for("b.pdf")
    assert b_ids  # b.pdf was indexed

    # Delete b.pdf from the folder and re-ingest.
    (docs / "b.pdf").unlink()
    pipeline.ingest(docs)

    # Gone from metadata...
    assert store.documents() == ["a.pdf"]
    assert store.chunk_ids_for("b.pdf") == []
    # ...and gone from the vector index (its ids never come back from a search).
    results = index.search(embedder.embed(["anything"])[0], k=10)
    assert set(results).isdisjoint(b_ids)


def test_index_and_metadata_survive_a_simulated_restart(tmp_path):
    docs = tmp_path / "documents"
    _write(docs, "a.pdf", "Expense reports go in the portal.")
    index_path = tmp_path / "index.faiss"

    embedder = CountingEmbedder(dim=16)
    index = VectorIndex(dim=embedder.dim)
    store = MetadataStore(tmp_path / "meta.sqlite")
    IngestionPipeline(
        parser=StubParser(),
        chunker=Chunker(max_tokens=512, overlap_tokens=64),
        embedder=embedder,
        index=index,
        store=store,
    ).ingest(docs)
    index.persist(index_path)
    store.close()

    # Simulate a restart: reload index + metadata from disk into fresh objects.
    reloaded_index = VectorIndex.load(index_path)
    reloaded_store = MetadataStore(tmp_path / "meta.sqlite")

    # Retrieval still works and resolves back to the source document.
    hits = reloaded_index.search(embedder.embed(["expense reports"])[0], k=1)
    assert hits
    assert reloaded_store.get(hits[0]).document == "a.pdf"

    # And incremental state persisted: re-ingesting the unchanged folder does nothing.
    again = IngestionPipeline(
        parser=StubParser(),
        chunker=Chunker(max_tokens=512, overlap_tokens=64),
        embedder=embedder,
        index=reloaded_index,
        store=reloaded_store,
    ).ingest(docs)
    assert again == 0
