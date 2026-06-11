import fitz  # PyMuPDF

from rag.parser import PdfParser
from rag.chunker import NaiveChunker
from rag.vector_index import VectorIndex
from rag.metadata_store import MetadataStore
from rag.ingestion import IngestionPipeline

from tests.fakes import FakeEmbedder


def _write_pdf(path, lines: list[str]):
    doc = fitz.open()
    for line in lines:
        page = doc.new_page()
        page.insert_text((72, 72), line)
    doc.save(str(path))
    doc.close()


def test_ingest_populates_metadata_with_source_document_and_page(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    _write_pdf(docs / "handbook.pdf", ["Expense reports go in the portal.", "Travel is reimbursed."])

    embedder = FakeEmbedder(dim=16)
    store = MetadataStore(tmp_path / "meta.sqlite")
    pipeline = IngestionPipeline(
        parser=PdfParser(),
        chunker=NaiveChunker(chunk_size=200, overlap=20),
        embedder=embedder,
        index=VectorIndex(dim=embedder.dim),
        store=store,
    )

    count = pipeline.ingest(docs)

    assert count >= 2
    # Every chunk traces back to the source document and a real page number.
    indexed = [store.get(i) for i in range(count)]
    assert all(c.document == "handbook.pdf" for c in indexed)
    assert {c.page for c in indexed} == {1, 2}
    assert any("Expense reports" in c.text for c in indexed)
