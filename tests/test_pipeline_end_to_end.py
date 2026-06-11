import fitz  # PyMuPDF

from rag.parser import PdfParser
from rag.chunker import Chunker
from rag.vector_index import VectorIndex
from rag.metadata_store import MetadataStore
from rag.ingestion import IngestionPipeline
from rag.retriever import Retriever
from rag.prompt_builder import PromptBuilder
from rag.pipeline import RagPipeline

from tests.fakes import FakeEmbedder, FakeGenerator


def _write_pdf(path, lines: list[str]):
    doc = fitz.open()
    for line in lines:
        page = doc.new_page()
        page.insert_text((72, 72), line)
    doc.save(str(path))
    doc.close()


def test_ask_returns_answer_grounded_in_cited_sources(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    _write_pdf(docs / "handbook.pdf", ["Submit expense reports in the portal."])

    embedder = FakeEmbedder(dim=16)
    index = VectorIndex(dim=embedder.dim)
    store = MetadataStore(tmp_path / "meta.sqlite")
    IngestionPipeline(
        parser=PdfParser(),
        chunker=Chunker(max_tokens=512, overlap_tokens=64),
        embedder=embedder,
        index=index,
        store=store,
    ).ingest(docs)

    pipeline = RagPipeline(
        retriever=Retriever(embedder=embedder, index=index, store=store, top_k=2),
        prompt_builder=PromptBuilder(),
        generator=FakeGenerator(),
    )

    answer = pipeline.ask("How do I submit an expense report?")

    assert answer.text.strip()  # a non-empty answer was generated
    assert answer.sources, "answer must carry the cited source chunks"
    # The cited sources trace back to the source document, so [1] resolves to a real page.
    assert answer.sources[0].document == "handbook.pdf"
    assert answer.sources[0].page == 1
