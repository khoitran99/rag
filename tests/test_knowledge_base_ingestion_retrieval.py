from pathlib import Path

import numpy as np

from rag.chunker import Chunker
from rag.config import Config
from rag.factory import load_pipeline, run_ingestion
from rag.ingestion import IngestionPipeline
from rag.knowledge_base import KnowledgeBase
from rag.metadata_store import MetadataStore
from rag.models import LayoutBlock
from rag.retriever import Retriever
from rag.vector_index import VectorIndex
from tests.fakes import FakeGenerator


class StubParser:
    def parse(self, pdf_path: Path):
        text = pdf_path.read_text(encoding="utf-8")
        return [LayoutBlock(document=pdf_path.name, page=1, kind="body", text=text)]


class SwitchingParser:
    def __init__(self):
        self.emit_blocks = False

    def parse(self, pdf_path: Path):
        if not self.emit_blocks:
            return []
        return StubParser().parse(pdf_path)


class KeywordEmbedder:
    terms = ("legacy", "backup", "fresh", "removed", "surviving")

    def __init__(self):
        self.dim = len(self.terms)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [float(text.lower().count(term)) for term in self.terms]
                for text in texts
            ],
            dtype="float32",
        )


def _write(folder: Path, name: str, text: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")


def _compose(tmp_path: Path, parser=None, top_k: int = 1):
    embedder = KeywordEmbedder()
    knowledge_base = KnowledgeBase(
        index=VectorIndex(dim=embedder.dim),
        store=MetadataStore(tmp_path / "metadata.sqlite"),
    )
    pipeline = IngestionPipeline(
        parser=parser or StubParser(),
        chunker=Chunker(max_tokens=512, overlap_tokens=64),
        embedder=embedder,
        knowledge_base=knowledge_base,
    )
    retriever = Retriever(
        embedder=embedder,
        knowledge_base=knowledge_base,
        top_k=top_k,
    )
    return pipeline, retriever


def test_changed_document_replaces_old_vectors_seen_by_retrieval(tmp_path):
    docs = tmp_path / "documents"
    _write(docs, "a.pdf", "legacy policy")
    _write(docs, "b.pdf", "legacy backup reference")

    pipeline, retriever = _compose(tmp_path)
    pipeline.ingest(docs)
    assert retriever.retrieve("legacy")[0].document == "a.pdf"

    _write(docs, "a.pdf", "fresh policy")
    pipeline.ingest(docs)

    results = retriever.retrieve("legacy")
    assert [chunk.document for chunk in results] == ["b.pdf"]
    assert results[0].text == "legacy backup reference"


def test_removed_document_disappears_from_retrieval(tmp_path):
    docs = tmp_path / "documents"
    _write(docs, "a.pdf", "removed policy")
    _write(docs, "b.pdf", "surviving policy")

    pipeline, retriever = _compose(tmp_path)
    pipeline.ingest(docs)
    assert retriever.retrieve("removed")[0].document == "a.pdf"

    (docs / "a.pdf").unlink()
    pipeline.ingest(docs)

    results = retriever.retrieve("removed")
    assert [chunk.document for chunk in results] == ["b.pdf"]


def test_matching_hash_without_indexed_chunks_does_not_skip_rebuild(tmp_path):
    docs = tmp_path / "documents"
    _write(docs, "a.pdf", "legacy policy")
    parser = SwitchingParser()

    pipeline, retriever = _compose(tmp_path, parser=parser)
    assert pipeline.ingest(docs) == 0

    parser.emit_blocks = True
    assert pipeline.ingest(docs) == 1

    results = retriever.retrieve("legacy")
    assert [chunk.document for chunk in results] == ["a.pdf"]


def test_factory_persists_and_loads_knowledge_base_from_created_directories(
    tmp_path, monkeypatch
):
    docs = tmp_path / "documents"
    _write(docs, "a.pdf", "legacy policy")
    config = Config(
        documents_dir=docs,
        index_path=tmp_path / "nested" / "faiss" / "index.faiss",
        db_path=tmp_path / "nested" / "sqlite" / "metadata.sqlite",
        embed_dim=len(KeywordEmbedder.terms),
        top_k=1,
    )
    monkeypatch.setattr("rag.factory._parser", lambda _config: StubParser())
    monkeypatch.setattr("rag.factory._embedder", lambda _config: KeywordEmbedder())
    monkeypatch.setattr(
        "rag.factory.OllamaGenerator",
        lambda model, host: FakeGenerator(),
    )

    assert run_ingestion(config) == 1

    assert config.index_path.exists()
    assert config.db_path.exists()
    answer = load_pipeline(config).ask("legacy")
    assert [chunk.document for chunk in answer.sources] == ["a.pdf"]
