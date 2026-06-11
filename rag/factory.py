"""Composition root: wire real (Ollama-backed) components from Config.

Kept separate from the modules themselves so tests can compose the same pipeline with
fakes instead. The CLI and the Streamlit UI both build through here.
"""

from __future__ import annotations

from rag.ai_parser import AiLayoutParser
from rag.chunker import Chunker
from rag.config import Config
from rag.embedder import OllamaEmbedder
from rag.generator import OllamaGenerator
from rag.ingestion import IngestionPipeline
from rag.metadata_store import MetadataStore
from rag.parser import PdfParser
from rag.pipeline import RagPipeline
from rag.prompt_builder import PromptBuilder
from rag.retriever import Retriever
from rag.vector_index import VectorIndex


def _embedder(config: Config) -> OllamaEmbedder:
    return OllamaEmbedder(model=config.embed_model, dim=config.embed_dim, host=config.ollama_host)


def _parser(config: Config):
    """Quick-path PyMuPDF by default; AI-based Docling layout+OCR when RAG_PARSER=ai."""
    if config.parser == "ai":
        from rag.layout_detector import DoclingLayoutDetector  # lazy: heavy optional dep

        return AiLayoutParser(DoclingLayoutDetector())
    return PdfParser()


def run_ingestion(config: Config) -> int:
    """Ingest the documents folder and persist the index + metadata to disk."""
    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    index = VectorIndex(dim=config.embed_dim)
    store = MetadataStore(config.db_path)
    pipeline = IngestionPipeline(
        parser=_parser(config),
        chunker=Chunker(
            max_tokens=config.chunk_max_tokens, overlap_tokens=config.chunk_overlap_tokens
        ),
        embedder=_embedder(config),
        index=index,
        store=store,
    )
    count = pipeline.ingest(config.documents_dir)
    index.persist(config.index_path)
    store.close()
    return count


def load_pipeline(config: Config) -> RagPipeline:
    """Build the query-time pipeline from the persisted index + metadata."""
    index = VectorIndex.load(config.index_path)
    store = MetadataStore(config.db_path)
    retriever = Retriever(
        embedder=_embedder(config), index=index, store=store, top_k=config.top_k
    )
    return RagPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=OllamaGenerator(model=config.llm_model, host=config.ollama_host),
    )
