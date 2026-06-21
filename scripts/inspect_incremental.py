"""Manual inspection of incremental, hash-based ingestion (Slice 4).

Drives a temp documents/ folder through four re-ingests — first index, no-op, modify, remove —
and prints how many chunks each run (re)indexed and how many texts were embedded, so the
skip-unchanged / re-index-changed / drop-removed behavior is visible. No Ollama needed (a stub
parser and a counting fake embedder stand in for the real backends).

Run:  .venv/bin/python scripts/inspect_incremental.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import Chunker
from rag.ingestion import IngestionPipeline
from rag.metadata_store import MetadataStore
from rag.models import LayoutBlock
from rag.vector_index import VectorIndex
from tests.fakes import CountingEmbedder


class StubParser:
    def parse(self, pdf_path):
        text = open(pdf_path, encoding="utf-8").read()
        return [LayoutBlock(document=pdf_path.name, page=1, kind="body", text=text)]


def write(folder, name, text):
    (folder / name).write_text(text, encoding="utf-8")


tmp = Path(tempfile.mkdtemp())
docs = tmp / "documents"
docs.mkdir()
write(docs, "a.pdf", "Expense reports go in the portal.")
write(docs, "b.pdf", "Travel is reimbursed within fourteen days.")

embedder = CountingEmbedder(dim=16)
pipeline = IngestionPipeline(
    parser=StubParser(),
    chunker=Chunker(max_tokens=512, overlap_tokens=64),
    embedder=embedder,
    index=VectorIndex(dim=embedder.dim),
    store=MetadataStore(tmp / "meta.sqlite"),
)


def run(label):
    before = embedder.embedded_count
    n = pipeline.ingest(docs)
    docs_now = sorted(pipeline._store.documents())
    print(f"{label:<28} reindexed={n}  embedded_this_run={embedder.embedded_count - before}  docs={docs_now}")


run("1. first ingest (a,b):")
run("2. re-ingest, no changes:")
write(docs, "a.pdf", "Updated expense policy with new limits.")
run("3. modify a.pdf:")
(docs / "b.pdf").unlink()
run("4. remove b.pdf:")
