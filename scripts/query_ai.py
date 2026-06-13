"""Query the AI-parsed store (Slice 3 manual verification).

Loads vector_store_ai/ (built by scripts/ingest_ai.py) and answers a question, so you can
compare against the quick-path store without fighting shell env-var quoting/paste splits.
Query time loads faiss + Ollama but not torch, so no OMP flag is needed.

Run:  .venv/bin/python scripts/query_ai.py "How does RAG retrieval work?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import Config
from rag.factory import load_pipeline

question = " ".join(sys.argv[1:]) or "How does RAG retrieval work?"

config = Config(
    index_path=Path("vector_store_ai/index.faiss"),
    db_path=Path("vector_store_ai/metadata.sqlite"),
)

answer = load_pipeline(config).ask(question)
print(answer.text)
print("\nSources:")
for i, src in enumerate(answer.sources, start=1):
    print(f"  [{i}] {src.document} p.{src.page}")
