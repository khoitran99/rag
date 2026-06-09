"""Command-line entry: `rag ingest` and `rag query`.

Thin glue over the factory. The Streamlit UI is the primary chat surface; the CLI exists
for scripting ingestion and one-off queries.
"""

from __future__ import annotations

import argparse

from rag.config import Config
from rag.factory import load_pipeline, run_ingestion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Local offline RAG over a PDF folder")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Parse, chunk, embed and index the documents folder")
    q = sub.add_parser("query", help="Ask a one-off question")
    q.add_argument("question")

    args = parser.parse_args(argv)
    config = Config.from_env()

    if args.command == "ingest":
        count = run_ingestion(config)
        print(f"Indexed {count} chunks from {config.documents_dir}/")
        return 0

    if args.command == "query":
        answer = load_pipeline(config).ask(args.question)
        print(answer.text)
        print("\nSources:")
        for i, src in enumerate(answer.sources, start=1):
            print(f"  [{i}] {src.document} p.{src.page}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
