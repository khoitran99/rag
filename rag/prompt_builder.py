"""Prompt assembly for the generation step.

Slice 1 builds a basic prompt: positionally-numbered context + question + a citation
instruction. Slice 5 upgrades this to the full Figure 22 composite (role-specific framing,
chain-of-thought, few-shot citation examples). The `build(query, chunks)` shape is stable.
"""

from __future__ import annotations

from rag.models import Chunk


class PromptBuilder:
    def build(self, query: str, chunks: list[Chunk]) -> str:
        context_lines = [
            f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks, start=1)
        ]
        context = "\n".join(context_lines)
        return (
            "Answer the question using only the context below. "
            "Cite the context you use with its bracketed number, e.g. [1].\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
