"""Top-level RAG orchestrator: retrieve -> build prompt -> generate.

Ties the query-time components together and returns the answer alongside the cited source
chunks, so the UI/CLI can show what the answer was grounded in. The positional citation
numbering in the prompt matches the order of `Answer.sources` (i.e. [1] -> sources[0]).
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.models import Chunk


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Chunk]


class RagPipeline:
    def __init__(self, *, retriever, prompt_builder, generator):
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._generator = generator

    def ask(self, question: str) -> Answer:
        chunks = self._retriever.retrieve(question)
        prompt = self._prompt_builder.build(question, chunks)
        text = self._generator.generate(prompt)
        return Answer(text=text, sources=chunks)
