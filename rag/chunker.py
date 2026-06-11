"""Structure-aware document chunking.

Rides on the layout structure the parser produces: a heading is kept with the body that
follows it, a table is kept intact, and only oversized body content is split (with token
overlap). Size is measured in real tokens (tiktoken cl100k_base) as a proxy for the
embedder's window. Pure: ``chunk([LayoutBlock]) -> [Chunk]`` with positionally-assigned ids.
"""

from __future__ import annotations

import tiktoken

from rag.models import Chunk, LayoutBlock

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


class Chunker:
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, blocks: list[LayoutBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        group: list[LayoutBlock] = []

        def flush() -> None:
            if group:
                chunks.extend(self._emit(group, start_id=len(chunks)))
                group.clear()

        for block in blocks:
            if block.kind == "table":
                # A table is its own chunk: flush what precedes it, emit it alone.
                flush()
                group.append(block)
                flush()
                continue

            starts_section = block.kind == "heading"
            different_source = bool(group) and (
                block.document != group[0].document or block.page != group[0].page
            )
            would_overflow = bool(group) and _token_count(
                self._join(group + [block])
            ) > self.max_tokens

            if starts_section or different_source or would_overflow:
                flush()
            group.append(block)

        flush()
        return chunks

    def _emit(self, group: list[LayoutBlock], start_id: int) -> list[Chunk]:
        first = group[0]
        joined = self._join(group)
        # Tables are kept intact; everything else is split to the token budget.
        texts = [joined] if first.kind == "table" else self._split_to_budget(joined)
        return [
            Chunk(chunk_id=start_id + i, document=first.document, page=first.page, text=text)
            for i, text in enumerate(texts)
        ]

    def _split_to_budget(self, text: str) -> list[str]:
        tokens = _ENCODING.encode(text)
        if len(tokens) <= self.max_tokens:
            return [text]
        step = self.max_tokens - self.overlap_tokens
        pieces: list[str] = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.max_tokens]
            pieces.append(_ENCODING.decode(window))
            if start + self.max_tokens >= len(tokens):
                break
        return pieces

    @staticmethod
    def _join(blocks: list[LayoutBlock]) -> str:
        return "\n".join(b.text for b in blocks)
