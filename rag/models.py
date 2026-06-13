"""Core domain types. See CONTEXT.md for the glossary these names come from."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutBlock:
    """A structural unit of a parsed Document: a heading, a body paragraph, or a table.

    The quick-path parser emits everything as ``kind="body"``; the AI-based parser (AiLayoutParser)
    fills in real headings and tables. Each block self-describes its source so chunking can
    stay a pure ``[LayoutBlock] -> [Chunk]`` function.
    """

    document: str
    page: int
    kind: str  # "heading" | "body" | "table"
    text: str


@dataclass(frozen=True)
class Chunk:
    """A contiguous piece of a parsed Document, the unit that gets embedded and indexed.

    `chunk_id` keys both the vector index and the metadata store; `document` and `page`
    are what citations resolve back to.
    """

    chunk_id: int
    document: str
    page: int
    text: str
