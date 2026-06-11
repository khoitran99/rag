"""Core domain types. See CONTEXT.md for the glossary these names come from."""

from __future__ import annotations

from dataclasses import dataclass


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
