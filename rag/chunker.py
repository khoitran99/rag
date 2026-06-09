"""Document chunking.

Slice 1 uses a naive length-based splitter. Slice 2 replaces it with the structure-aware
Chunker that rides on LayoutParser blocks; this naive splitter is the placeholder seam.
"""

from __future__ import annotations


class NaiveChunker:
    """Splits text into fixed-size character windows with overlap.

    Pure and deterministic. Chunk ids and source metadata are assigned by the ingestion
    pipeline, so this class only deals in text.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        step = self.chunk_size - self.overlap
        pieces = []
        for start in range(0, len(text), step):
            piece = text[start : start + self.chunk_size].strip()
            if piece:
                pieces.append(piece)
            if start + self.chunk_size >= len(text):
                break
        return pieces
