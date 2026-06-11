"""Deterministic test doubles for the heavy/non-deterministic seams.

Per the project's testing decisions, the real backends (Ollama for embedding/generation,
Docling for layout detection) are non-deterministic and left untested; the pipeline and the
AI-based parser are exercised through these fakes injected in their place.
"""

from __future__ import annotations

import hashlib

import numpy as np

from rag.layout_detector import DetectedRegion


class FakeEmbedder:
    """Maps text -> a deterministic unit-ish vector derived from a hash of the text.

    Same text always yields the same vector, and distinct texts yield distinct vectors,
    which is all the pipeline needs to be exercised deterministically.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(seed)
            rows.append(rng.standard_normal(self.dim))
        return np.asarray(rows, dtype="float32")


class FakeGenerator:
    """Echoes the prompt's question and cites the first source, deterministically."""

    def generate(self, prompt: str) -> str:
        return "Answer grounded in the provided context [1]."


class FakeLayoutDetector:
    """Returns a canned list of DetectedRegions, ignoring the actual PDF.

    Stands in for the heavy Docling/LayoutParser backend so the AiLayoutParser's real logic
    (label->kind mapping, reading-order sorting, source stamping) can be tested deterministically.
    """

    def __init__(self, regions: list[DetectedRegion]):
        self._regions = regions

    def detect(self, pdf_path) -> list[DetectedRegion]:
        return list(self._regions)
