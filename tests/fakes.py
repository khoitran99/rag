"""Deterministic test doubles for the LLM-backed seams (Embedder, Generator).

Per the project's testing decisions, the real Ollama adapters are non-deterministic and
left untested; the pipeline is exercised through these fakes injected in their place.
"""

from __future__ import annotations

import hashlib

import numpy as np


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
