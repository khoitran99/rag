"""Text encoder seam.

`Embedder` is the interface the pipeline depends on; tests inject a deterministic fake.
`OllamaEmbedder` is the real, fully-offline adapter (nomic-embed-text via Ollama). Per the
project's testing decisions this adapter is non-deterministic and left untested — it is
verified manually once Ollama is running. See ADR-0001 for why a dedicated embedder is used
instead of the lecture's CLIP text encoder.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OllamaEmbedder:
    def __init__(self, model: str, dim: int, host: str = "http://localhost:11434"):
        self.model = model
        self.dim = dim
        self._host = host

    def embed(self, texts: list[str]) -> np.ndarray:
        import ollama  # lazy import: only needed for real runs

        client = ollama.Client(host=self._host)
        rows = [client.embeddings(model=self.model, prompt=t)["embedding"] for t in texts]
        return np.asarray(rows, dtype="float32")
