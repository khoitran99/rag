"""Generation seam.

`Generator` is the interface the pipeline depends on; tests inject a deterministic fake.
`OllamaGenerator` is the real, fully-offline adapter (Qwen2.5-7B-Instruct via Ollama),
non-deterministic and left untested per the project's testing decisions.
"""

from __future__ import annotations

from typing import Protocol


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


class OllamaGenerator:
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self._host = host

    def generate(self, prompt: str) -> str:
        import ollama  # lazy import: only needed for real runs

        client = ollama.Client(host=self._host)
        response = client.generate(model=self.model, prompt=prompt)
        return response["response"]
