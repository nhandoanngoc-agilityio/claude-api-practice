"""Embeddings.

Anthropic doesn't serve embeddings; the course uses Voyage AI. `VoyageEmbedder`
does that when `voyageai` is installed and VOYAGE_API_KEY is set. Otherwise
`HashingEmbedder` gives a deterministic, offline bag-of-words vector, which is
enough to run and test the whole pipeline.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class Embedder(Protocol):
    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]: ...


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class HashingEmbedder:
    def __init__(self, dims: int = 512) -> None:
        self.dims = dims

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dims
            for tok in tokenize(text):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                vec[h % self.dims] += 1.0 if (h >> 63) & 1 else -1.0
            out.append(normalize(vec))
        return out


class VoyageEmbedder:
    def __init__(self, model: str = "voyage-3-large") -> None:
        import voyageai  # optional dependency: `uv sync --extra voyage`

        self.client = voyageai.Client()
        self.model = model

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        return self.client.embed(texts, model=self.model, input_type=input_type).embeddings


def default_embedder() -> Embedder:
    if os.environ.get("VOYAGE_API_KEY"):
        try:
            return VoyageEmbedder()
        except ImportError:
            pass
    return HashingEmbedder()
