"""Retrievers: semantic (vector), lexical (BM25), and hybrid (reciprocal rank fusion).

Each retriever returns `(chunk, score)` pairs ranked best-first, so they can
be swapped, combined, and reranked without knowing about each other.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from claude_api_lab.rag.chunking import Chunk
from claude_api_lab.rag.embeddings import Embedder, tokenize

Hit = tuple[Chunk, float]


class Retriever(Protocol):
    def add(self, chunks: list[Chunk], texts: list[str] | None = None) -> None: ...
    def search(self, query: str, k: int = 5) -> list[Hit]: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class VectorIndex:
    embedder: Embedder
    chunks: list[Chunk] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def add(self, chunks: list[Chunk], texts: list[str] | None = None) -> None:
        """`texts` lets contextual retrieval embed an enriched version of each chunk."""
        self.chunks += chunks
        self.vectors += self.embedder.embed(texts or [c.text for c in chunks], input_type="document")

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self.chunks:
            return []
        q = self.embedder.embed([query], input_type="query")[0]
        scored = [(c, cosine(q, v)) for c, v in zip(self.chunks, self.vectors, strict=True)]
        return sorted(scored, key=lambda h: h[1], reverse=True)[:k]


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    chunks: list[Chunk] = field(default_factory=list)
    docs: list[Counter[str]] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    df: Counter[str] = field(default_factory=Counter)

    def add(self, chunks: list[Chunk], texts: list[str] | None = None) -> None:
        for chunk, text in zip(chunks, texts or [c.text for c in chunks], strict=True):
            tokens = tokenize(text)
            self.chunks.append(chunk)
            self.docs.append(Counter(tokens))
            self.lengths.append(len(tokens))
            self.df.update(set(tokens))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        n = len(self.chunks)
        if not n:
            return []
        avg = sum(self.lengths) / n or 1.0
        scores = []
        for chunk, tf, length in zip(self.chunks, self.docs, self.lengths, strict=True):
            score = 0.0
            for term in set(tokenize(query)):
                if term not in tf:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                f = tf[term]
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / avg))
            scores.append((chunk, score))
        return sorted(scores, key=lambda h: h[1], reverse=True)[:k]


@dataclass
class HybridRetriever:
    """Runs every retriever and merges their rankings with reciprocal rank fusion."""

    retrievers: list[Retriever]
    rrf_k: int = 60

    def add(self, chunks: list[Chunk], texts: list[str] | None = None) -> None:
        for r in self.retrievers:
            r.add(chunks, texts)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return reciprocal_rank_fusion([r.search(query, k * 2) for r in self.retrievers], self.rrf_k)[:k]


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    fused: dict[tuple[str, int], float] = {}
    by_key: dict[tuple[str, int], Chunk] = {}
    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking, start=1):
            key = (chunk.source, chunk.id)
            by_key[key] = chunk
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(((by_key[key], s) for key, s in fused.items()), key=lambda h: h[1], reverse=True)
