"""The full RAG pipeline: contextual retrieval -> hybrid search -> rerank -> answer.

* contextual retrieval ... before indexing, Claude writes a sentence placing
                           each chunk in its document; that sentence is
                           prepended to what gets embedded and BM25-indexed.
                           The whole document sits in a cached prefix, so
                           only the first chunk pays full input price.
* reranking .............. Claude orders the hybrid search candidates by
                           relevance (structured output: a list of ids).
* grounded answer ........ the top chunks go into the prompt in XML tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from claude_api_lab import llm
from claude_api_lab.rag.chunking import Chunk, make_chunks
from claude_api_lab.rag.embeddings import Embedder, default_embedder
from claude_api_lab.rag.search import BM25Index, Hit, HybridRetriever, VectorIndex


class Ranking(BaseModel):
    ids: list[int]


def situate_chunk(client: Any, document: str, chunk: Chunk) -> str:
    system = [
        {"type": "text", "text": "You place excerpts in the context of their source document."},
        {
            "type": "text",
            "text": f"<document>\n{document}\n</document>",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    prompt = (
        f"<chunk>\n{chunk.text}\n</chunk>\n\n"
        "Write one short sentence that situates this chunk within the document, to improve "
        "search retrieval. Answer with only that sentence."
    )
    return llm.ask(client, prompt, system=system, model=llm.FAST_MODEL, max_tokens=300)


def rerank(client: Any, query: str, hits: list[Hit], k: int) -> list[Hit]:
    if len(hits) <= 1:
        return hits[:k]
    listing = "\n".join(f'<doc id="{i}">\n{c.text}\n</doc>' for i, (c, _) in enumerate(hits))
    prompt = (
        f"<question>\n{query}\n</question>\n\n<documents>\n{listing}\n</documents>\n\n"
        f"Return the ids of the {k} documents most relevant to the question, most relevant first."
    )
    ranking = llm.extract(client, prompt, Ranking, model=llm.FAST_MODEL, effort="low")
    seen: set[int] = set()
    ordered = []
    for i in ranking.ids:
        if 0 <= i < len(hits) and i not in seen:
            seen.add(i)
            ordered.append(hits[i])
    return ordered[:k]


@dataclass
class RagPipeline:
    client: Any
    embedder: Embedder = field(default_factory=default_embedder)
    contextual: bool = True
    use_reranker: bool = True

    def __post_init__(self) -> None:
        self.retriever = HybridRetriever([VectorIndex(self.embedder), BM25Index()])

    def add_document(self, text: str, source: str, strategy: str = "section") -> list[Chunk]:
        chunks = make_chunks(text, source, strategy)
        texts = None
        if self.contextual:
            texts = [f"{situate_chunk(self.client, text, c)}\n\n{c.text}" for c in chunks]
        self.retriever.add(chunks, texts)
        return chunks

    def retrieve(self, query: str, k: int = 3) -> list[Hit]:
        candidates = self.retriever.search(query, k * 3)
        return rerank(self.client, query, candidates, k) if self.use_reranker else candidates[:k]

    def answer(self, query: str, k: int = 3) -> str:
        hits = self.retrieve(query, k)
        context = "\n".join(f'<chunk source="{c.source}">\n{c.text}\n</chunk>' for c, _ in hits)
        prompt = (
            f"<context>\n{context}\n</context>\n\n<question>\n{query}\n</question>\n\n"
            'Answer using only the context. If it is not there, say "Not in the knowledge base."'
        )
        return llm.ask(self.client, prompt)
