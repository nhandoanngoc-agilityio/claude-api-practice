import pytest
from fakes import FakeClient, message, text

from claude_api_lab.rag import chunking
from claude_api_lab.rag.chunking import Chunk
from claude_api_lab.rag.embeddings import HashingEmbedder
from claude_api_lab.rag.pipeline import RagPipeline, Ranking, rerank
from claude_api_lab.rag.search import BM25Index, HybridRetriever, VectorIndex, reciprocal_rank_fusion

DOC = """# Report

## Financial Summary
Q3 revenue was $4.2M with gross margin of 61%.

## Team
Headcount grew from 58 to 71 engineers and staff.

## Risks
One vendor supplies all servo motors.
"""


def test_chunk_by_size_overlaps_and_covers_text():
    text_ = "abcdefghij" * 10
    chunks = chunking.chunk_by_size(text_, size=30, overlap=10)
    assert chunks[0] == text_[:30] and chunks[1].startswith(text_[20:30])
    assert chunks[-1].endswith(text_[-5:])
    with pytest.raises(ValueError):
        chunking.chunk_by_size("x", size=10, overlap=10)


def test_chunk_by_sentence_and_section():
    assert chunking.chunk_by_sentence("A. B. C. D.", per_chunk=2, overlap=1) == ["A. B.", "B. C.", "C. D."]
    sections = chunking.chunk_by_section(DOC)
    assert [s.splitlines()[0] for s in sections] == [
        "# Report",
        "## Financial Summary",
        "## Team",
        "## Risks",
    ]


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(64)
    a, b = e.embed(["servo motor vendor", "servo motor vendor"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1) < 1e-9


def chunks():
    return chunking.make_chunks(DOC, "report.md")


def test_vector_and_bm25_find_the_right_section():
    for index in (VectorIndex(HashingEmbedder()), BM25Index()):
        index.add(chunks())
        top = index.search("servo motors vendor", k=1)[0][0]
        assert "servo" in top.text


def test_bm25_ranks_exact_term_matches_first():
    idx = BM25Index()
    idx.add(chunks())
    hits = idx.search("headcount", k=4)
    assert "Headcount" in hits[0][0].text and hits[0][1] > 0
    assert all(score == 0 for _, score in hits[1:])


def test_rrf_rewards_agreement():
    a, b, c = (Chunk(i, str(i), "s") for i in range(3))
    fused = reciprocal_rank_fusion([[(a, 1), (b, 1)], [(b, 1), (c, 1)]])
    assert fused[0][0] == b


def test_hybrid_returns_k_unique_chunks():
    h = HybridRetriever([VectorIndex(HashingEmbedder()), BM25Index()])
    h.add(chunks())
    hits = h.search("revenue margin", k=2)
    assert len(hits) == 2 and len({c.id for c, _ in hits}) == 2
    assert "revenue" in hits[0][0].text


def test_rerank_reorders_and_drops_bad_ids():
    hits = [(Chunk(i, f"c{i}"), 0.0) for i in range(3)]
    client = FakeClient(Ranking(ids=[2, 2, 9, 0]))
    assert [c.id for c, _ in rerank(client, "q", hits, 2)] == [2, 0]


def test_pipeline_contextual_indexing_caches_document_and_answers():
    n = len(chunks())
    situations = [message(text(f"Context {i}.")) for i in range(n)]
    client = FakeClient(*situations, Ranking(ids=[0]), message(text("61%")))
    rag = RagPipeline(client, embedder=HashingEmbedder())
    rag.add_document(DOC, "report.md")
    first_system = client.calls[0][1]["system"]
    assert first_system[-1]["cache_control"] == {"type": "ephemeral"}
    assert "<document>" in first_system[-1]["text"]
    assert rag.answer("What is gross margin?", k=1) == "61%"
    assert '<chunk source="report.md">' in client.last()["messages"][0]["content"]


def test_pipeline_without_context_or_reranker_makes_one_call():
    client = FakeClient(message(text("ok")))
    rag = RagPipeline(client, embedder=HashingEmbedder(), contextual=False, use_reranker=False)
    rag.add_document(DOC, "r")
    rag.answer("revenue")
    assert len(client.calls) == 1
