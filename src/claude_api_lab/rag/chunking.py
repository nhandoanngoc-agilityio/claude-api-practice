"""Three chunking strategies from the course.

* by size ....... fixed character windows with overlap (works on anything)
* by sentence ... groups of sentences with sentence overlap
* by section .... split on markdown headings (best when the document has structure)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: int
    text: str
    source: str = ""


def chunk_by_size(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    step = size - overlap
    return [
        text[i : i + size] for i in range(0, max(len(text) - overlap, 1), step) if text[i : i + size].strip()
    ]


def chunk_by_sentence(text: str, per_chunk: int = 5, overlap: int = 1) -> list[str]:
    if per_chunk <= 0 or not 0 <= overlap < per_chunk:
        raise ValueError("need per_chunk > 0 and 0 <= overlap < per_chunk")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    step = per_chunk - overlap
    return [" ".join(sentences[i : i + per_chunk]) for i in range(0, max(len(sentences) - overlap, 1), step)]


def chunk_by_section(text: str) -> list[str]:
    parts = re.split(r"\n(?=#{1,6} )", "\n" + text)
    return [p.strip() for p in parts if p.strip()]


def make_chunks(text: str, source: str = "", strategy: str = "section") -> list[Chunk]:
    splitter = {"section": chunk_by_section, "size": chunk_by_size, "sentence": chunk_by_sentence}[strategy]
    return [Chunk(i, t, source) for i, t in enumerate(splitter(text))]
