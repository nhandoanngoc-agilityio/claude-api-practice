"""Images, PDFs and citations.

Content blocks other than text go in the user turn, before the question.
Citations make Claude point at the exact passage (or page) it relied on.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_api_lab import llm

IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def image_block(path: Path) -> dict[str, Any]:
    media_type = mimetypes.guess_type(path.name)[0]
    if media_type not in IMAGE_TYPES:
        raise ValueError(f"unsupported image type for {path.name}: {media_type}")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64(path)}}


def pdf_block(path: Path, *, citations: bool = False, title: str | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": b64(path)},
    }
    if title:
        block["title"] = title
    if citations:
        block["citations"] = {"enabled": True}
    return block


def text_document_block(text: str, title: str, *, citations: bool = True) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": text},
        "title": title,
    }
    if citations:
        block["citations"] = {"enabled": True}
    return block


def describe_image(client: Any, path: Path, question: str = "Describe this image.") -> str:
    content = [image_block(path), {"type": "text", "text": question}]
    return llm.ask(client, [{"role": "user", "content": content}])


def ask_pdf(client: Any, path: Path, question: str) -> str:
    content = [pdf_block(path), {"type": "text", "text": question}]
    return llm.ask(client, [{"role": "user", "content": content}])


@dataclass
class Cited:
    text: str
    quotes: list[str]
    where: list[str]


def ask_with_citations(client: Any, documents: list[dict[str, Any]], question: str) -> list[Cited]:
    """Return each answer span with the quotes and locations that support it."""
    content = [*documents, {"type": "text", "text": question}]
    response = llm.send(client, [{"role": "user", "content": content}])
    spans = []
    for block in response.content:
        if block.type != "text":
            continue
        cites = getattr(block, "citations", None) or []
        spans.append(Cited(block.text, [c.cited_text for c in cites], [location(c) for c in cites]))
    return spans


def location(citation: Any) -> str:
    title = getattr(citation, "document_title", None) or f"doc {citation.document_index}"
    match citation.type:
        case "page_location":
            first, last = citation.start_page_number, citation.end_page_number - 1  # end is exclusive
            return f"{title} p.{first}" if last <= first else f"{title} pp.{first}-{last}"
        case "char_location":
            return f"{title} chars {citation.start_char_index}-{citation.end_char_index}"
        case _:
            return title


def render_citations(spans: list[Cited]) -> str:
    out, notes = [], []
    for span in spans:
        marks = ""
        for quote, where in zip(span.quotes, span.where, strict=True):
            notes.append(f'[{len(notes) + 1}] {where}: "{quote.strip()}"')
            marks += f"[{len(notes)}]"
        out.append(span.text + marks)
    return "".join(out) + ("\n\n" + "\n".join(notes) if notes else "")
