"""Prompt caching.

Caching is a prefix match over tools -> system -> messages. Put the large,
stable part first and mark the end of it with `cache_control`; everything
after the breakpoint (the question) can vary freely. Check the result in
`usage.cache_read_input_tokens`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from claude_api_lab import llm


@dataclass
class CachedDocumentChat:
    """Ask many questions about one large document, paying full price once."""

    client: Any
    document: str
    instructions: str = "Answer questions about the document. Quote it when you can."
    history: list[dict[str, Any]] = field(default_factory=list)

    def system(self) -> list[dict[str, Any]]:
        return [
            {"type": "text", "text": self.instructions},
            {
                "type": "text",
                "text": f"<document>\n{self.document}\n</document>",
                "cache_control": {"type": "ephemeral"},
            },
        ]

    def ask(self, question: str) -> tuple[str, dict[str, int]]:
        self.history.append({"role": "user", "content": question})
        # Top-level cache_control also caches the growing conversation, so each
        # turn reads the previous turns from cache too.
        response = llm.send(
            self.client, self.history, system=self.system(), cache_control={"type": "ephemeral"}
        )
        self.history.append({"role": "assistant", "content": response.content})
        u = response.usage
        stats = {
            "input": u.input_tokens,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        }
        return llm.text_of(response), stats
