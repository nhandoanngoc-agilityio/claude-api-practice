"""Extended thinking.

On current models thinking is adaptive: Claude decides how much to think,
and `effort` sets how hard it tries. The fixed `budget_tokens` from the course
is rejected on these models. `display: "summarized"` returns a readable
summary of the reasoning (the default leaves it empty).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from claude_api_lab import llm


@dataclass
class Thought:
    reasoning: str
    answer: str


def think(client: Any, problem: str, effort: llm.Effort = "high") -> Thought:
    response = llm.send(
        client,
        problem,
        thinking={"type": "adaptive", "display": "summarized"},
        effort=effort,
    )
    reasoning = "\n".join(b.thinking for b in response.content if b.type == "thinking" and b.thinking)
    return Thought(reasoning=reasoning, answer=llm.text_of(response))
