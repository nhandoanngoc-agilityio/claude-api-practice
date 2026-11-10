"""The four workflow patterns from the course, applied to writing about documents.

* chaining ............. draft -> critique against rules -> revise (one focused call each)
* routing .............. classify the request, then use the specialist prompt for that category
* parallelization ...... run independent sub-tasks concurrently, then aggregate
* evaluator-optimizer .. generate, grade with a rubric, feed the feedback back until it passes
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from claude_api_lab import llm

# --- chaining -------------------------------------------------------------

STYLE_RULES = [
    "No marketing clichés such as 'game-changer', 'revolutionize' or 'unlock'.",
    "No em dashes.",
    "Every claim about numbers must quote the number.",
]


def chain_write(client: Any, topic: str, source: str) -> dict[str, str]:
    """Each step does one job, so each is easy to prompt, test and debug."""
    draft = llm.ask(
        client,
        f"<source>\n{source}\n</source>\n\nWrite a 150-word briefing on: {topic}",
    )
    rules = "\n".join(f"- {r}" for r in STYLE_RULES)
    critique = llm.ask(
        client,
        f"<draft>\n{draft}\n</draft>\n\n<rules>\n{rules}\n</rules>\n\n"
        "List every place the draft breaks a rule. If none, reply exactly NONE.",
        effort="low",
    )
    if critique.strip() == "NONE":
        return {"draft": draft, "critique": critique, "final": draft}
    final = llm.ask(
        client,
        f"<draft>\n{draft}\n</draft>\n\n<problems>\n{critique}\n</problems>\n\n"
        "Rewrite the draft fixing every problem. Reply with only the rewritten text.",
    )
    return {"draft": draft, "critique": critique, "final": final}


# --- routing --------------------------------------------------------------

Category = Literal["summary", "data_question", "rewrite", "other"]


class Route(BaseModel):
    category: Category
    reason: str


SPECIALISTS: dict[str, str] = {
    "summary": "You write tight executive summaries: five bullets maximum, most important first.",
    "data_question": "You answer numeric questions precisely, quoting figures and showing any arithmetic.",
    "rewrite": "You are an editor. Keep the meaning, improve clarity, and match the requested tone.",
    "other": "You are a helpful assistant for questions about business documents.",
}


def route(client: Any, request: str, document: str) -> tuple[Route, str]:
    decision = llm.extract(
        client,
        f"Classify this request about a document.\n<request>\n{request}\n</request>",
        Route,
        model=llm.FAST_MODEL,
        effort="low",
    )
    answer = llm.ask(
        client,
        f"<document>\n{document}\n</document>\n\n{request}",
        system=SPECIALISTS[decision.category],
    )
    return decision, answer


# --- parallelization ------------------------------------------------------

LENSES = {
    "finance": "Assess the financial health described in the document.",
    "risk": "List the main risks the document reveals or implies.",
    "people": "Assess the team, hiring and organisational signals in the document.",
}


def parallel_review(client: Any, document: str, max_workers: int = 3) -> dict[str, str]:
    def review(lens: str) -> str:
        return llm.ask(
            client, f"<document>\n{document}\n</document>\n\n{LENSES[lens]} Keep it under 100 words."
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        parts = dict(zip(LENSES, pool.map(review, LENSES), strict=True))
    joined = "\n".join(f'<review lens="{k}">\n{v}\n</review>' for k, v in parts.items())
    parts["overall"] = llm.ask(
        client, f"{joined}\n\nCombine these reviews into a single three-sentence verdict."
    )
    return parts


# --- evaluator-optimizer --------------------------------------------------


class Verdict(BaseModel):
    passed: bool
    score: int = Field(ge=1, le=10)
    feedback: str


@dataclass
class Optimized:
    output: str
    attempts: list[tuple[str, Verdict]] = field(default_factory=list)


def evaluate_optimize(
    client: Any, task: str, rubric: list[str], *, max_rounds: int = 3, pass_score: int = 8
) -> Optimized:
    criteria = "\n".join(f"- {c}" for c in rubric)
    result = Optimized(output="")
    feedback = ""
    for _ in range(max_rounds):
        prompt = f"<task>\n{task}\n</task>"
        if feedback:
            prompt += (
                f"\n\n<previous_attempt>\n{result.output}\n</previous_attempt>"
                f"\n<feedback>\n{feedback}\n</feedback>"
            )
        result.output = llm.ask(client, prompt + "\n\nReply with only the deliverable.")
        verdict = llm.extract(
            client,
            f"<task>\n{task}\n</task>\n<output>\n{result.output}\n</output>\n<rubric>\n{criteria}\n</rubric>\n\n"
            f"Grade the output against the rubric. passed is true only if score >= {pass_score}.",
            Verdict,
            model=llm.FAST_MODEL,
        )
        result.attempts.append((result.output, verdict))
        if verdict.passed and verdict.score >= pass_score:
            break
        feedback = verdict.feedback
    return result
