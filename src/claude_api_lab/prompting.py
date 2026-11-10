"""Chapter 3 - Prompt engineering.

A small builder that applies the course's techniques in a fixed order, so
each one shows up as a separate, testable piece of the final prompt:

  1. be clear and direct ......... `task` leads with an action verb
  2. be specific ................. `guidelines` (qualities) and `steps` (process)
  3. structure with XML tags ..... `context(...)` wraps each input in its own tag
  4. provide examples ............ `example(...)` one/multi-shot, with optional reasoning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape


def xml(tag: str, body: str, **attrs: str) -> str:
    attr_text = "".join(f' {k}="{escape(v, quote=True)}"' for k, v in attrs.items())
    return f"<{tag}{attr_text}>\n{body.strip()}\n</{tag}>"


@dataclass
class Example:
    input: str
    output: str
    why: str | None = None  # explain what makes the output ideal


@dataclass
class PromptBuilder:
    task: str
    guidelines: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    contexts: list[tuple[str, str]] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)
    output_format: str | None = None

    def guideline(self, text: str) -> PromptBuilder:
        self.guidelines.append(text)
        return self

    def step(self, text: str) -> PromptBuilder:
        self.steps.append(text)
        return self

    def context(self, tag: str, body: str) -> PromptBuilder:
        self.contexts.append((tag, body))
        return self

    def example(self, input: str, output: str, why: str | None = None) -> PromptBuilder:
        self.examples.append(Example(input, output, why))
        return self

    def build(self) -> str:
        parts = [self.task.strip()]
        for tag, body in self.contexts:
            parts.append(xml(tag, body))
        if self.guidelines:
            parts.append("Guidelines:\n" + "\n".join(f"- {g}" for g in self.guidelines))
        if self.steps:
            parts.append(
                "Follow these steps:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(self.steps, 1))
            )
        if self.examples:
            rendered = []
            for ex in self.examples:
                body = xml("sample_input", ex.input) + "\n" + xml("ideal_output", ex.output)
                if ex.why:
                    body += "\n" + xml("why_ideal", ex.why)
                rendered.append(xml("example", body))
            parts.append("Here are examples of good answers:\n" + "\n".join(rendered))
        if self.output_format:
            parts.append(self.output_format.strip())
        return "\n\n".join(parts)


# The production prompt that the eval chapter scores. Version 1 is the naive
# prompt the course starts from; version 2 applies every technique above.
def doc_qa_prompt_v1(question: str, document: str) -> str:
    return f"{document}\n\n{question}"


def doc_qa_prompt_v2(question: str, document: str) -> str:
    return (
        PromptBuilder(task="Answer the question using only the document provided.")
        .context("document", document)
        .context("question", question)
        .guideline("Quote figures exactly as they appear in the document.")
        .guideline('If the document does not contain the answer, say "Not in the document."')
        .guideline("Keep the answer under 80 words.")
        .step("Find every passage relevant to the question.")
        .step("Check whether those passages fully answer it.")
        .step("Write the answer, citing the section heading it came from.")
        .example(
            input="What was Q3 revenue?",
            output="Q3 revenue was $4.2M (section: Financial Summary).",
            why="It quotes the figure exactly and names its section.",
        )
        .build()
    )
