"""Chapter 2 - Prompt evaluation.

The course's eval workflow:

  1. draft a prompt
  2. generate a test dataset ............ generate_dataset()  (Claude writes cases)
  3. run the prompt on each case ........ run_case()
  4. grade each output
       - code grader .................... grade_by_code()      (format/syntax checks)
       - model grader ................... grade_by_model()     (rubric -> score + reasoning)
  5. aggregate and compare versions ..... run_eval() -> EvalReport.average
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from claude_api_lab import llm

Format = Literal["python", "json", "regex", "text"]


class EvalCase(BaseModel):
    task: str = Field(description="The request a user would send.")
    format: Format = Field(description="The format the answer must be in.")
    solution_criteria: str = Field(description="What a fully correct answer must contain.")


class Dataset(BaseModel):
    cases: list[EvalCase]


class ModelGrade(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: int = Field(ge=1, le=10)


@dataclass
class CaseResult:
    case: EvalCase
    output: str
    score: float
    reasoning: str


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def average(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "average": round(self.average, 2),
            "results": [
                {"task": r.case.task, "score": r.score, "reasoning": r.reasoning, "output": r.output}
                for r in self.results
            ],
        }


def generate_dataset(client: Any, topic: str, n: int = 3) -> Dataset:
    prompt = (
        f"Generate {n} distinct evaluation tasks for this area: {topic}.\n"
        "Each task must be answerable with one short piece of Python, JSON, or a regex, "
        "and solvable in under a minute. Vary the format across tasks."
    )
    return llm.extract(client, prompt, Dataset, model=llm.FAST_MODEL)


def save_dataset(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset.model_dump_json(indent=2))


def load_dataset(path: Path) -> Dataset:
    return Dataset.model_validate_json(path.read_text())


def grade_by_code(output: str, fmt: Format) -> float:
    """10 if the output parses in its declared format, else 0."""
    text = strip_fences(output)
    try:
        if fmt == "json":
            json.loads(text)
        elif fmt == "python":
            ast.parse(text)
        elif fmt == "regex":
            re.compile(text)
    except (ValueError, SyntaxError, re.error):
        return 0.0
    return 10.0


def strip_fences(text: str) -> str:
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def grade_by_model(client: Any, case: EvalCase, output: str) -> ModelGrade:
    prompt = (
        "You are an expert reviewer. Grade the solution against the criteria.\n\n"
        f"<task>\n{case.task}\n</task>\n\n"
        f"<solution>\n{output}\n</solution>\n\n"
        f"<criteria>\n{case.solution_criteria}\n</criteria>\n\n"
        "Score 1-10. Be strict: a 10 meets every criterion with nothing extra."
    )
    return llm.extract(client, prompt, ModelGrade, model=llm.FAST_MODEL)


def run_case(client: Any, case: EvalCase, prompt_fn: Callable[[EvalCase], str]) -> CaseResult:
    output = llm.ask(client, prompt_fn(case), effort="low")
    code_score = grade_by_code(output, case.format)
    grade = grade_by_model(client, case, output)
    score = (code_score + grade.score) / 2
    return CaseResult(case=case, output=output, score=score, reasoning=grade.reasoning)


def run_eval(
    client: Any, dataset: Dataset, prompt_fn: Callable[[EvalCase], str], concurrency: int = 4
) -> EvalReport:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda c: run_case(client, c, prompt_fn), dataset.cases))
    return EvalReport(results=results)


def baseline_prompt(case: EvalCase) -> str:
    return case.task


def engineered_prompt(case: EvalCase) -> str:
    return (
        f"Solve the task below.\n\n<task>\n{case.task}\n</task>\n\n"
        f"Respond with only the {case.format} answer: no explanation, no markdown fences."
    )
