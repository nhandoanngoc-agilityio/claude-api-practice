"""The manual tool-use loop, plus the course's "batch tool".

Course techniques:
  * handling tool_use blocks and returning tool_result blocks
  * multi-turn: keep calling until stop_reason is no longer "tool_use"
  * multiple tools in one turn: run them all, return ALL results in ONE user message
  * errors go back to Claude as `is_error: True` rather than crashing the loop
  * batch tool: one tool that runs several tool calls, nudging Claude to
    parallelise (the API also emits parallel tool_use blocks natively)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from claude_api_lab import llm

Handler = Callable[..., str]

BATCH_TOOL = {
    "name": "batch_tool",
    "description": (
        "Invoke several other tools at once. Use it whenever two or more tool calls do not "
        "depend on each other's results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "invocations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Tool to call."},
                        "arguments": {"type": "string", "description": "JSON-encoded tool arguments."},
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["invocations"],
        "additionalProperties": False,
    },
}


@dataclass
class ToolBox:
    """Maps tool names to handlers and runs tool_use blocks."""

    schemas: list[dict[str, Any]]
    handlers: dict[str, Handler]
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def with_batch(self) -> ToolBox:
        return ToolBox(self.schemas + [BATCH_TOOL], {**self.handlers, "batch_tool": self._batch})

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        handler = self.handlers.get(name)
        if handler is None:
            raise KeyError(f"unknown tool {name!r}")
        return handler(**arguments)

    def _batch(self, invocations: list[dict[str, str]]) -> str:
        results = []
        for inv in invocations:
            try:
                output = self.run(inv["name"], json.loads(inv["arguments"]))
                results.append({"tool": inv["name"], "output": output})
            except Exception as exc:  # noqa: BLE001 - reported back to Claude
                results.append({"tool": inv["name"], "error": str(exc)})
        return json.dumps(results)

    def result_block(self, block: Any) -> dict[str, Any]:
        try:
            content = self.run(block.name, dict(block.input))
            return {"type": "tool_result", "tool_use_id": block.id, "content": content}
        except Exception as exc:  # noqa: BLE001 - reported back to Claude
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Error: {exc}",
                "is_error": True,
            }


def run_tool_loop(
    client: Any,
    toolbox: ToolBox,
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    max_turns: int = 10,
    **extra: Any,
) -> Any:
    """Call Claude until it stops asking for tools; returns the final message.

    `messages` is extended in place, so callers keep the whole transcript.
    """
    for _ in range(max_turns):
        response = llm.send(client, messages, system=system, tools=toolbox.schemas, **extra)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            continue  # a server tool paused; re-send so it resumes
        if response.stop_reason != "tool_use":
            return response
        results = [toolbox.result_block(b) for b in response.content if b.type == "tool_use"]
        messages.append({"role": "user", "content": results})
    raise RuntimeError(f"tool loop did not finish within {max_turns} turns")
