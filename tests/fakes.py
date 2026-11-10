"""A scripted stand-in for anthropic.Anthropic, so tests run offline and free."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any


def text(t: str, citations: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t, citations=citations)


def tool_use(name: str, input: dict[str, Any], id: str = "tu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input, id=id)


def thinking(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=t)


def message(*blocks: Any, stop_reason: str = "end_turn", **usage: int) -> SimpleNamespace:
    u = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    u.update(usage)
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(**u),
        container=SimpleNamespace(id="cont_1"),
    )


class _Stream:
    def __init__(self, msg: Any) -> None:
        self.msg = msg
        self.text_stream = iter([b.text for b in msg.content if b.type == "text"])

    def get_final_message(self) -> Any:
        return self.msg


class FakeMessages:
    def __init__(self, client: FakeClient) -> None:
        self.client = client

    def create(self, **params: Any) -> Any:
        return self.client._next("create", params)

    def parse(self, **params: Any) -> Any:
        parsed = self.client._next("parse", params)
        return SimpleNamespace(parsed_output=parsed, stop_reason="end_turn", stop_details=None, usage=None)

    @contextmanager
    def stream(self, **params: Any) -> Iterator[_Stream]:
        yield _Stream(self.client._next("stream", params))

    def count_tokens(self, **params: Any) -> Any:
        self.client.calls.append(("count_tokens", params))
        return SimpleNamespace(input_tokens=42)


class FakeClient:
    """Pops one scripted reply per call; records every call's params.

    A reply that is callable is called with the params, for replies that
    depend on the request (e.g. a reranker echoing ids).
    """

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.messages = FakeMessages(self)
        self.beta = SimpleNamespace(messages=self.messages)

    def _next(self, kind: str, params: dict[str, Any]) -> Any:
        self.calls.append((kind, params))
        if not self.replies:
            raise AssertionError(f"unexpected {kind} call: {params.get('messages', '')!r:.200}")
        reply = self.replies.pop(0)
        return reply(params) if callable(reply) else reply

    def last(self) -> dict[str, Any]:
        return self.calls[-1][1]
