"""Chapter 1 - Accessing the API.

The one place requests are built, so every chapter shares model choice,
refusal handling and token accounting.

Course technique -> code here:
  * making a request ............. send()
  * multi-turn conversations ..... Conversation
  * system prompts ............... Conversation(system=...) / send(system=...)
  * streaming .................... stream_text()
  * structured data .............. extract() (structured outputs; replaces prefill)
  * stop sequences ............... send(stop_sequences=[...])
  * "temperature" ................ effort= (sampling params are rejected on
                                   current models; effort is the dial now)
  * counting tokens / cost ....... count_tokens(), Usage
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic
from pydantic import BaseModel

MODEL = os.environ.get("LAB_MODEL", "claude-opus-5-5")
# A cheaper model for graders and bulk helpers (evals, contextual retrieval).
FAST_MODEL = os.environ.get("LAB_FAST_MODEL", "claude-haiku-5-5")

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# Server-side refusal fallback: on a policy decline the API re-runs the
# request on a fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACKS_ENABLED = os.environ.get("LAB_FALLBACKS", "1") != "0"


class Refused(RuntimeError):
    """Claude (and any fallback) declined the request."""


def make_client() -> anthropic.Anthropic:
    """Credentials come from the environment (ANTHROPIC_API_KEY or an `ant` profile)."""
    return anthropic.Anthropic()


def text_of(message: Any) -> str:
    """Join the text blocks of a response; ignores thinking and tool blocks."""
    return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


def check_stop(message: Any) -> Any:
    if getattr(message, "stop_reason", None) == "refusal":
        details = getattr(message, "stop_details", None)
        raise Refused(getattr(details, "explanation", None) or "request declined")
    return message


@dataclass
class Usage:
    """Running token totals across calls, for cost reporting."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, message: Any) -> None:
        u = getattr(message, "usage", None)
        if u is None:
            return
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0


USAGE = Usage()


def _params(
    messages: list[dict[str, Any]],
    *,
    model: str | None,
    system: str | list[dict[str, Any]] | None,
    max_tokens: int,
    effort: Effort | None,
    **extra: Any,
) -> dict[str, Any]:
    params: dict[str, Any] = {"model": model or MODEL, "max_tokens": max_tokens, "messages": messages}
    if system:
        params["system"] = system
    if effort:
        params["output_config"] = {**params.get("output_config", {}), "effort": effort}
    for key, value in extra.items():
        if value is None:
            continue
        if key == "output_config":
            params["output_config"] = {**params.get("output_config", {}), **value}
        else:
            params[key] = value
    return params


def send(
    client: Any,
    messages: list[dict[str, Any]] | str,
    *,
    model: str | None = None,
    system: str | list[dict[str, Any]] | None = None,
    max_tokens: int = 16000,
    effort: Effort | None = None,
    **extra: Any,
) -> Any:
    """One Messages API request. Accepts a bare string as a single user turn."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    params = _params(messages, model=model, system=system, max_tokens=max_tokens, effort=effort, **extra)
    if FALLBACKS_ENABLED and params["model"] == MODEL:
        message = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **params)
    else:
        message = client.messages.create(**params)
    USAGE.add(message)
    return check_stop(message)


def ask(client: Any, prompt: str, **kwargs: Any) -> str:
    return text_of(send(client, prompt, **kwargs))


def stream_text(
    client: Any,
    messages: list[dict[str, Any]] | str,
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 64000,
    effort: Effort | None = None,
    on_final: Any = None,
    **extra: Any,
) -> Iterator[str]:
    """Yield text as it is generated (the chat-UI pattern).

    `on_final(message)` receives the accumulated message once the stream ends.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    params = _params(messages, model=model, system=system, max_tokens=max_tokens, effort=effort, **extra)
    with client.messages.stream(**params) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
    USAGE.add(final)
    check_stop(final)
    if on_final is not None:
        on_final(final)


def extract[T: BaseModel](
    client: Any,
    prompt: str,
    schema: type[T],
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 16000,
    effort: Effort | None = None,
) -> T:
    """Structured output: the reply is validated against a Pydantic model.

    The course taught assistant-message prefill plus a stop sequence to get
    clean JSON. Prefill is rejected on current models; structured outputs
    guarantee the shape instead.
    """
    params = _params(
        [{"role": "user", "content": prompt}],
        model=model,
        system=system,
        max_tokens=max_tokens,
        effort=effort,
    )
    response = client.messages.parse(output_format=schema, **params)
    USAGE.add(response)
    check_stop(response)
    return response.parsed_output


def count_tokens(
    client: Any, messages: list[dict[str, Any]], *, system: str | None = None, **extra: Any
) -> int:
    params: dict[str, Any] = {"model": MODEL, "messages": messages, **extra}
    if system:
        params["system"] = system
    return client.messages.count_tokens(**params).input_tokens


@dataclass
class Conversation:
    """Multi-turn chat. The API is stateless, so the history travels every turn.

    Assistant turns are stored as the full content list (not just text) so
    thinking and tool blocks are passed back unchanged.
    """

    client: Any
    system: str | None = None
    effort: Effort | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)

    def say(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        reply = send(self.client, self.messages, system=self.system, effort=self.effort)
        self.messages.append({"role": "assistant", "content": reply.content})
        return text_of(reply)

    def say_streaming(self, text: str) -> Iterator[str]:
        self.messages.append({"role": "user", "content": text})

        def keep(final: Any) -> None:
            self.messages.append({"role": "assistant", "content": final.content})

        yield from stream_text(
            self.client, self.messages, system=self.system, effort=self.effort, on_final=keep
        )
