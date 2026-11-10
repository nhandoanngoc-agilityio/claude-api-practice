"""The web search server tool.

Server tools run on Anthropic's side: declare them and read the result
blocks. Errors arrive as a result block (HTTP 200), not as an exception.
"""

from __future__ import annotations

from typing import Any

from claude_api_lab import llm


def web_search_tool(max_uses: int = 5, allowed_domains: list[str] | None = None) -> dict[str, Any]:
    tool: dict[str, Any] = {"type": "web_search_20260209", "name": "web_search", "max_uses": max_uses}
    if allowed_domains:
        tool["allowed_domains"] = allowed_domains
    return tool


def sources(message: Any) -> list[dict[str, str]]:
    """URLs the search returned, plus any error codes."""
    found: list[dict[str, str]] = []
    for block in message.content:
        if block.type != "web_search_tool_result":
            continue
        if isinstance(block.content, list):
            found += [{"title": r.title, "url": r.url} for r in block.content]
        else:
            found.append({"error": block.content.error_code})
    return found


def search(
    client: Any, question: str, allowed_domains: list[str] | None = None
) -> tuple[str, list[dict[str, str]]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    tools = [web_search_tool(allowed_domains=allowed_domains)]
    response = llm.send(client, messages, tools=tools)
    while response.stop_reason == "pause_turn":
        messages.append({"role": "assistant", "content": response.content})
        response = llm.send(client, messages, tools=tools)
    return llm.text_of(response), sources(response)
