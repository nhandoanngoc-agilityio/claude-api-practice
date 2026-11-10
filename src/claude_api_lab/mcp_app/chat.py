"""An MCP client that chats with Claude using the server's primitives.

* tools ....... MCP tools are listed, converted to Claude tool schemas,
                and every tool_use is routed back to the MCP server.
* resources ... `@doc_id` in a message inlines that document's resource,
                so Claude needn't spend a tool call fetching it.
* prompts ..... `/format doc_id` or `/summarize doc_id` expands the server's
                prompt template into the user turn.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mcp import Client
from mcp.types import TextContent, TextResourceContents

from claude_api_lab import llm

MENTION = re.compile(r"@([\w-]+)")


class McpChat:
    def __init__(self, client: Any, mcp: Client) -> None:
        self.client = client
        self.mcp = mcp
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []

    async def load_tools(self) -> None:
        listed = await self.mcp.list_tools()
        self.tools = [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in listed.tools
        ]

    async def doc_ids(self) -> list[str]:
        return json.loads(await self._read("docs://documents"))

    async def _read(self, uri: str) -> str:
        result = await self.mcp.read_resource(uri)
        return "".join(c.text for c in result.contents if isinstance(c, TextResourceContents))

    async def expand(self, text: str) -> str:
        """Apply /commands (prompts) and @mentions (resources) to a user message."""
        if text.startswith("/"):
            name, _, arg = text[1:].partition(" ")
            prompt = await self.mcp.get_prompt(name, {"doc_id": arg.strip()})
            text = "\n".join(m.content.text for m in prompt.messages if isinstance(m.content, TextContent))
        known = set(await self.doc_ids())
        attached = []
        for doc_id in dict.fromkeys(MENTION.findall(text)):
            if doc_id in known:
                body = await self._read(f"docs://documents/{doc_id}")
                attached.append(f'<document id="{doc_id}">\n{body}\n</document>')
        if attached:
            text = "The user referenced these documents:\n" + "\n".join(attached) + "\n\n" + text
        return text

    async def call_tool(self, block: Any) -> dict[str, Any]:
        result = await self.mcp.call_tool(block.name, dict(block.input))
        text = "\n".join(c.text for c in result.content if isinstance(c, TextContent))
        out = {"type": "tool_result", "tool_use_id": block.id, "content": text}
        if result.is_error:
            out["is_error"] = True
        return out

    async def send(self, user_text: str, max_turns: int = 10) -> str:
        if not self.tools:
            await self.load_tools()
        self.messages.append({"role": "user", "content": await self.expand(user_text)})
        for _ in range(max_turns):
            response = llm.send(self.client, self.messages, tools=self.tools)
            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                return llm.text_of(response)
            results = [await self.call_tool(b) for b in response.content if b.type == "tool_use"]
            self.messages.append({"role": "user", "content": results})
        raise RuntimeError("MCP chat did not finish within max_turns")
