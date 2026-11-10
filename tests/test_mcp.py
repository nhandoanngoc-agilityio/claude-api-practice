import asyncio
import shutil
from pathlib import Path

import pytest
from fakes import FakeClient, message, text, tool_use
from mcp import Client

from claude_api_lab.mcp_app.chat import McpChat
from claude_api_lab.mcp_app.server import create_server

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def docs(tmp_path):
    for p in DATA.glob("*.md"):
        shutil.copy(p, tmp_path)
    return tmp_path


def run(coro):
    return asyncio.run(coro)


def test_server_exposes_tools_resources_and_prompts(docs):
    async def go():
        async with Client(create_server(docs)) as mcp:
            tools = {t.name for t in (await mcp.list_tools()).tools}
            prompts = {p.name for p in (await mcp.list_prompts()).prompts}
            chat = McpChat(FakeClient(), mcp)
            ids = await chat.doc_ids()
            return tools, prompts, ids

    tools, prompts, ids = run(go())
    assert tools == {"read_doc", "edit_doc"}
    assert prompts == {"format", "summarize"}
    assert ids == ["northwind_q3_report", "onboarding_guide"]


def test_mentions_inline_resources_and_commands_expand_prompts(docs):
    async def go():
        async with Client(create_server(docs)) as mcp:
            chat = McpChat(FakeClient(), mcp)
            mentioned = await chat.expand("What does @onboarding_guide say? @nope")
            command = await chat.expand("/summarize northwind_q3_report")
            return mentioned, command

    mentioned, command = run(go())
    assert '<document id="onboarding_guide">' in mentioned and "Test Floor Safety" in mentioned
    assert "nope" not in mentioned.split("\n\n")[0]
    assert "<document_id>northwind_q3_report</document_id>" in command


def test_chat_routes_tool_calls_to_mcp_server(docs):
    client = FakeClient(
        message(
            tool_use(
                "edit_doc",
                {"doc_id": "onboarding_guide", "old_str": "every second Tuesday", "new_str": "every Tuesday"},
            ),
            stop_reason="tool_use",
        ),
        message(tool_use("read_doc", {"doc_id": "missing"}, "t2"), stop_reason="tool_use"),
        message(text("Updated the release cadence.")),
    )

    async def go():
        async with Client(create_server(docs)) as mcp:
            chat = McpChat(client, mcp)
            reply = await chat.send("Releases are weekly now; fix the guide.")
            return reply, chat

    reply, chat = run(go())
    assert reply == "Updated the release cadence."
    assert "every Tuesday" in (docs / "onboarding_guide.md").read_text()
    assert chat.messages[2]["content"][0]["content"] == "Edited onboarding_guide"
    error = chat.messages[4]["content"][0]
    assert error["is_error"] is True and "not found" in error["content"]
    assert {t["name"] for t in client.calls[0][1]["tools"]} == {"read_doc", "edit_doc"}
