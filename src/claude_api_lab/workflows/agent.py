"""An agent: Claude chooses the steps, with a few general tools.

Course points applied:
  * give the agent abstract, composable tools rather than one tool per task
    (a text editor plus the reminder tools, not "write_meeting_notes")
  * environment inspection: after each edit the agent views the file again,
    so it acts on what is actually on disk, not on what it meant to write
  * bounded autonomy: a turn limit and a sandboxed workspace
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claude_api_lab import llm
from claude_api_lab.tools import reminders, text_editor
from claude_api_lab.tools.loop import ToolBox, run_tool_loop

SYSTEM = """You are a document desk agent working in a sandboxed workspace.
Use the text editor to read and change files. After every change, view the file again
to confirm the change landed before you report it. Use the date tools for any date
arithmetic and set_reminder for follow-ups. Stop when the task is done and summarize
what changed, naming each file."""


def build_toolbox(workspace: Path, store: reminders.ReminderStore) -> ToolBox:
    editor = text_editor.TextEditor(workspace)
    return ToolBox(
        schemas=[text_editor.TOOL, *reminders.SCHEMAS],
        handlers={
            text_editor.TOOL["name"]: editor,
            "get_current_datetime": reminders.get_current_datetime,
            "add_duration_to_datetime": reminders.add_duration_to_datetime,
            "set_reminder": store.set_reminder,
        },
    )


def run_agent(client: Any, task: str, workspace: Path, *, max_turns: int = 20) -> tuple[str, ToolBox]:
    workspace.mkdir(parents=True, exist_ok=True)
    store = reminders.ReminderStore()
    toolbox = build_toolbox(workspace, store)
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    final = run_tool_loop(client, toolbox, messages, system=SYSTEM, max_turns=max_turns, effort="high")
    return llm.text_of(final), toolbox
