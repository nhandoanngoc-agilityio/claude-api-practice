import json
from datetime import datetime

import pytest
from fakes import FakeClient, message, text, tool_use

from claude_api_lab.tools import reminders, text_editor, web
from claude_api_lab.tools.loop import ToolBox, run_tool_loop


def make_box(store):
    return ToolBox(
        reminders.SCHEMAS,
        {
            "get_current_datetime": lambda date_format: reminders.get_current_datetime(
                date_format, now=datetime(2026, 11, 10, 9, 30)
            ),
            "add_duration_to_datetime": reminders.add_duration_to_datetime,
            "set_reminder": store.set_reminder,
        },
    )


def test_tool_functions():
    assert reminders.add_duration_to_datetime("2026-11-10", 3, "days").startswith("Friday, November 13, 2026")
    with pytest.raises(reminders.ToolInputError):
        reminders.add_duration_to_datetime("2026-11-10", 1, "fortnights")
    with pytest.raises(reminders.ToolInputError):
        reminders.get_current_datetime("")


def test_schemas_are_strict_and_closed():
    for schema in reminders.SCHEMAS:
        assert schema["strict"] is True
        assert schema["input_schema"]["additionalProperties"] is False
        assert set(schema["input_schema"]["required"]) == set(schema["input_schema"]["properties"])


def test_loop_runs_parallel_tools_and_returns_all_results_in_one_message():
    store = reminders.ReminderStore()
    client = FakeClient(
        message(
            tool_use("get_current_datetime", {"date_format": "%Y-%m-%d"}, "a"),
            tool_use("set_reminder", {"content": "call Hadley", "timestamp": "2026-11-11 09:00"}, "b"),
            stop_reason="tool_use",
        ),
        message(text("Done.")),
    )
    messages = [{"role": "user", "content": "remind me"}]
    final = run_tool_loop(client, make_box(store), messages)
    assert final.content[0].text == "Done."
    results = messages[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[0]["content"] == "2026-11-10"
    assert store.reminders == [{"content": "call Hadley", "timestamp": "2026-11-11 09:00"}]
    assert client.calls[0][1]["tools"] == reminders.SCHEMAS


def test_tool_errors_go_back_to_claude_as_is_error():
    client = FakeClient(
        message(
            tool_use("add_duration_to_datetime", {"datetime_str": "bad", "duration": 1}),
            stop_reason="tool_use",
        ),
        message(text("Sorry, bad date.")),
    )
    messages = [{"role": "user", "content": "x"}]
    run_tool_loop(client, make_box(reminders.ReminderStore()), messages)
    result = messages[2]["content"][0]
    assert result["is_error"] is True
    assert "could not parse" in result["content"]


def test_loop_gives_up_after_max_turns():
    looping = [message(tool_use("get_current_datetime", {"date_format": "%Y"}), stop_reason="tool_use")] * 3
    with pytest.raises(RuntimeError):
        run_tool_loop(FakeClient(*looping), make_box(reminders.ReminderStore()), [], max_turns=3)


def test_batch_tool_runs_each_invocation_and_reports_errors():
    box = make_box(reminders.ReminderStore()).with_batch()
    out = json.loads(
        box.run(
            "batch_tool",
            {
                "invocations": [
                    {"name": "get_current_datetime", "arguments": json.dumps({"date_format": "%H:%M"})},
                    {"name": "nope", "arguments": "{}"},
                ]
            },
        )
    )
    assert out[0] == {"tool": "get_current_datetime", "output": "09:30"}
    assert "unknown tool" in out[1]["error"]


def test_text_editor_commands_and_sandbox(tmp_path):
    ed = text_editor.TextEditor(tmp_path)
    ed("create", "notes.md", file_text="one\ntwo\n")
    ed("str_replace", "notes.md", old_str="two", new_str="TWO")
    ed("insert", "notes.md", insert_line=1, insert_text="middle")
    assert (tmp_path / "notes.md").read_text() == "one\nmiddle\nTWO\n"
    assert ed("view", "notes.md", view_range=[2, 2]) == "2\tmiddle"
    ed("create", "notes.md", file_text="new")
    assert (tmp_path / "notes.md.bak").read_text() == "one\nmiddle\nTWO\n"
    with pytest.raises(text_editor.EditorError):
        ed("view", "../outside.txt")
    with pytest.raises(text_editor.EditorError):
        ed("str_replace", "notes.md", old_str="missing", new_str="x")


def test_text_editor_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("s")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(text_editor.EditorError):
        text_editor.TextEditor(root)("view", "link/secret")


def test_text_editor_backup_cannot_escape_through_symlink(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched")
    root = tmp_path / "root"
    root.mkdir()
    (root / "notes.md").write_text("old")
    (root / "notes.md.bak").symlink_to(outside)
    with pytest.raises(text_editor.EditorError):
        text_editor.TextEditor(root)("create", "notes.md", file_text="new")
    assert outside.read_text() == "untouched"
    assert (root / "notes.md").read_text() == "old"


def test_web_search_sources_handles_results_and_errors():
    from types import SimpleNamespace as NS

    ok = NS(type="web_search_tool_result", content=[NS(title="T", url="https://u")])
    err = NS(type="web_search_tool_result", content=NS(error_code="max_uses_exceeded"))
    assert web.sources(message(ok, err, text("a"))) == [
        {"title": "T", "url": "https://u"},
        {"error": "max_uses_exceeded"},
    ]


def test_web_search_resumes_paused_turn():
    client = FakeClient(message(text("..."), stop_reason="pause_turn"), message(text("answer")))
    answer, _ = web.search(client, "q", ["example.com"])
    assert answer == "answer"
    assert client.last()["tools"][0]["allowed_domains"] == ["example.com"]
    assert len(client.last()["messages"]) == 2
