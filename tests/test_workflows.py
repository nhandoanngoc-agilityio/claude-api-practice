from fakes import FakeClient, message, text, tool_use

from claude_api_lab import cli
from claude_api_lab.workflows import agent, patterns


def test_chain_skips_revision_when_critique_is_clean():
    client = FakeClient(message(text("draft")), message(text("NONE")))
    out = patterns.chain_write(client, "Q3", "src")
    assert out["final"] == "draft" and len(client.calls) == 2


def test_chain_revises_with_critique():
    client = FakeClient(message(text("draft")), message(text("uses em dash")), message(text("fixed")))
    out = patterns.chain_write(client, "Q3", "src")
    assert out["final"] == "fixed"
    assert "<problems>\nuses em dash" in client.last()["messages"][0]["content"]


def test_route_uses_specialist_system_prompt():
    client = FakeClient(patterns.Route(category="data_question", reason="numbers"), message(text("61%")))
    decision, answer = patterns.route(client, "Gross margin?", "doc")
    assert decision.category == "data_question" and answer == "61%"
    assert client.last()["system"] == patterns.SPECIALISTS["data_question"]


def test_parallel_review_runs_each_lens_then_aggregates():
    replies = [message(text(f"r{i}")) for i in range(len(patterns.LENSES))] + [message(text("verdict"))]
    out = patterns.parallel_review(FakeClient(*replies), "doc", max_workers=1)
    assert set(out) == {*patterns.LENSES, "overall"} and out["overall"] == "verdict"


def test_evaluator_optimizer_feeds_back_until_pass():
    client = FakeClient(
        message(text("v1")),
        patterns.Verdict(passed=False, score=4, feedback="add a risk"),
        message(text("v2")),
        patterns.Verdict(passed=True, score=9, feedback="good"),
    )
    out = patterns.evaluate_optimize(client, "write", ["has a risk"])
    assert out.output == "v2" and len(out.attempts) == 2
    assert "<feedback>\nadd a risk" in client.calls[2][1]["messages"][0]["content"]


def test_evaluator_optimizer_stops_at_max_rounds():
    fail = patterns.Verdict(passed=False, score=3, feedback="no")
    client = FakeClient(message(text("a")), fail, message(text("b")), fail)
    assert len(patterns.evaluate_optimize(client, "t", ["c"], max_rounds=2).attempts) == 2


def test_agent_edits_then_inspects_its_workspace(tmp_path):
    client = FakeClient(
        message(
            tool_use(
                "str_replace_based_edit_tool",
                {"command": "create", "path": "todo.md", "file_text": "- ship\n"},
            ),
            stop_reason="tool_use",
        ),
        message(
            tool_use("str_replace_based_edit_tool", {"command": "view", "path": "todo.md"}, "t2"),
            stop_reason="tool_use",
        ),
        message(text("Created todo.md.")),
    )
    summary, box = agent.run_agent(client, "make a todo", tmp_path / "ws")
    assert summary == "Created todo.md."
    assert (tmp_path / "ws" / "todo.md").read_text() == "- ship\n"
    assert [c[1]["command"] for c in box.calls] == ["create", "view"]
    assert client.calls[0][1]["system"] == agent.SYSTEM


def test_cli_parser_knows_every_chapter_command():
    parser = cli.build_parser()
    for argv in (
        ["chat"],
        ["ask", "q"],
        ["eval"],
        ["remind", "x"],
        ["search", "q"],
        ["rag", "q"],
        ["think", "p"],
        ["see", "a.png"],
        ["cite", "q"],
        ["cache", "q"],
        ["code", "d.csv", "r"],
        ["batch", "q"],
        ["mcp"],
        ["workflow", "route"],
        ["agent", "t"],
    ):
        assert parser.parse_args(argv).fn
