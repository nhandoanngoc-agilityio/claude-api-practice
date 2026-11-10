from types import SimpleNamespace as NS

import pytest
from fakes import FakeClient, message, text, thinking

from claude_api_lab.features import batches, caching, code_exec, media
from claude_api_lab.features import thinking as think_mod


def test_think_uses_adaptive_summarized_and_splits_reasoning():
    client = FakeClient(message(thinking("step 1"), text("42")))
    out = think_mod.think(client, "6*7?", effort="max")
    assert (out.reasoning, out.answer) == ("step 1", "42")
    params = client.last()
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["output_config"]["effort"] == "max"
    assert "budget_tokens" not in str(params)


def test_image_and_pdf_blocks(tmp_path):
    png = tmp_path / "chart.png"
    png.write_bytes(b"\x89PNG")
    block = media.image_block(png)
    assert block["source"]["media_type"] == "image/png" and block["source"]["data"] == "iVBORw=="
    with pytest.raises(ValueError):
        media.image_block(tmp_path / "x.bmp")
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF")
    assert media.pdf_block(pdf, citations=True)["citations"] == {"enabled": True}


def test_image_goes_before_question(tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"x")
    client = FakeClient(message(text("a cat")))
    media.describe_image(client, png, "What?")
    content = client.last()["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "text"]


def test_citations_are_collected_and_rendered():
    cite = NS(
        type="char_location",
        cited_text="revenue was $4.2M",
        document_index=0,
        document_title="report",
        start_char_index=10,
        end_char_index=27,
    )
    page = NS(
        type="page_location",
        cited_text="p",
        document_index=0,
        document_title="deck",
        start_page_number=3,
        end_page_number=4,
    )
    client = FakeClient(message(text("Revenue: $4.2M", [cite]), text(" and more", [page]), text(".")))
    spans = media.ask_with_citations(client, [media.text_document_block("...", "report")], "Revenue?")
    rendered = media.render_citations(spans)
    assert rendered.startswith("Revenue: $4.2M[1] and more[2].")
    assert '[1] report chars 10-27: "revenue was $4.2M"' in rendered
    assert "[2] deck p.3:" in rendered


def test_cached_chat_marks_document_breakpoint_and_reports_cache_hits():
    client = FakeClient(
        message(text("a1"), cache_creation_input_tokens=5000),
        message(text("a2"), cache_read_input_tokens=5000),
    )
    chat = caching.CachedDocumentChat(client, "BIG DOC")
    _, first = chat.ask("q1")
    _, second = chat.ask("q2")
    assert first["cache_write"] == 5000 and second["cache_read"] == 5000
    params = client.last()
    assert params["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["system"] == caching.CachedDocumentChat(client, "BIG DOC").system()  # stable prefix


class FakeFiles:
    def __init__(self):
        self.downloaded = []

    def upload(self, file):
        return NS(id="file_in")

    def retrieve_metadata(self, file_id):
        return NS(filename="../../evil/chart.png")

    def download(self, file_id):
        return NS(write_to_file=lambda p: (p.write_bytes(b"img"), self.downloaded.append(p)))


def test_code_execution_uploads_file_and_saves_outputs_safely(tmp_path):
    result = NS(
        type="bash_code_execution_result",
        stdout="mean=5",
        content=[NS(type="bash_code_execution_output", file_id="f1")],
    )
    response = message(NS(type="bash_code_execution_tool_result", content=result), text("Mean is 5."))
    client = FakeClient(response)
    client.files = FakeFiles()
    data = tmp_path / "d.csv"
    data.write_text("x\n5\n")
    out = code_exec.analyse_file(client, data, "mean?", tmp_path / "out")
    assert out.summary == "Mean is 5." and out.stdout == ["mean=5"]
    assert out.saved == [tmp_path / "out" / "chart.png"]  # traversal stripped
    sent = client.last()
    assert sent["messages"][0]["content"][1] == {"type": "container_upload", "file_id": "file_in"}
    assert sent["tools"] == [code_exec.TOOL]


def test_batches_keyed_by_custom_id():
    created = {}

    class Batches:
        def create(self, requests):
            created["requests"] = requests
            return NS(id="b1")

        def retrieve(self, batch_id):
            return NS(processing_status="ended")

        def results(self, batch_id):
            return [
                NS(custom_id="q1", result=NS(type="errored")),
                NS(custom_id="q0", result=NS(type="succeeded", message=message(text("A0")))),
            ]

    client = FakeClient()
    client.messages.batches = Batches()
    assert batches.submit(client, {"q0": "a", "q1": "b"}) == "b1"
    assert [r["custom_id"] for r in created["requests"]] == ["q0", "q1"]
    batches.wait(client, "b1", poll_seconds=0)
    assert batches.results(client, "b1") == {"q1": "<errored>", "q0": "A0"}
