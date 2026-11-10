"""Code execution with the Files API.

Upload a file once, mount it into the code execution container with a
`container_upload` block, let Claude analyse it with code, then download any
files it produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_api_lab import llm

TOOL = {"type": "code_execution_20260120", "name": "code_execution"}


@dataclass
class Analysis:
    summary: str
    stdout: list[str] = field(default_factory=list)
    saved: list[Path] = field(default_factory=list)


def upload(client: Any, path: Path) -> str:
    return client.files.upload(file=path).id


def analyse_file(client: Any, path: Path, request: str, out_dir: Path) -> Analysis:
    file_id = upload(client, path)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": request}, {"type": "container_upload", "file_id": file_id}],
        }
    ]
    response = llm.send(client, messages, tools=[TOOL])
    while response.stop_reason == "pause_turn":
        messages.append({"role": "assistant", "content": response.content})
        response = llm.send(client, messages, tools=[TOOL], container=response.container.id)
    return collect(client, response, out_dir)


def collect(client: Any, response: Any, out_dir: Path) -> Analysis:
    analysis = Analysis(summary=llm.text_of(response))
    for block in response.content:
        if block.type != "bash_code_execution_tool_result":
            continue
        result = block.content
        if result.type != "bash_code_execution_result":
            continue
        if result.stdout:
            analysis.stdout.append(result.stdout)
        for ref in result.content or []:
            if ref.type == "bash_code_execution_output":
                analysis.saved.append(download(client, ref.file_id, out_dir))
    return analysis


def download(client: Any, file_id: str, out_dir: Path) -> Path:
    meta = client.files.retrieve_metadata(file_id)
    name = Path(meta.filename).name  # basename only: the name is model-chosen
    if name in ("", ".", ".."):
        raise ValueError(f"refusing unsafe filename {meta.filename!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    client.files.download(file_id).write_to_file(target)
    return target
