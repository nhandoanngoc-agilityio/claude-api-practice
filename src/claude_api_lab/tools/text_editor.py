"""The Anthropic-defined text editor tool, with a sandboxed handler.

Claude already knows this tool's schema; you declare only its type and name
and implement the commands. Every `path` is model output, so each one is
resolved and confined to a root directory before it touches the disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOL = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}


class EditorError(ValueError):
    pass


@dataclass
class TextEditor:
    root: Path

    def __post_init__(self) -> None:
        self.root = self.root.resolve()

    def resolve(self, path: str) -> Path:
        candidate = (self.root / path.lstrip("/")).resolve()
        if not candidate.is_relative_to(self.root):
            raise EditorError(f"path escapes the workspace: {path}")
        return candidate

    def __call__(self, command: str, path: str, **kw: Any) -> str:
        target = self.resolve(path)
        match command:
            case "view":
                return self._view(target, kw.get("view_range"))
            case "create":
                if target.exists():
                    # The backup path gets the same confinement check as `path`:
                    # a planted `<name>.bak` symlink must not redirect the write.
                    backup = self.resolve(str(target.relative_to(self.root)) + ".bak")
                    backup.write_text(target.read_text())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(kw["file_text"])
                return f"Created {path}"
            case "str_replace":
                text = self._read(target)
                old, new = kw["old_str"], kw.get("new_str", "")
                count = text.count(old)
                if count != 1:
                    raise EditorError(f"old_str must appear exactly once, found {count}")
                target.write_text(text.replace(old, new, 1))
                return f"Edited {path}"
            case "insert":
                lines = self._read(target).splitlines(keepends=True)
                line = int(kw["insert_line"])
                if not 0 <= line <= len(lines):
                    raise EditorError(f"insert_line out of range 0..{len(lines)}")
                new_text = kw["insert_text"] if kw["insert_text"].endswith("\n") else kw["insert_text"] + "\n"
                lines.insert(line, new_text)
                target.write_text("".join(lines))
                return f"Inserted at line {line} in {path}"
            case _:
                raise EditorError(f"unsupported command {command!r}")

    def _read(self, target: Path) -> str:
        if not target.is_file():
            raise EditorError(f"no such file: {target.relative_to(self.root)}")
        return target.read_text()

    def _view(self, target: Path, view_range: list[int] | None) -> str:
        if target.is_dir():
            return "\n".join(sorted(str(p.relative_to(self.root)) for p in target.iterdir()))
        lines = self._read(target).splitlines()
        start, end = (view_range or [1, len(lines)])[:2]
        end = len(lines) if end == -1 else end
        return "\n".join(f"{i}\t{line}" for i, line in enumerate(lines[start - 1 : end], start))
