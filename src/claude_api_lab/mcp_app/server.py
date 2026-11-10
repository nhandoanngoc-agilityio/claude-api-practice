"""An MCP server exposing a small document store through all three primitives.

  * tools ....... model-controlled actions: read_doc, edit_doc
  * resources ... app-controlled data: docs://documents (list), docs://documents/{doc_id}
  * prompts ..... user-controlled templates: format, summarize

Run it on stdio:  `uv run python -m claude_api_lab.mcp_app.server [docs_dir]`
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

DEFAULT_DOCS = Path(__file__).resolve().parents[3] / "data"


class DocStore:
    """Markdown files in one directory; ids are file stems."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.md"))

    def path(self, doc_id: str) -> Path:
        if doc_id not in self.ids():
            raise ToolError(f"Document {doc_id!r} not found. Available: {', '.join(self.ids())}")
        return self.root / f"{doc_id}.md"

    def read(self, doc_id: str) -> str:
        return self.path(doc_id).read_text()

    def edit(self, doc_id: str, old_str: str, new_str: str) -> None:
        path = self.path(doc_id)
        text = path.read_text()
        if text.count(old_str) != 1:
            raise ToolError(f"old_str must appear exactly once in {doc_id}, found {text.count(old_str)}")
        path.write_text(text.replace(old_str, new_str, 1))


def create_server(docs_dir: Path = DEFAULT_DOCS) -> MCPServer:
    store = DocStore(docs_dir)
    server = MCPServer("doc-desk")

    @server.tool(description="Read the full contents of a document and return it as text.")
    def read_doc(doc_id: str = Field(description="Id of the document to read.")) -> str:
        return store.read(doc_id)

    @server.tool(description="Edit a document by replacing one exact, unique string with new text.")
    def edit_doc(
        doc_id: str = Field(description="Id of the document to edit."),
        old_str: str = Field(description="Text to replace; must appear exactly once, whitespace included."),
        new_str: str = Field(description="Replacement text."),
    ) -> str:
        store.edit(doc_id, old_str, new_str)
        return f"Edited {doc_id}"

    @server.resource("docs://documents", mime_type="application/json")
    def list_docs() -> list[str]:
        return store.ids()

    @server.resource("docs://documents/{doc_id}", mime_type="text/plain")
    def fetch_doc(doc_id: str) -> str:
        return store.read(doc_id)

    @server.prompt(name="format", description="Rewrite a document in clean, well-structured markdown.")
    def format_doc(doc_id: str = Field(description="Id of the document to format.")) -> str:
        return (
            f"Reformat the document <document_id>{doc_id}</document_id> as markdown. "
            "Read it with read_doc, add headings, lists and tables where they help, keep every fact, "
            "then save the result with edit_doc. Reply with a one-line summary of what changed."
        )

    @server.prompt(name="summarize", description="Summarize a document for a busy reader.")
    def summarize_doc(doc_id: str = Field(description="Id of the document to summarize.")) -> str:
        return (
            f"Read <document_id>{doc_id}</document_id> with read_doc and summarize it in at most five bullet "
            "points, leading with the most decision-relevant fact."
        )

    return server


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCS
    create_server(root).run("stdio")
