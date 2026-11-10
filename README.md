# claude-api-lab

Every technique from Anthropic Academy's **Building with the Claude API** course, applied to one small
project: a *document desk* that reads, searches, edits and reports on a company's documents
(`data/` holds a fictional robotics company's Q3 report and onboarding guide).

Each chapter is one module and one `lab` command, and every module has offline tests that use a scripted
fake client. No API key is needed to run the tests.

## Setup

```bash
uv sync
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```bash
uv run pytest
```

Optional: `uv sync --extra voyage` plus `VOYAGE_API_KEY` switches RAG to Voyage embeddings. Without
them, RAG uses a local hashing embedder.

Models: `LAB_MODEL` (default `claude-opus-5-5`) and `LAB_FAST_MODEL` (default `claude-haiku-5-5`, used
by graders, the router and contextual retrieval). `LAB_FALLBACKS=0` turns off server-side refusal
fallbacks.

## Course map

| Course topic | Where | Try it |
|---|---|---|
| **1. Accessing the API** | | |
| Making a request, multi-turn conversations, system prompts | `llm.py` (`send`, `Conversation`) | `lab chat` |
| Streaming | `llm.stream_text`, `Conversation.say_streaming` | `lab chat` |
| Controlling output ("temperature") | `effort=` on every call (see drift notes) | `lab chat --effort high` |
| Structured data (prefill + stop sequences) | `llm.extract` (structured outputs), `stop_sequences=` | used by evals, router, reranker |
| Token counting, usage and cost | `llm.count_tokens`, `llm.USAGE` | printed after each command |
| **2. Prompt evaluation** | | |
| Generating a test dataset | `evals.generate_dataset` | `lab eval` |
| Running the eval | `evals.run_eval` (concurrent) | `lab eval --verbose` |
| Code-based grading | `evals.grade_by_code` (JSON / Python / regex syntax) | |
| Model-based grading | `evals.grade_by_model` (strengths, weaknesses, reasoning, score) | |
| Comparing prompt versions | baseline vs engineered prompt averages | `lab eval` |
| **3. Prompt engineering** | | |
| Clear and direct, being specific, XML tags, examples | `prompting.PromptBuilder` | `lab ask "..." --engineered` |
| **4. Tool use** | | |
| Tool functions and schemas (strict) | `tools/reminders.py` | |
| Handling tool_use / tool_result, multi-turn loop, multiple tools | `tools/loop.run_tool_loop` | `lab remind "remind me in 3 days to call Hadley"` |
| Tool errors (`is_error`) | `ToolBox.result_block` | |
| Batch tool | `tools/loop.BATCH_TOOL` | `lab remind ...` |
| SDK tool runner | `tools/runner.py` | `lab remind ... --runner` |
| Text editor tool | `tools/text_editor.py` (sandboxed paths) | `lab agent ...` |
| Web search tool | `tools/web.py` | `lab search "..." --domain example.com` |
| **5. RAG** | | |
| Chunking by size, sentence, section | `rag/chunking.py` | |
| Embeddings | `rag/embeddings.py` (Voyage or local) | |
| Vector search, BM25 lexical search | `rag/search.py` | |
| Hybrid search with reciprocal rank fusion | `rag/search.HybridRetriever` | `lab rag "..." --no-rerank` |
| Reranking | `rag/pipeline.rerank` | `lab rag "..."` |
| Contextual retrieval (with prompt caching) | `rag/pipeline.situate_chunk` | `lab rag "..."` (`--no-context` to compare) |
| **6. Features of Claude** | | |
| Extended thinking | `features/thinking.py` | `lab think "..."` |
| Image support | `features/media.describe_image` | `lab see photo.png` |
| PDF support | `features/media.ask_pdf` | `lab see report.pdf "..."` |
| Citations | `features/media.ask_with_citations` | `lab cite "What was Q3 revenue?"` |
| Prompt caching | `features/caching.py` | `lab cache "q1" "q2" "q3"` |
| Code execution and the Files API | `features/code_exec.py` | `lab code data.csv "chart the trend"` |
| Message Batches | `features/batches.py` | `lab batch "q1" "q2"` |
| **7. Model Context Protocol** | | |
| MCP server: tools, resources, prompts | `mcp_app/server.py` | `uv run python -m claude_api_lab.mcp_app.server` |
| MCP client: tools, `@mention` resources, `/command` prompts | `mcp_app/chat.py` | `lab mcp` |
| **8. Agents and workflows** | | |
| Chaining | `workflows/patterns.chain_write` | `lab workflow chain` |
| Routing | `workflows/patterns.route` | `lab workflow route --input "..."` |
| Parallelization | `workflows/patterns.parallel_review` | `lab workflow parallel` |
| Evaluator-optimizer | `workflows/patterns.evaluate_optimize` | `lab workflow optimize` |
| Agents, tools, environment inspection | `workflows/agent.py` | `lab agent "add a release checklist to notes.md"` |

Not built here: the course's lessons on Claude Code and on computer use. Those cover products rather
than API code.

## API drift since the course was recorded

The course predates some API changes. This lab uses the current form of each:

| Course taught | Current API (used here) |
|---|---|
| `temperature` to trade creativity for consistency | Sampling parameters are rejected on current models. `output_config.effort` (`low`…`max`) is the dial. |
| Prefill the assistant turn plus a stop sequence to get JSON | Prefill is rejected. Structured outputs (`messages.parse(output_format=Model)`) guarantee the schema. |
| `thinking: {type: "enabled", budget_tokens: N}` | `thinking: {type: "adaptive", display: "summarized"}`, with depth set by effort. |
| `tool_choice: {type: "tool"}` to force a call | Forced tool choice is rejected on current models. Use `strict: true` schemas and steer from the prompt. |
| `client.beta.files` with a beta header | `client.files` (generally available). |
| `FastMCP` from `mcp.server.fastmcp` | MCP SDK 2.x: `MCPServer` from `mcp.server.mcpserver`, `mcp.Client`. |

All default-model calls also send `fallbacks: "default"`. If the model declines a request, the API
re-runs it on a fallback model inside the same call. A refusal still raises `llm.Refused`.

## Layout

```
src/claude_api_lab/
  llm.py          ch1  requests, streaming, structured output, usage
  evals.py        ch2  dataset generation, code + model grading
  prompting.py    ch3  PromptBuilder
  tools/          ch4  schemas, manual loop, batch tool, tool runner, text editor, web search
  rag/            ch5  chunking, embeddings, vector/BM25/hybrid, rerank, contextual retrieval
  features/       ch6  thinking, images/PDF/citations, caching, code execution, batches
  mcp_app/        ch7  MCP server and client
  workflows/      ch8  chaining, routing, parallelization, evaluator-optimizer, agent
  cli.py               `lab` entry point
tests/                 offline tests with a scripted fake client
```
