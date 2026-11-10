"""`lab <command>`: one command per course chapter, all against the sample docs in data/."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from claude_api_lab import evals, llm, prompting

DATA = Path(__file__).resolve().parents[2] / "data"
REPORT = DATA / "northwind_q3_report.md"


def _client():
    return llm.make_client()


def cmd_chat(args: argparse.Namespace) -> None:
    convo = llm.Conversation(_client(), system=args.system, effort=args.effort)
    print("Chat (empty line to quit).")
    while line := input("> ").strip():
        for chunk in convo.say_streaming(line):
            print(chunk, end="", flush=True)
        print()


def cmd_ask(args: argparse.Namespace) -> None:
    doc = REPORT.read_text()
    build = prompting.doc_qa_prompt_v2 if args.engineered else prompting.doc_qa_prompt_v1
    for chunk in llm.stream_text(_client(), build(args.question, doc)):
        print(chunk, end="", flush=True)
    print()


def cmd_eval(args: argparse.Namespace) -> None:
    client = _client()
    path = Path(args.dataset)
    if not path.exists():
        evals.save_dataset(evals.generate_dataset(client, args.topic, args.n), path)
        print(f"Generated dataset -> {path}")
    dataset = evals.load_dataset(path)
    for name, fn in (("baseline", evals.baseline_prompt), ("engineered", evals.engineered_prompt)):
        report = evals.run_eval(client, dataset, fn)
        print(f"{name}: average {report.average:.2f}")
        if args.verbose:
            print(json.dumps(report.to_dict(), indent=2))


def cmd_remind(args: argparse.Namespace) -> None:
    from claude_api_lab.tools import reminders, runner
    from claude_api_lab.tools.loop import ToolBox, run_tool_loop

    client = _client()
    store = reminders.ReminderStore()
    if args.runner:
        print(runner.remind(client, args.request, store))
    else:
        box = ToolBox(
            reminders.SCHEMAS,
            {
                "get_current_datetime": reminders.get_current_datetime,
                "add_duration_to_datetime": reminders.add_duration_to_datetime,
                "set_reminder": store.set_reminder,
            },
        ).with_batch()
        final = run_tool_loop(client, box, [{"role": "user", "content": args.request}])
        print(llm.text_of(final))
        print(f"tool calls: {[name for name, _ in box.calls]}")
    print(f"reminders: {store.reminders}")


def cmd_search(args: argparse.Namespace) -> None:
    from claude_api_lab.tools import web

    answer, found = web.search(_client(), args.question, args.domain or None)
    print(answer)
    for src in found:
        print(" -", src)


def cmd_rag(args: argparse.Namespace) -> None:
    from claude_api_lab.rag.pipeline import RagPipeline

    rag = RagPipeline(_client(), contextual=not args.no_context, use_reranker=not args.no_rerank)
    for path in sorted(DATA.glob("*.md")):
        rag.add_document(path.read_text(), path.name)
    for chunk, score in rag.retrieve(args.question):
        print(f"[{score:.3f}] {chunk.source}#{chunk.id}: {chunk.text[:70]!r}")
    print("\n" + rag.answer(args.question))


def cmd_think(args: argparse.Namespace) -> None:
    from claude_api_lab.features import thinking

    thought = thinking.think(_client(), args.problem, args.effort)
    print("--- reasoning summary ---\n" + thought.reasoning + "\n--- answer ---\n" + thought.answer)


def cmd_see(args: argparse.Namespace) -> None:
    from claude_api_lab.features import media

    path = Path(args.path)
    if path.suffix.lower() == ".pdf":
        print(media.ask_pdf(_client(), path, args.question))
    else:
        print(media.describe_image(_client(), path, args.question))


def cmd_cite(args: argparse.Namespace) -> None:
    from claude_api_lab.features import media

    path = Path(args.path) if args.path else REPORT
    if path.suffix.lower() == ".pdf":
        doc = media.pdf_block(path, citations=True, title=path.name)
    else:
        doc = media.text_document_block(path.read_text(), path.name)
    print(media.render_citations(media.ask_with_citations(_client(), [doc], args.question)))


def cmd_cache(args: argparse.Namespace) -> None:
    from claude_api_lab.features.caching import CachedDocumentChat

    # Repeat the report so the prefix clears the minimum cacheable size.
    chat = CachedDocumentChat(_client(), REPORT.read_text() * 8)
    for q in args.questions:
        answer, stats = chat.ask(q)
        print(f"Q: {q}\nA: {answer}\n   tokens {stats}\n")


def cmd_code(args: argparse.Namespace) -> None:
    from claude_api_lab.features import code_exec

    result = code_exec.analyse_file(_client(), Path(args.path), args.request, Path(args.out))
    print(result.summary)
    for p in result.saved:
        print("saved", p)


def cmd_batch(args: argparse.Namespace) -> None:
    from claude_api_lab.features import batches

    client = _client()
    prompts = {f"q{i}": q for i, q in enumerate(args.questions)}
    batch_id = batches.submit(client, prompts)
    print(f"batch {batch_id} submitted; waiting...")
    batches.wait(client, batch_id, poll_seconds=args.poll)
    for custom_id, text in sorted(batches.results(client, batch_id).items()):
        print(f"{custom_id}: {text}")


def cmd_mcp(args: argparse.Namespace) -> None:
    from mcp import Client

    from claude_api_lab.mcp_app.chat import McpChat
    from claude_api_lab.mcp_app.server import create_server

    async def main() -> None:
        async with Client(create_server(Path(args.docs))) as mcp:
            chat = McpChat(_client(), mcp)
            print(f"Docs: {', '.join(await chat.doc_ids())}. Use @doc_id, /format doc_id, /summarize doc_id.")
            while line := input("> ").strip():
                print(await chat.send(line))

    asyncio.run(main())


def cmd_workflow(args: argparse.Namespace) -> None:
    from claude_api_lab.workflows import patterns

    client, doc = _client(), REPORT.read_text()
    match args.pattern:
        case "chain":
            print(json.dumps(patterns.chain_write(client, args.input or "Q3 results", doc), indent=2))
        case "route":
            decision, answer = patterns.route(client, args.input or "What was gross margin?", doc)
            print(f"[{decision.category}] {decision.reason}\n\n{answer}")
        case "parallel":
            print(json.dumps(patterns.parallel_review(client, doc), indent=2))
        case "optimize":
            out = patterns.evaluate_optimize(
                client,
                args.input or "Write a two-sentence investor update from this report:\n" + doc,
                ["Quotes revenue and growth exactly", "Mentions one risk", "Two sentences, no hype"],
            )
            for i, (_, v) in enumerate(out.attempts, 1):
                print(f"attempt {i}: score {v.score} passed={v.passed} - {v.feedback}")
            print("\n" + out.output)


def cmd_agent(args: argparse.Namespace) -> None:
    from claude_api_lab.workflows.agent import run_agent

    summary, box = run_agent(_client(), args.task, Path(args.workspace))
    print(summary)
    print(f"tool calls: {[name for name, _ in box.calls]}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    effort = {"choices": ["low", "medium", "high", "xhigh", "max"]}

    s = sub.add_parser("chat", help="ch1: streaming multi-turn chat")
    s.add_argument("--system", default="You are a concise assistant.")
    s.add_argument("--effort", default="low", **effort)
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("ask", help="ch3: question the report with a naive or engineered prompt")
    s.add_argument("question")
    s.add_argument("--engineered", action="store_true")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("eval", help="ch2: generate a dataset and score two prompt versions")
    s.add_argument("--topic", default="data-processing tasks for a robotics company's reporting scripts")
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--dataset", default="outputs/dataset.json")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("remind", help="ch4: tools - manual loop + batch tool, or the SDK tool runner")
    s.add_argument("request")
    s.add_argument("--runner", action="store_true")
    s.set_defaults(fn=cmd_remind)

    s = sub.add_parser("search", help="ch4: web search server tool")
    s.add_argument("question")
    s.add_argument("--domain", action="append")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("rag", help="ch5: contextual hybrid retrieval + rerank over data/")
    s.add_argument("question")
    s.add_argument("--no-context", action="store_true")
    s.add_argument("--no-rerank", action="store_true")
    s.set_defaults(fn=cmd_rag)

    s = sub.add_parser("think", help="ch6: adaptive extended thinking")
    s.add_argument("problem")
    s.add_argument("--effort", default="high", **effort)
    s.set_defaults(fn=cmd_think)

    s = sub.add_parser("see", help="ch6: image or PDF input")
    s.add_argument("path")
    s.add_argument("question", nargs="?", default="Describe this in detail.")
    s.set_defaults(fn=cmd_see)

    s = sub.add_parser("cite", help="ch6: citations over a text or PDF document")
    s.add_argument("question")
    s.add_argument("--path")
    s.set_defaults(fn=cmd_cite)

    s = sub.add_parser("cache", help="ch6: prompt caching across several questions")
    s.add_argument("questions", nargs="+")
    s.set_defaults(fn=cmd_cache)

    s = sub.add_parser("code", help="ch6: code execution + Files API")
    s.add_argument("path")
    s.add_argument("request")
    s.add_argument("--out", default="outputs")
    s.set_defaults(fn=cmd_code)

    s = sub.add_parser("batch", help="ch6: Message Batches")
    s.add_argument("questions", nargs="+")
    s.add_argument("--poll", type=float, default=30)
    s.set_defaults(fn=cmd_batch)

    s = sub.add_parser("mcp", help="ch7: chat through the MCP document server")
    s.add_argument("--docs", default=str(DATA))
    s.set_defaults(fn=cmd_mcp)

    s = sub.add_parser("workflow", help="ch8: chain | route | parallel | optimize")
    s.add_argument("pattern", choices=["chain", "route", "parallel", "optimize"])
    s.add_argument("--input")
    s.set_defaults(fn=cmd_workflow)

    s = sub.add_parser("agent", help="ch8: tool-using agent in a sandboxed workspace")
    s.add_argument("task")
    s.add_argument("--workspace", default="outputs/workspace")
    s.set_defaults(fn=cmd_agent)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except llm.Refused as exc:
        sys.exit(f"Claude declined: {exc}")
    finally:
        u = llm.USAGE
        if u.input_tokens or u.output_tokens:
            print(
                f"\n[usage] in={u.input_tokens} out={u.output_tokens} "
                f"cache_read={u.cache_read_input_tokens} cache_write={u.cache_creation_input_tokens}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
