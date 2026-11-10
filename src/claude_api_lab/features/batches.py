"""Message Batches: many independent requests at half price, asynchronously.

Results come back in any order; always key them by custom_id.
"""

from __future__ import annotations

import time
from typing import Any

from claude_api_lab import llm


def submit(client: Any, prompts: dict[str, str], *, max_tokens: int = 4000) -> str:
    requests = [
        {
            "custom_id": custom_id,
            "params": {
                "model": llm.MODEL,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        for custom_id, prompt in prompts.items()
    ]
    return client.messages.batches.create(requests=requests).id


def wait(client: Any, batch_id: str, *, poll_seconds: float = 30, timeout: float = 24 * 3600) -> None:
    deadline = time.monotonic() + timeout
    while client.messages.batches.retrieve(batch_id).processing_status != "ended":
        if time.monotonic() > deadline:
            raise TimeoutError(f"batch {batch_id} still running after {timeout}s")
        time.sleep(poll_seconds)


def results(client: Any, batch_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in client.messages.batches.results(batch_id):
        if item.result.type == "succeeded":
            out[item.custom_id] = llm.text_of(item.result.message)
        else:
            out[item.custom_id] = f"<{item.result.type}>"
    return out
