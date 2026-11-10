"""The same reminder assistant, written with the SDK's tool runner.

Once the manual loop is understood, the runner removes it: decorate a typed
function and the SDK builds the schema from the signature and docstring,
calls the tool, and loops until Claude is done.
"""

from __future__ import annotations

from typing import Any

from anthropic import beta_tool

from claude_api_lab import llm
from claude_api_lab.tools import reminders


def build_tools(store: reminders.ReminderStore) -> list[Any]:
    @beta_tool
    def get_current_datetime(date_format: str) -> str:
        """Return the current local date and time.

        Args:
            date_format: strftime pattern, e.g. '%Y-%m-%d %H:%M:%S'.
        """
        return reminders.get_current_datetime(date_format)

    @beta_tool
    def add_duration_to_datetime(datetime_str: str, duration: int, unit: str, input_format: str) -> str:
        """Add a duration to a date; use for all date arithmetic.

        Args:
            datetime_str: The starting date.
            duration: Amount to add; negative to subtract.
            unit: seconds, minutes, hours, days or weeks.
            input_format: strftime pattern of datetime_str.
        """
        return reminders.add_duration_to_datetime(datetime_str, duration, unit, input_format)

    @beta_tool
    def set_reminder(content: str, timestamp: str) -> str:
        """Schedule a reminder for the user.

        Args:
            content: What to remind the user about.
            timestamp: When, e.g. '2026-11-20 09:00'.
        """
        return store.set_reminder(content, timestamp)

    return [get_current_datetime, add_duration_to_datetime, set_reminder]


def remind(client: Any, request: str, store: reminders.ReminderStore) -> str:
    runner = client.beta.messages.tool_runner(
        model=llm.MODEL,
        max_tokens=16000,
        tools=build_tools(store),
        messages=[{"role": "user", "content": request}],
    )
    final = runner.until_done()
    llm.USAGE.add(final)
    return llm.text_of(llm.check_stop(final))
