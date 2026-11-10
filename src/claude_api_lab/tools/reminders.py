"""Tool functions and their JSON schemas.

Course technique: write a plain function, then a schema whose description
says what the tool does, when to use it, and what it returns. `strict: True`
makes the API guarantee the input matches the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

UNITS = ("seconds", "minutes", "hours", "days", "weeks")


class ToolInputError(ValueError):
    """Raised for bad input; the loop returns it to Claude as is_error."""


def get_current_datetime(date_format: str = "%Y-%m-%d %H:%M:%S", *, now: datetime | None = None) -> str:
    if not date_format:
        raise ToolInputError("date_format cannot be empty")
    return (now or datetime.now()).strftime(date_format)


def add_duration_to_datetime(
    datetime_str: str, duration: int, unit: str = "days", input_format: str = "%Y-%m-%d"
) -> str:
    if unit not in UNITS:
        raise ToolInputError(f"unit must be one of {', '.join(UNITS)}")
    try:
        start = datetime.strptime(datetime_str, input_format)
    except ValueError as exc:
        raise ToolInputError(f"could not parse {datetime_str!r} with {input_format!r}") from exc
    return (start + timedelta(**{unit: duration})).strftime("%A, %B %d, %Y %I:%M:%S %p")


@dataclass
class ReminderStore:
    reminders: list[dict[str, str]] = field(default_factory=list)

    def set_reminder(self, content: str, timestamp: str) -> str:
        if not content.strip():
            raise ToolInputError("content cannot be empty")
        self.reminders.append({"content": content, "timestamp": timestamp})
        return f"Reminder set for {timestamp}: {content}"


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    return {**schema, "strict": True}


GET_CURRENT_DATETIME = _strict(
    {
        "name": "get_current_datetime",
        "description": (
            "Returns the current local date and time, formatted with a strftime pattern. "
            "Use it whenever the user's request depends on today's date or the current time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date_format": {
                    "type": "string",
                    "description": "strftime pattern, e.g. '%Y-%m-%d %H:%M:%S'.",
                }
            },
            "required": ["date_format"],
            "additionalProperties": False,
        },
    }
)

ADD_DURATION = _strict(
    {
        "name": "add_duration_to_datetime",
        "description": (
            "Adds a duration to a date and returns the resulting date and time in long form. "
            "Use it for any date arithmetic instead of computing dates yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "datetime_str": {"type": "string", "description": "The starting date."},
                "duration": {"type": "integer", "description": "Amount to add; negative to subtract."},
                "unit": {"type": "string", "enum": list(UNITS)},
                "input_format": {"type": "string", "description": "strftime pattern of datetime_str."},
            },
            "required": ["datetime_str", "duration", "unit", "input_format"],
            "additionalProperties": False,
        },
    }
)

SET_REMINDER = _strict(
    {
        "name": "set_reminder",
        "description": "Schedules a reminder for the user at an exact timestamp. Returns a confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remind the user about."},
                "timestamp": {"type": "string", "description": "When, e.g. '2026-11-20 09:00'."},
            },
            "required": ["content", "timestamp"],
            "additionalProperties": False,
        },
    }
)

SCHEMAS = [GET_CURRENT_DATETIME, ADD_DURATION, SET_REMINDER]
