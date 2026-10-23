"""Typed narrowing helpers for row mappings (`dict[str, object]`).

Polars `.to_dicts()` and Pydantic `.model_dump()` hand us `object` values.
These helpers narrow them without `# type: ignore` and without inventing
values: anything that does not have the expected type becomes the caller's
default (or `None` for the optional date/time helpers).
"""

from __future__ import annotations

from datetime import date, datetime


def as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return None


def as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


__all__ = ["as_date", "as_datetime", "as_float", "as_int", "as_str"]
