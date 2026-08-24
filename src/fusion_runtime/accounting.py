from __future__ import annotations

from typing import Any

USAGE_MISSING = "usage_missing"
ATTEMPT_USAGE_MISSING = "attempt_usage_missing"
USAGE_TOKENS_MISSING = "usage_tokens_missing"
USAGE_VALUE_INVALID = "usage_value_invalid"
USAGE_TOTAL_MISMATCH = "usage_total_mismatch"

_ISSUE_ORDER = (
    USAGE_MISSING,
    ATTEMPT_USAGE_MISSING,
    USAGE_TOKENS_MISSING,
    USAGE_VALUE_INVALID,
    USAGE_TOTAL_MISMATCH,
)


def assess_usage(
    usage: dict[str, Any],
    *,
    report_seen: bool,
    reported_for_all_attempts: bool = True,
) -> tuple[bool, tuple[str, ...]]:
    """Return all-attempt coverage and deterministic accounting issue codes."""

    usage_reported = report_seen and bool(usage) and reported_for_all_attempts
    if not report_seen or not usage:
        return usage_reported, (USAGE_MISSING,)

    issues: set[str] = set()
    if not reported_for_all_attempts:
        issues.add(ATTEMPT_USAGE_MISSING)

    counters = list(_token_counters(usage))
    if not counters:
        issues.add(USAGE_TOKENS_MISSING)
    elif any(not _valid_counter(value) for _key, value in counters):
        issues.add(USAGE_VALUE_INVALID)

    input_tokens = _first_valid_counter(usage, "input_tokens", "prompt_tokens")
    output_tokens = _first_valid_counter(usage, "output_tokens", "completion_tokens")
    total_tokens = _first_valid_counter(usage, "total_tokens")
    if (
        input_tokens is not None
        and output_tokens is not None
        and total_tokens is not None
        and total_tokens != input_tokens + output_tokens
    ):
        issues.add(USAGE_TOTAL_MISMATCH)

    return usage_reported, tuple(issue for issue in _ISSUE_ORDER if issue in issues)


def _token_counters(values: dict[str, Any]):
    for key, value in values.items():
        normalized = str(key).strip().lower()
        if isinstance(value, dict):
            yield from _token_counters(value)
        elif normalized in {"token", "tokens"} or normalized.endswith(("_token", "_tokens")):
            yield normalized, value


def _valid_counter(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _first_valid_counter(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if _valid_counter(value):
            return value
    return None
