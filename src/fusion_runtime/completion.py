from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .types import (
    CompletionOutcome,
    CompletionRecord,
    Finish,
    ModelResponse,
    ModelStreamEvent,
    StreamError,
    TextDelta,
    ToolCallDelta,
    Usage,
)

FINAL_ANSWER_MISSING = "final_answer_missing"
OUTPUT_TRUNCATED = "output_truncated"
INVALID_TOOL_CALL = "invalid_tool_call"
PROVIDER_PROTOCOL_ERROR = "provider_protocol_error"

_FAILURE_TAG_ORDER = (
    FINAL_ANSWER_MISSING,
    OUTPUT_TRUNCATED,
    INVALID_TOOL_CALL,
    PROVIDER_PROTOCOL_ERROR,
)
_TRUNCATED_REASONS = {"length", "max_tokens", "max_output_tokens", "token_limit"}
_FINISH_REASON_ALIASES = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "completed": "stop",
    "complete": "stop",
    "function_call": "tool_calls",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "max_output_tokens": "length",
    "token_limit": "length",
}


def normalize_finish_reason(reason: str | None) -> str | None:
    """Map common provider finish reasons to one stable internal vocabulary."""

    if reason is None:
        return None
    normalized = reason.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    return _FINISH_REASON_ALIASES.get(normalized, normalized)


def classify_response(response: ModelResponse) -> CompletionOutcome:
    """Classify a completed non-streaming main-model response deterministically."""

    tool_call_count, valid_tool_calls, invalid_tool_call = _tool_call_summary(response.tool_calls)
    finish_reason = normalize_finish_reason(response.finish_reason)
    if finish_reason == "tool_calls" and tool_call_count == 0:
        invalid_tool_call = True
    return _build_outcome(
        terminal=True,
        finish_reason=finish_reason,
        has_text_output=bool(response.content.strip()),
        tool_call_count=tool_call_count,
        valid_tool_calls=valid_tool_calls,
        invalid_tool_call=invalid_tool_call,
        usage_reported=bool(response.usage),
    )


@dataclass
class _StreamToolCall:
    id: str | None = None
    name: str | None = None
    argument_parts: list[str] = field(default_factory=list)
    conflicting_identity: bool = False

    def add(self, event: ToolCallDelta) -> None:
        if event.id is not None:
            if self.id is not None and self.id != event.id:
                self.conflicting_identity = True
            self.id = event.id
        if event.name is not None:
            if self.name is not None and self.name != event.name:
                self.conflicting_identity = True
            self.name = event.name
        if event.arguments:
            self.argument_parts.append(event.arguments)

    def as_tool_call(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": "".join(self.argument_parts),
            },
        }


class CompletionTracker:
    """Build immutable completion snapshots while a canonical stream is consumed."""

    def __init__(self, record: CompletionRecord | None = None) -> None:
        self.record = record or CompletionRecord()
        self._text_parts: list[str] = []
        self._tool_calls: dict[int, _StreamToolCall] = {}
        self._finish_reason: str | None = None
        self._usage_event_seen = False
        self._all_usage_reported = True
        self._stream_error: StreamError | None = None
        self._terminal_seen = False
        self._ended_without_terminal = False
        self._cancelled = False
        self._publish()

    @property
    def terminal_seen(self) -> bool:
        return self._terminal_seen

    def observe(self, event: ModelStreamEvent) -> None:
        if isinstance(event, TextDelta):
            self._text_parts.append(event.text)
        elif isinstance(event, ToolCallDelta):
            self._tool_calls.setdefault(event.index, _StreamToolCall()).add(event)
        elif isinstance(event, Finish):
            self._finish_reason = normalize_finish_reason(event.reason)
            self._terminal_seen = True
        elif isinstance(event, Usage):
            self._usage_event_seen = True
            self._all_usage_reported = self._all_usage_reported and event.reported_for_all_attempts
        elif isinstance(event, StreamError):
            self._stream_error = event
            self._terminal_seen = True
        self._publish()

    def end(self) -> None:
        """Record a provider/plugin stream that stopped without a terminal event."""

        if not self._terminal_seen and not self._cancelled:
            self._ended_without_terminal = True
            self._publish()

    def cancel(self) -> None:
        """Record consumer cancellation without blaming the model for missing output."""

        if not self._terminal_seen:
            self._cancelled = True
            self._publish()

    def _publish(self) -> None:
        calls = [state.as_tool_call() for _, state in sorted(self._tool_calls.items())]
        tool_call_count, valid_tool_calls, invalid_tool_call = _tool_call_summary(calls)
        if any(state.conflicting_identity for state in self._tool_calls.values()):
            invalid_tool_call = True
        if self._finish_reason == "tool_calls" and tool_call_count == 0:
            invalid_tool_call = True
        error_code = self._stream_error.code if self._stream_error is not None else None
        self.record.outcome = _build_outcome(
            terminal=self._terminal_seen,
            finish_reason=self._finish_reason,
            has_text_output=bool("".join(self._text_parts).strip()),
            tool_call_count=tool_call_count,
            valid_tool_calls=valid_tool_calls,
            invalid_tool_call=invalid_tool_call,
            usage_reported=self._usage_event_seen and self._all_usage_reported,
            stream_error_code=error_code,
            ended_without_terminal=self._ended_without_terminal,
            cancelled=self._cancelled,
        )


def _build_outcome(
    *,
    terminal: bool,
    finish_reason: str | None,
    has_text_output: bool,
    tool_call_count: int,
    valid_tool_calls: int,
    invalid_tool_call: bool,
    usage_reported: bool,
    stream_error_code: str | None = None,
    ended_without_terminal: bool = False,
    cancelled: bool = False,
) -> CompletionOutcome:
    has_tool_call = tool_call_count > 0
    has_valid_tool_call = valid_tool_calls > 0
    has_public_output = has_text_output or has_valid_tool_call
    infrastructure_failure = stream_error_code is not None or ended_without_terminal
    finalized_by_model = terminal and stream_error_code is None

    tags: set[str] = set()
    if (finalized_by_model or infrastructure_failure) and not has_public_output:
        tags.add(FINAL_ANSWER_MISSING)
    if finish_reason in _TRUNCATED_REASONS:
        tags.add(OUTPUT_TRUNCATED)
    if (finalized_by_model or infrastructure_failure) and invalid_tool_call:
        tags.add(INVALID_TOOL_CALL)
    if stream_error_code == PROVIDER_PROTOCOL_ERROR or ended_without_terminal:
        tags.add(PROVIDER_PROTOCOL_ERROR)

    if cancelled:
        status = "cancelled"
    elif infrastructure_failure:
        status = "failed"
    elif not terminal:
        status = "pending"
    elif tags:
        status = "incomplete"
    else:
        status = "completed"

    return CompletionOutcome(
        status=status,
        finish_reason=finish_reason,
        has_public_output=has_public_output,
        has_text_output=has_text_output,
        has_tool_call=has_tool_call,
        has_valid_tool_call=has_valid_tool_call,
        tool_call_count=tool_call_count,
        usage_reported=usage_reported,
        infrastructure_failure=infrastructure_failure,
        failure_tags=tuple(tag for tag in _FAILURE_TAG_ORDER if tag in tags),
    )


def _tool_call_summary(tool_calls: list[dict[str, Any]]) -> tuple[int, int, bool]:
    valid = sum(1 for call in tool_calls if _valid_tool_call(call))
    return len(tool_calls), valid, valid != len(tool_calls)


def _valid_tool_call(call: dict[str, Any]) -> bool:
    if not isinstance(call, dict):
        return False
    call_id = call.get("id")
    function = call.get("function")
    if not isinstance(call_id, str) or not call_id.strip() or not isinstance(function, dict):
        return False
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name.strip() or not isinstance(arguments, str):
        return False
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed_arguments, dict)
