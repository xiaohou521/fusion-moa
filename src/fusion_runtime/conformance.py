from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

from .types import Finish, ModelStreamEvent, StreamError, TextDelta, ToolCallDelta, Usage


class StreamContractError(ValueError):
    """Raised when a provider stream violates the public event contract."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = tuple(violations)
        super().__init__("; ".join(violations))


async def collect_stream(events: AsyncIterator[ModelStreamEvent]) -> list[ModelStreamEvent]:
    """Collect a provider stream for plugin conformance tests."""

    return [event async for event in events]


def stream_violations(events: Iterable[ModelStreamEvent]) -> list[str]:
    """Return every ordering or value violation in a canonical stream."""

    items = list(events)
    violations: list[str] = []
    finish_indices = [index for index, event in enumerate(items) if isinstance(event, Finish)]
    error_indices = [index for index, event in enumerate(items) if isinstance(event, StreamError)]
    if len(finish_indices) + len(error_indices) == 0:
        violations.append("stream has no Finish or StreamError terminal event")
    if len(finish_indices) > 1:
        violations.append("stream has more than one Finish event")
    if len(error_indices) > 1:
        violations.append("stream has more than one StreamError event")
    if finish_indices and error_indices:
        violations.append("stream cannot contain both Finish and StreamError")
    if error_indices and error_indices[0] != len(items) - 1:
        violations.append("StreamError must be the final event")
    if finish_indices:
        for event in items[finish_indices[0] + 1 :]:
            if not isinstance(event, Usage):
                violations.append("only Usage may follow Finish")
                break

    for event in items:
        if isinstance(event, TextDelta) and not event.text:
            violations.append("TextDelta.text must not be empty")
        elif isinstance(event, ToolCallDelta):
            if event.index < 0:
                violations.append("ToolCallDelta.index must be non-negative")
            if event.id is None and event.name is None and not event.arguments:
                violations.append("ToolCallDelta must carry id, name, or arguments")
        elif isinstance(event, Finish) and not event.reason:
            violations.append("Finish.reason must not be empty")
        elif isinstance(event, Usage):
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in event.usage.values()
            ):
                violations.append("Usage values must be non-negative integers")
        elif isinstance(event, StreamError):
            if not event.message:
                violations.append("StreamError.message must not be empty")
            if not event.code:
                violations.append("StreamError.code must not be empty")
    return violations


def assert_stream_conforms(events: Iterable[ModelStreamEvent]) -> None:
    """Raise one aggregate error if canonical stream events are invalid."""

    violations = stream_violations(events)
    if violations:
        raise StreamContractError(violations)
