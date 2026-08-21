from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FusionRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class CompletionOutcome:
    """Provider-neutral classification of one authoritative main-model completion."""

    status: str = "pending"
    finish_reason: str | None = None
    has_public_output: bool = False
    has_text_output: bool = False
    has_tool_call: bool = False
    has_valid_tool_call: bool = False
    tool_call_count: int = 0
    usage_reported: bool = False
    infrastructure_failure: bool = False
    failure_tags: tuple[str, ...] = ()


@dataclass
class CompletionRecord:
    """Mutable holder whose value is replaced with immutable stream snapshots."""

    outcome: CompletionOutcome = field(default_factory=CompletionOutcome)


@dataclass(frozen=True)
class FusionResult:
    response: ModelResponse
    route: str
    experts_used: tuple[str, ...] = ()
    fallback_reason: str | None = None
    trace_id: str = ""
    completion: CompletionOutcome = field(default_factory=CompletionOutcome, repr=False)


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(frozen=True)
class Finish:
    reason: str


@dataclass(frozen=True)
class Usage:
    usage: dict[str, int]


@dataclass(frozen=True)
class StreamError:
    message: str
    code: str = "provider_stream_error"
    retryable: bool = False


ModelStreamEvent = TextDelta | ToolCallDelta | Finish | Usage | StreamError


@dataclass(frozen=True)
class PreparedCall:
    model_name: str
    request: FusionRequest
    route: str
    experts_used: tuple[str, ...] = ()
    fallback_reason: str | None = None


@dataclass(frozen=True)
class FusionStream:
    events: AsyncIterator[ModelStreamEvent]
    route: str
    experts_used: tuple[str, ...] = ()
    fallback_reason: str | None = None
    trace_id: str = ""
    completion: CompletionRecord = field(default_factory=CompletionRecord, repr=False)
