from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

ThinkingMode = Literal["provider-default", "disabled", "bounded"]
THINKING_MODES: frozenset[str] = frozenset({"provider-default", "disabled", "bounded"})


@dataclass(frozen=True)
class ThinkingConfig:
    """Provider-neutral thinking controls for one model call."""

    mode: ThinkingMode = "provider-default"
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in THINKING_MODES:
            raise ValueError(f"unknown thinking mode: {self.mode!r}")
        if self.mode == "bounded":
            if (
                not isinstance(self.budget_tokens, int)
                or isinstance(self.budget_tokens, bool)
                or self.budget_tokens <= 0
            ):
                raise ValueError("bounded thinking requires a positive budget_tokens")
        elif self.budget_tokens is not None:
            raise ValueError("budget_tokens is only valid for bounded thinking")


@dataclass(frozen=True)
class FusionRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    max_tokens: int | None = None
    temperature: float | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reasoning_effort is not None and self.thinking.mode != "provider-default":
            raise ValueError(
                "reasoning_effort cannot be combined with a normalized thinking override"
            )


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, Any] = field(default_factory=dict)
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


@dataclass(frozen=True)
class RecoveryOutcome:
    """Bounded recovery evidence for one public completion request."""

    attempts: int = 0
    succeeded: bool = False
    duration_ms: float = 0.0
    failure_code: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    initial_completion: CompletionOutcome | None = field(default=None, repr=False)


@dataclass
class CompletionRecord:
    """Mutable holder whose value is replaced with immutable stream snapshots."""

    outcome: CompletionOutcome = field(default_factory=CompletionOutcome)


@dataclass
class RecoveryRecord:
    """Mutable holder for recovery evidence produced while a stream is consumed."""

    outcome: RecoveryOutcome = field(default_factory=RecoveryOutcome)


@dataclass(frozen=True)
class FusionResult:
    response: ModelResponse
    route: str
    experts_used: tuple[str, ...] = ()
    fallback_reason: str | None = None
    trace_id: str = ""
    completion: CompletionOutcome = field(default_factory=CompletionOutcome, repr=False)
    recovery: RecoveryOutcome = field(default_factory=RecoveryOutcome, repr=False)


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
    usage: dict[str, Any]
    reported_for_all_attempts: bool = True


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
    recovery: RecoveryRecord = field(default_factory=RecoveryRecord, repr=False)
