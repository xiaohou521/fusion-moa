from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .accounting import assess_usage
from .errors import ProviderError
from .recovery import merge_usage
from .types import (
    FusionRequest,
    ModelResponse,
    StructuredOutputConfig,
    ThinkingConfig,
)

_TRUNCATED_REASONS = {"length", "max_tokens", "max_output_tokens", "token_limit"}


class ExpertRuntimeAccess(Protocol):
    async def call_model(self, name: str, request: FusionRequest) -> ModelResponse: ...


@dataclass(frozen=True)
class AdaptiveExpertResult:
    action: str
    advice: str
    valid: bool
    failure: str | None
    attempts: int
    selected_max_tokens: int
    usage: dict[str, Any]
    usage_complete: bool


@dataclass(frozen=True)
class ExpertCorrection:
    """A compact correction that cannot redefine the final response budget."""

    risk_class: str
    must_fix: tuple[str, ...]
    counterexample: str
    solution_delta: str


@dataclass(frozen=True)
class ConstrainedExpertResult:
    action: str
    correction: ExpertCorrection | None
    valid: bool
    failure: str | None
    failure_retryable: bool
    attempts: int
    selected_max_tokens: int
    usage: dict[str, Any]
    usage_complete: bool
    recoveries: tuple[str, ...] = ()


def expert_advice_schema(max_advice_chars: int) -> dict[str, Any]:
    """Return the strict advise-or-abstain envelope used by adaptive experts."""

    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "action": {"const": "advise"},
                    "advice": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": max_advice_chars,
                    },
                },
                "required": ["action", "advice"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "action": {"const": "abstain"},
                    "advice": {"const": ""},
                },
                "required": ["action", "advice"],
                "additionalProperties": False,
            },
        ]
    }


def expert_correction_schema(
    *,
    max_must_fix_items: int,
    max_item_chars: int,
    max_counterexample_chars: int,
    max_solution_delta_chars: int,
) -> dict[str, Any]:
    """Return the strict correction-or-abstain envelope for mandatory experts."""

    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "action": {"const": "advise"},
                    "risk_class": {
                        "type": "string",
                        "enum": ["correctness", "completeness", "complexity", "tool-use"],
                    },
                    "must_fix": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": max_must_fix_items,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": max_item_chars,
                        },
                    },
                    "counterexample": {
                        "type": "string",
                        "maxLength": max_counterexample_chars,
                    },
                    "solution_delta": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": max_solution_delta_chars,
                    },
                },
                "required": [
                    "action",
                    "risk_class",
                    "must_fix",
                    "counterexample",
                    "solution_delta",
                ],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"action": {"const": "abstain"}},
                "required": ["action"],
                "additionalProperties": False,
            },
        ]
    }


async def call_constrained_expert(
    runtime: ExpertRuntimeAccess,
    model_name: str,
    messages: list[dict[str, Any]],
    *,
    token_tiers: tuple[int, ...],
    retry_attempts: int,
    max_must_fix_items: int,
    max_item_chars: int,
    max_counterexample_chars: int,
    max_solution_delta_chars: int,
    thinking_mode: str,
    temperature: float | None,
    seed: int | None,
    role: str,
) -> ConstrainedExpertResult:
    """Obtain one valid compact review with bounded retries and tier expansion."""

    _validate_tiers(token_tiers)
    if (
        not isinstance(retry_attempts, int)
        or isinstance(retry_attempts, bool)
        or retry_attempts < 0
    ):
        raise ValueError("retry_attempts must be a non-negative integer")
    schema = expert_correction_schema(
        max_must_fix_items=max_must_fix_items,
        max_item_chars=max_item_chars,
        max_counterexample_chars=max_counterexample_chars,
        max_solution_delta_chars=max_solution_delta_chars,
    )
    usage: dict[str, Any] = {}
    usage_complete = True
    attempts = 0
    recoveries: list[str] = []
    for tier_index, max_tokens in enumerate(token_tiers):
        response: ModelResponse | None = None
        for retry_index in range(retry_attempts + 1):
            attempts += 1
            try:
                response = await runtime.call_model(
                    model_name,
                    FusionRequest(
                        messages=messages,
                        tools=[],
                        reasoning_effort=None,
                        thinking=ThinkingConfig(mode=thinking_mode),
                        structured_output=StructuredOutputConfig(
                            mode="json-schema",
                            name="expert_correction",
                            schema=schema,
                        ),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        seed=seed,
                        metadata={
                            "fusion_expert_role": role,
                            "fusion_expert_attempt": attempts,
                            "fusion_expert_retry": retry_index,
                            "fusion_expert_max_tokens": max_tokens,
                        },
                    ),
                )
                break
            except ProviderError as exc:
                usage_complete = False
                if exc.retryable and retry_index < retry_attempts:
                    recoveries.append(type(exc).__name__)
                    continue
                return ConstrainedExpertResult(
                    action="abstain",
                    correction=None,
                    valid=False,
                    failure=f"expert call failed: {type(exc).__name__}",
                    failure_retryable=exc.retryable,
                    attempts=attempts,
                    selected_max_tokens=max_tokens,
                    usage=usage,
                    usage_complete=False,
                    recoveries=tuple(recoveries),
                )
            except Exception as exc:
                return ConstrainedExpertResult(
                    action="abstain",
                    correction=None,
                    valid=False,
                    failure=f"expert call failed: {type(exc).__name__}",
                    failure_retryable=False,
                    attempts=attempts,
                    selected_max_tokens=max_tokens,
                    usage=usage,
                    usage_complete=False,
                    recoveries=tuple(recoveries),
                )

        if response is None:
            raise AssertionError("constrained expert retry loop produced no response")
        usage = merge_usage(usage, response.usage)
        _reported, issues = assess_usage(response.usage, report_seen=bool(response.usage))
        usage_complete = usage_complete and not issues
        action, correction, valid, failure = parse_expert_correction(
            response.content,
            max_must_fix_items=max_must_fix_items,
            max_item_chars=max_item_chars,
            max_counterexample_chars=max_counterexample_chars,
            max_solution_delta_chars=max_solution_delta_chars,
        )
        if _is_truncated(
            response,
            parse_failure=failure,
            requested_max_tokens=max_tokens,
        ):
            if tier_index + 1 < len(token_tiers):
                continue
            return ConstrainedExpertResult(
                action="abstain",
                correction=None,
                valid=False,
                failure="expert exhausted adaptive token tiers",
                failure_retryable=False,
                attempts=attempts,
                selected_max_tokens=max_tokens,
                usage=usage,
                usage_complete=usage_complete,
                recoveries=tuple(recoveries),
            )
        if _normalize_finish_reason(response.finish_reason) != "stop":
            return ConstrainedExpertResult(
                action="abstain",
                correction=None,
                valid=False,
                failure=f"expert finish reason was {response.finish_reason}",
                failure_retryable=False,
                attempts=attempts,
                selected_max_tokens=max_tokens,
                usage=usage,
                usage_complete=usage_complete,
                recoveries=tuple(recoveries),
            )
        if not valid:
            return ConstrainedExpertResult(
                action="abstain",
                correction=None,
                valid=False,
                failure=f"expert envelope invalid: {failure}",
                failure_retryable=False,
                attempts=attempts,
                selected_max_tokens=max_tokens,
                usage=usage,
                usage_complete=usage_complete,
                recoveries=tuple(recoveries),
            )
        return ConstrainedExpertResult(
            action=action,
            correction=correction,
            valid=True,
            failure=None,
            failure_retryable=False,
            attempts=attempts,
            selected_max_tokens=max_tokens,
            usage=usage,
            usage_complete=usage_complete,
            recoveries=tuple(recoveries),
        )
    raise AssertionError("unreachable constrained expert loop")


def parse_expert_correction(
    content: str,
    *,
    max_must_fix_items: int,
    max_item_chars: int,
    max_counterexample_chars: int,
    max_solution_delta_chars: int,
) -> tuple[str, ExpertCorrection | None, bool, str | None]:
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return "abstain", None, False, "json_invalid"
    if not isinstance(value, dict):
        return "abstain", None, False, "schema_mismatch"
    if value == {"action": "abstain"}:
        return "abstain", None, True, None
    required = {"action", "risk_class", "must_fix", "counterexample", "solution_delta"}
    if set(value) != required or value.get("action") != "advise":
        return "abstain", None, False, "schema_mismatch"
    risk_class = value.get("risk_class")
    if risk_class not in {"correctness", "completeness", "complexity", "tool-use"}:
        return "abstain", None, False, "risk_class_invalid"
    must_fix = value.get("must_fix")
    if (
        not isinstance(must_fix, list)
        or not 1 <= len(must_fix) <= max_must_fix_items
        or any(
            not isinstance(item, str) or not item.strip() or len(item) > max_item_chars
            for item in must_fix
        )
        or len(set(must_fix)) != len(must_fix)
    ):
        return "abstain", None, False, "must_fix_invalid"
    counterexample = value.get("counterexample")
    if not isinstance(counterexample, str) or len(counterexample) > max_counterexample_chars:
        return "abstain", None, False, "counterexample_invalid"
    solution_delta = value.get("solution_delta")
    if (
        not isinstance(solution_delta, str)
        or not solution_delta.strip()
        or len(solution_delta) > max_solution_delta_chars
    ):
        return "abstain", None, False, "solution_delta_invalid"
    return (
        "advise",
        ExpertCorrection(
            risk_class=risk_class,
            must_fix=tuple(item.strip() for item in must_fix),
            counterexample=counterexample.strip(),
            solution_delta=solution_delta.strip(),
        ),
        True,
        None,
    )


async def call_adaptive_expert(
    runtime: ExpertRuntimeAccess,
    model_name: str,
    messages: list[dict[str, Any]],
    *,
    token_tiers: tuple[int, ...],
    max_advice_chars: int,
    thinking_mode: str,
    temperature: float | None,
    seed: int | None,
    role: str,
) -> AdaptiveExpertResult:
    """Retry a structured expert only when evidence shows output truncation."""

    _validate_tiers(token_tiers)
    schema = expert_advice_schema(max_advice_chars)
    usage: dict[str, Any] = {}
    usage_complete = True
    for attempt, max_tokens in enumerate(token_tiers, start=1):
        try:
            response = await runtime.call_model(
                model_name,
                FusionRequest(
                    messages=messages,
                    tools=[],
                    reasoning_effort=None,
                    thinking=ThinkingConfig(mode=thinking_mode),
                    structured_output=StructuredOutputConfig(
                        mode="json-schema",
                        name="expert_advice",
                        schema=schema,
                    ),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    metadata={
                        "fusion_expert_role": role,
                        "fusion_expert_attempt": attempt,
                        "fusion_expert_max_tokens": max_tokens,
                    },
                ),
            )
        except Exception as exc:
            return AdaptiveExpertResult(
                action="abstain",
                advice="",
                valid=False,
                failure=f"expert call failed: {type(exc).__name__}",
                attempts=attempt,
                selected_max_tokens=max_tokens,
                usage=usage,
                usage_complete=False,
            )

        usage = merge_usage(usage, response.usage)
        _reported, issues = assess_usage(response.usage, report_seen=bool(response.usage))
        usage_complete = usage_complete and not issues
        action, advice, valid, failure = parse_expert_advice(
            response.content,
            max_advice_chars=max_advice_chars,
        )
        if _is_truncated(
            response,
            parse_failure=failure,
            requested_max_tokens=max_tokens,
        ):
            if attempt < len(token_tiers):
                continue
            return AdaptiveExpertResult(
                action="abstain",
                advice="",
                valid=False,
                failure="expert exhausted adaptive token tiers",
                attempts=attempt,
                selected_max_tokens=max_tokens,
                usage=usage,
                usage_complete=usage_complete,
            )
        if _normalize_finish_reason(response.finish_reason) != "stop":
            return AdaptiveExpertResult(
                action="abstain",
                advice="",
                valid=False,
                failure=f"expert finish reason was {response.finish_reason}",
                attempts=attempt,
                selected_max_tokens=max_tokens,
                usage=usage,
                usage_complete=usage_complete,
            )
        if not valid:
            return AdaptiveExpertResult(
                action="abstain",
                advice="",
                valid=False,
                failure=f"expert envelope invalid: {failure}",
                attempts=attempt,
                selected_max_tokens=max_tokens,
                usage=usage,
                usage_complete=usage_complete,
            )
        return AdaptiveExpertResult(
            action=action,
            advice=advice,
            valid=True,
            failure=None,
            attempts=attempt,
            selected_max_tokens=max_tokens,
            usage=usage,
            usage_complete=usage_complete,
        )
    raise AssertionError("unreachable adaptive expert loop")


def parse_expert_advice(
    content: str,
    *,
    max_advice_chars: int,
) -> tuple[str, str, bool, str | None]:
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return "abstain", "", False, "json_invalid"
    if not isinstance(value, dict) or set(value) != {"action", "advice"}:
        return "abstain", "", False, "schema_mismatch"
    action = value.get("action")
    advice = value.get("advice")
    if not isinstance(advice, str):
        return "abstain", "", False, "advice_not_string"
    if action == "advise":
        bounded = advice.strip()
        if not bounded:
            return "abstain", "", False, "advice_empty"
        if len(advice) > max_advice_chars:
            return "abstain", "", False, "advice_too_long"
        return "advise", bounded, True, None
    if action == "abstain":
        if advice != "":
            return "abstain", "", False, "abstain_not_empty"
        return "abstain", "", True, None
    return "abstain", "", False, "action_invalid"


def _is_truncated(
    response: ModelResponse,
    *,
    parse_failure: str | None,
    requested_max_tokens: int,
) -> bool:
    if _normalize_finish_reason(response.finish_reason) == "length":
        return True
    completion_tokens = _completion_tokens(response.usage)
    return (
        parse_failure == "json_invalid"
        and completion_tokens is not None
        and completion_tokens >= requested_max_tokens
    )


def _normalize_finish_reason(reason: str | None) -> str:
    normalized = str(reason or "").strip().lower()
    return "length" if normalized in _TRUNCATED_REASONS else normalized


def _completion_tokens(usage: dict[str, Any]) -> int | None:
    for name in ("completion_tokens", "output_tokens"):
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _validate_tiers(token_tiers: tuple[int, ...]) -> None:
    if not token_tiers or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in token_tiers
    ):
        raise ValueError("expert token tiers must be positive integers")
    if tuple(sorted(set(token_tiers))) != token_tiers:
        raise ValueError("expert token tiers must be strictly increasing")
