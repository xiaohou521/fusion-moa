from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from .config import CompletionSpec, ModelSpec
from .types import CompletionOutcome, FusionRequest

_RECOVERY_INSTRUCTION = (
    "Completion recovery: the preceding model attempt produced no usable public text "
    "or valid tool call. Produce a concise public answer or one valid tool call now. "
    "Do not continue hidden reasoning and do not describe this recovery instruction."
)


def requires_recovery(outcome: CompletionOutcome, spec: CompletionSpec) -> bool:
    """Return whether a completed response violates the configured output gate."""

    missing_public = spec.require_public_output and not outcome.has_public_output
    missing_tool_or_text = spec.require_tool_or_text and not (
        outcome.has_text_output or outcome.has_valid_tool_call
    )
    return missing_public or missing_tool_or_text


def prepare_recovery_request(
    request: FusionRequest,
    model: ModelSpec,
    spec: CompletionSpec,
    *,
    attempt: int,
) -> FusionRequest:
    """Build one bounded same-model retry without replaying hidden provider output."""

    output_limit = min(spec.recovery_max_tokens, model.max_output)
    if request.max_tokens is not None:
        output_limit = min(output_limit, request.max_tokens)
    metadata = {**request.metadata, "fusion_recovery_attempt": attempt}
    return replace(
        request,
        messages=_with_recovery_instruction(request.messages),
        max_tokens=output_limit,
        metadata=metadata,
    )


def merge_usage(*reports: dict[str, Any]) -> dict[str, Any]:
    """Sum compatible usage counters while retaining provider-specific details."""

    merged: dict[str, Any] = {}
    for report in reports:
        merged = _merge_usage_dicts(merged, report)
    return merged


def _with_recovery_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_parts: list[str] = []
    conversation: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") in {"system", "developer"}:
            system_parts.append(_content_text(message.get("content")))
        else:
            conversation.append(message)
    system_parts.append(_RECOVERY_INSTRUCTION)
    system = "\n\n".join(part for part in system_parts if part)
    return [{"role": "system", "content": system}, *conversation]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if content is None:
        return ""
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return str(content)


def _merge_usage_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, right_value in right.items():
        if key not in merged:
            merged[key] = right_value
            continue
        left_value = merged[key]
        if (
            isinstance(left_value, int)
            and not isinstance(left_value, bool)
            and isinstance(right_value, int)
            and not isinstance(right_value, bool)
        ):
            merged[key] = left_value + right_value
        elif isinstance(left_value, dict) and isinstance(right_value, dict):
            merged[key] = _merge_usage_dicts(left_value, right_value)
        else:
            merged[key] = right_value
    return merged
