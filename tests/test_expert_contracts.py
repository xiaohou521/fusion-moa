from __future__ import annotations

import pytest

from fusion_runtime.expert_contracts import call_adaptive_expert, parse_expert_advice
from fusion_runtime.types import ModelResponse


def response(
    content: str,
    *,
    finish_reason: str = "stop",
    completion_tokens: int = 20,
    usage: bool = True,
) -> ModelResponse:
    counters = (
        {
            "prompt_tokens": 10,
            "completion_tokens": completion_tokens,
            "total_tokens": 10 + completion_tokens,
        }
        if usage
        else {}
    )
    return ModelResponse(
        content=content,
        finish_reason=finish_reason,
        usage=counters,
    )


class FakeRuntime:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = []

    async def call_model(self, model_name, request):
        self.calls.append((model_name, request))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def invoke(runtime: FakeRuntime):
    return await call_adaptive_expert(
        runtime,
        "reviewer",
        [{"role": "user", "content": "review"}],
        token_tiers=(512, 1024, 2048),
        max_advice_chars=1600,
        thinking_mode="provider-default",
        temperature=0.2,
        seed=7,
        role="reviewer",
    )


@pytest.mark.asyncio
async def test_invalid_json_at_reported_ceiling_retries_once() -> None:
    runtime = FakeRuntime(
        [
            response("{", completion_tokens=512),
            response('{"action":"abstain","advice":""}', completion_tokens=12),
        ]
    )

    result = await invoke(runtime)

    assert result.valid is True
    assert result.action == "abstain"
    assert result.attempts == 2
    assert result.selected_max_tokens == 1024
    assert [call.max_tokens for _name, call in runtime.calls] == [512, 1024]


@pytest.mark.asyncio
async def test_invalid_json_below_ceiling_fails_without_budget_expansion() -> None:
    runtime = FakeRuntime([response("{", completion_tokens=511)])

    result = await invoke(runtime)

    assert result.valid is False
    assert result.failure == "expert envelope invalid: json_invalid"
    assert len(runtime.calls) == 1


@pytest.mark.asyncio
async def test_all_tiers_exhausted_is_bounded_and_accounted() -> None:
    runtime = FakeRuntime(
        [
            response("{", finish_reason="length", completion_tokens=512),
            response("{", finish_reason="max_tokens", completion_tokens=1024),
            response("{", finish_reason="token_limit", completion_tokens=2048),
        ]
    )

    result = await invoke(runtime)

    assert result.valid is False
    assert result.failure == "expert exhausted adaptive token tiers"
    assert result.attempts == 3
    assert result.usage["completion_tokens"] == 3584


@pytest.mark.asyncio
async def test_missing_usage_on_any_attempt_keeps_accounting_incomplete() -> None:
    runtime = FakeRuntime(
        [
            response("{", finish_reason="length", usage=False),
            response('{"action":"advise","advice":"use a heap"}'),
        ]
    )

    result = await invoke(runtime)

    assert result.valid is True
    assert result.usage_complete is False


@pytest.mark.parametrize(
    ("content", "valid", "failure"),
    [
        ('{"action":"advise","advice":"use a heap"}', True, None),
        ('{"action":"abstain","advice":""}', True, None),
        ('{"action":"advise","advice":""}', False, "advice_empty"),
        ('{"action":"abstain","advice":"reason"}', False, "abstain_not_empty"),
        ('{"action":"advise","advice":"ok","extra":1}', False, "schema_mismatch"),
    ],
)
def test_envelope_semantics_are_strict(content, valid, failure) -> None:
    _action, _advice, actual_valid, actual_failure = parse_expert_advice(
        content,
        max_advice_chars=1600,
    )

    assert actual_valid is valid
    assert actual_failure == failure
