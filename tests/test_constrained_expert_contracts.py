from __future__ import annotations

import pytest

from fusion_runtime.errors import ProviderHTTPError, ProviderTransportError
from fusion_runtime.expert_contracts import (
    call_constrained_expert,
    expert_correction_schema,
    parse_expert_correction,
)
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


def valid_correction() -> str:
    return (
        "{"
        '"action":"advise",'
        '"risk_class":"correctness",'
        '"must_fix":["preserve stable ordering"],'
        '"counterexample":"equal priorities",'
        '"solution_delta":"add the original index to the key"'
        "}"
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


async def invoke(runtime: FakeRuntime, *, retry_attempts: int = 1):
    return await call_constrained_expert(
        runtime,
        "reviewer",
        [{"role": "user", "content": "review"}],
        token_tiers=(512, 1024, 2048),
        retry_attempts=retry_attempts,
        max_must_fix_items=3,
        max_item_chars=240,
        max_counterexample_chars=400,
        max_solution_delta_chars=600,
        thinking_mode="provider-default",
        temperature=0.2,
        seed=7,
        role="reviewer",
    )


def parse(content: str):
    return parse_expert_correction(
        content,
        max_must_fix_items=3,
        max_item_chars=240,
        max_counterexample_chars=400,
        max_solution_delta_chars=600,
    )


def test_schema_has_no_budget_control_and_bounds_every_correction_field() -> None:
    schema = expert_correction_schema(
        max_must_fix_items=3,
        max_item_chars=240,
        max_counterexample_chars=400,
        max_solution_delta_chars=600,
    )

    advise = schema["oneOf"][0]
    assert "output_budget" not in advise["properties"]
    assert advise["properties"]["must_fix"]["maxItems"] == 3
    assert advise["properties"]["must_fix"]["items"]["maxLength"] == 240
    assert advise["properties"]["counterexample"]["maxLength"] == 400
    assert advise["properties"]["solution_delta"]["maxLength"] == 600


@pytest.mark.parametrize(
    ("content", "valid", "failure"),
    [
        (valid_correction(), True, None),
        ('{"action":"abstain"}', True, None),
        ('{"action":"abstain","must_fix":[]}', False, "schema_mismatch"),
        (
            '{"action":"advise","risk_class":"other","must_fix":["x"],'
            '"counterexample":"","solution_delta":"y"}',
            False,
            "risk_class_invalid",
        ),
        (
            '{"action":"advise","risk_class":"correctness","must_fix":[],'
            '"counterexample":"","solution_delta":"y"}',
            False,
            "must_fix_invalid",
        ),
    ],
)
def test_correction_semantics_are_strict(content, valid, failure) -> None:
    _action, _correction, actual_valid, actual_failure = parse(content)

    assert actual_valid is valid
    assert actual_failure == failure


@pytest.mark.asyncio
async def test_retryable_transport_retries_same_tier_once() -> None:
    runtime = FakeRuntime([ProviderTransportError(), response(valid_correction())])

    result = await invoke(runtime)

    assert result.valid is True
    assert result.attempts == 2
    assert result.selected_max_tokens == 512
    assert result.recoveries == ("ProviderTransportError",)
    assert result.usage_complete is False
    assert [call.max_tokens for _name, call in runtime.calls] == [512, 512]


@pytest.mark.asyncio
async def test_nonretryable_provider_error_does_not_retry() -> None:
    runtime = FakeRuntime([ProviderHTTPError(400)])

    result = await invoke(runtime)

    assert result.valid is False
    assert result.failure == "expert call failed: ProviderHTTPError"
    assert result.failure_retryable is False
    assert len(runtime.calls) == 1


@pytest.mark.asyncio
async def test_invalid_json_at_reported_ceiling_expands_tier_not_transport_retry() -> None:
    runtime = FakeRuntime(
        [
            response("{", completion_tokens=512),
            response('{"action":"abstain"}', completion_tokens=12),
        ]
    )

    result = await invoke(runtime)

    assert result.valid is True
    assert result.action == "abstain"
    assert result.selected_max_tokens == 1024
    assert [call.max_tokens for _name, call in runtime.calls] == [512, 1024]


@pytest.mark.asyncio
async def test_semantic_failure_never_buys_a_larger_tier() -> None:
    runtime = FakeRuntime([response('{"action":"advise"}', completion_tokens=20)])

    result = await invoke(runtime)

    assert result.valid is False
    assert result.failure == "expert envelope invalid: schema_mismatch"
    assert [call.max_tokens for _name, call in runtime.calls] == [512]


@pytest.mark.asyncio
async def test_exhausted_tiers_are_bounded_and_usage_is_merged() -> None:
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
