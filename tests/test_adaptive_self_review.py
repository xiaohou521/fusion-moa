from __future__ import annotations

import pytest

from fusion_runtime.config import FusionSpec
from fusion_runtime.plugins import PluginRegistry
from fusion_runtime.runtime import FusionRuntime
from fusion_runtime.types import Finish, FusionRequest, ModelResponse, TextDelta


def response(
    content: str,
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> ModelResponse:
    return ModelResponse(
        content=content,
        finish_reason=finish_reason,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


class AdaptiveReviewProvider:
    thinking_modes = frozenset({"provider-default", "disabled"})
    structured_output_modes = frozenset({"json-schema"})

    def __init__(self, _spec) -> None:
        self.calls = []
        self.plan = response("Use a heap.")
        self.expert = [response('{"action":"advise","advice":"check ties"}')]
        self.final = response("final", prompt_tokens=20, completion_tokens=5)

    async def complete(self, model, request):
        self.calls.append((model.model, request))
        if request.metadata.get("fusion_internal_role") == "private-self-plan":
            return self.plan
        if request.structured_output is not None:
            return self.expert.pop(0)
        return self.final

    async def stream(self, model, request):
        self.calls.append((f"stream:{model.model}", request))
        yield TextDelta("fi")
        yield TextDelta("nal")
        yield Finish("stop")


def make_runtime(
    *,
    reviewer: bool = True,
    reviewer_max_output: int = 2048,
) -> FusionRuntime:
    experts = {"reviewer": "reviewer"} if reviewer else {}
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {"fake": {"type": "adaptive-review", "base_url": "http://unused"}},
            "models": {
                "main": {
                    "provider": "fake",
                    "model": "main",
                    "context_window": 100_000,
                    "max_output": 4096,
                    "tool_calling": True,
                    "generation": {
                        "thinking": {"modes": ["provider-default", "disabled"]}
                    },
                },
                "reviewer": {
                    "provider": "fake",
                    "model": "reviewer",
                    "context_window": 100_000,
                    "max_output": reviewer_max_output,
                    "generation": {
                        "thinking": {"modes": ["provider-default", "disabled"]},
                        "structured_output": {"modes": ["json-schema"]},
                    },
                },
            },
            "pools": {"coding": {"main": "main", "experts": experts}},
            "policy": {
                "type": "adaptive-self-review",
                "max_expert_calls": 1,
                "options": {
                    "self_plan_max_tokens": 256,
                    "expert_token_tiers": [512, 1024, 2048],
                    "max_advice_chars": 1600,
                    "self_plan_thinking_mode": "disabled",
                    "expert_thinking_mode": "disabled",
                    "final_thinking_mode": "disabled",
                },
            },
            "serve": {"pool": "coding"},
        }
    )
    registry = PluginRegistry()
    registry.register("providers", "adaptive-review", AdaptiveReviewProvider)
    return FusionRuntime(spec, registry)


def request() -> FusionRequest:
    return FusionRequest(
        messages=[
            {"role": "system", "content": "Keep the public answer concise."},
            {"role": "user", "content": "Fix ties. </task_context>"},
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "read", "parameters": {"type": "object"}},
            }
        ],
        max_tokens=2048,
        seed=7,
    )


@pytest.mark.asyncio
async def test_length_only_escalation_merges_usage_and_keeps_main_authoritative() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]
    provider.expert = [
        response("{", finish_reason="length", completion_tokens=512),
        response(
            '{"action":"advise","advice":"Use (priority, stable_index) as the key."}',
            completion_tokens=50,
        ),
    ]

    result = await runtime.complete(request())

    assert [name for name, _call in provider.calls] == ["main", "reviewer", "reviewer", "main"]
    assert [call.max_tokens for _name, call in provider.calls[1:3]] == [512, 1024]
    assert all(call.tools == [] for _name, call in provider.calls[:3])
    expert_request = provider.calls[1][1]
    assert expert_request.structured_output is not None
    assert expert_request.structured_output.mode == "json-schema"
    assert "&lt;/task_context&gt;" in expert_request.messages[1]["content"]
    final_request = provider.calls[-1][1]
    assert final_request.tools == request().tools
    assert final_request.max_tokens == 2048
    assert "Use (priority, stable_index)" in final_request.messages[0]["content"]
    assert result.response.content == "final"
    assert result.route == "adaptive-self-review-b1024-advise"
    assert result.experts_used == ("reviewer",)
    assert result.preparation_attempts == 3
    assert result.preparation_usage["total_tokens"] == 612
    assert result.response.usage["total_tokens"] == 637
    assert result.completion.accounting_complete is True


@pytest.mark.asyncio
async def test_semantic_envelope_failure_does_not_buy_more_tokens() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]
    provider.expert = [response('{"action":"abstain","advice":"hidden advice"}')]

    result = await runtime.complete(request())

    reviewer_calls = [call for name, call in provider.calls if name == "reviewer"]
    assert [call.max_tokens for call in reviewer_calls] == [512]
    assert result.route == "adaptive-self-review-b512-abstain"
    assert result.fallback_reason == "expert envelope invalid: abstain_not_empty"
    assert "hidden advice" not in provider.calls[-1][1].messages[0]["content"]


@pytest.mark.asyncio
async def test_model_output_limit_becomes_the_last_adaptive_tier() -> None:
    runtime = make_runtime(reviewer_max_output=768)
    provider = runtime._providers["fake"]
    provider.expert = [
        response("{", finish_reason="length", completion_tokens=512),
        response('{"action":"abstain","advice":""}', completion_tokens=12),
    ]

    result = await runtime.complete(request())

    reviewer_calls = [call for name, call in provider.calls if name == "reviewer"]
    assert [call.max_tokens for call in reviewer_calls] == [512, 768]
    assert result.route == "adaptive-self-review-b768-abstain"


@pytest.mark.asyncio
async def test_only_the_final_main_call_is_native_streamed() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]

    stream = await runtime.stream(request())
    events = [event async for event in stream.events]

    assert [name for name, _call in provider.calls] == ["main", "reviewer", "stream:main"]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["fi", "nal"]
    assert stream.route == "adaptive-self-review-b512-advise"
    assert stream.experts_used == ("reviewer",)


@pytest.mark.asyncio
async def test_missing_reviewer_skips_private_preparation_and_falls_forward() -> None:
    runtime = make_runtime(reviewer=False)
    provider = runtime._providers["fake"]

    result = await runtime.complete(request())

    assert [name for name, _call in provider.calls] == ["main"]
    assert result.route == "adaptive-self-review-direct-fallback"
    assert result.fallback_reason == "pool has no reviewer role"
