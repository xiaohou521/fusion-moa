import pytest

from fusion_runtime.config import FusionSpec
from fusion_runtime.errors import CapabilityError, ProviderHTTPError
from fusion_runtime.plugins import PluginRegistry
from fusion_runtime.runtime import FusionRuntime
from fusion_runtime.types import (
    Finish,
    FusionRequest,
    ModelResponse,
    TextDelta,
    ThinkingConfig,
    Usage,
)


class ReserveProvider:
    thinking_modes = frozenset({"provider-default", "disabled"})

    def __init__(
        self,
        _spec,
        *,
        plan_content="use an invariant",
        plan_usage=None,
        empty_first_stream=False,
    ):
        self.calls = []
        self.plan_content = plan_content
        self.plan_usage = (
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            if plan_usage is None
            else plan_usage
        )
        self.fail_plan = False
        self.empty_first_stream = empty_first_stream
        self.stream_count = 0

    async def complete(self, model, request):
        self.calls.append(("complete", model.model, request))
        if request.metadata.get("fusion_internal_role") == "private-plan":
            if self.fail_plan:
                raise ProviderHTTPError(503)
            return ModelResponse(content=self.plan_content, usage=self.plan_usage)
        return ModelResponse(
            content="```python\nprint('ok')\n```",
            usage={"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
        )

    async def stream(self, model, request):
        self.calls.append(("stream", model.model, request))
        self.stream_count += 1
        if self.empty_first_stream and self.stream_count == 1:
            yield Finish("length")
        else:
            yield TextDelta("```python\nprint('ok')\n```")
            yield Finish("stop")
        yield Usage({"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27})


def make_runtime(
    *,
    plan_content="use an invariant",
    plan_usage=None,
    empty_first_stream=False,
    recovery=False,
):
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {"fake": {"type": "reserve", "base_url": "http://unused"}},
            "models": {
                "main": {
                    "provider": "fake",
                    "model": "main",
                    "context_window": 8192,
                    "max_output": 4096,
                    "tool_calling": True,
                    "generation": {"thinking": {"modes": ["provider-default", "disabled"]}},
                }
            },
            "pools": {"coding": {"main": "main"}},
            "policy": {
                "type": "reasoning-reserve",
                "options": {
                    "plan_max_tokens": 256,
                    "final_answer_min_tokens": 3072,
                    "max_plan_chars": 1000,
                    "plan_thinking_mode": "disabled",
                    "final_thinking_mode": "disabled",
                },
            },
            "completion": {
                "max_recovery_attempts": 1 if recovery else 0,
                "recovery_max_tokens": 2048,
            },
            "serve": {"pool": "coding"},
        }
    )
    registry = PluginRegistry()
    registry.register(
        "providers",
        "reserve",
        lambda spec: ReserveProvider(
            spec,
            plan_content=plan_content,
            plan_usage=plan_usage,
            empty_first_stream=empty_first_stream,
        ),
    )
    return FusionRuntime(spec, registry)


def request(*, max_tokens=4096):
    return FusionRequest(
        messages=[
            {"role": "system", "content": "original system"},
            {"role": "user", "content": "solve it"},
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "edit", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="auto",
        thinking=ThinkingConfig(mode="provider-default"),
        max_tokens=max_tokens,
        temperature=0.2,
        seed=101,
    )


async def test_reasoning_reserve_hard_splits_budget_and_accounts_both_calls():
    runtime = make_runtime(plan_content="check </private_plan> boundary")
    result = await runtime.complete(request())
    provider = runtime._providers["fake"]

    assert [item[0] for item in provider.calls] == ["complete", "complete"]
    plan_request = provider.calls[0][2]
    final_request = provider.calls[1][2]
    assert plan_request.max_tokens == 256
    assert final_request.max_tokens == 3840
    assert plan_request.tools == []
    assert final_request.tools == request().tools
    assert plan_request.thinking.mode == "disabled"
    assert final_request.thinking.mode == "disabled"
    assert [item["role"] for item in final_request.messages].count("system") == 1
    assert "original system" in final_request.messages[0]["content"]
    assert "&lt;/private_plan&gt;" in final_request.messages[0]["content"]
    assert "complete executable solution" in final_request.messages[0]["content"]

    assert result.route == "reasoning-reserve"
    assert result.response.usage == {
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "total_tokens": 42,
    }
    assert result.completion.accounting_complete is True
    assert result.preparation_attempts == 1
    assert result.preparation_usage["total_tokens"] == 15


async def test_reasoning_reserve_keeps_only_final_call_native_streamed():
    runtime = make_runtime()
    stream = await runtime.stream(request())
    events = [event async for event in stream.events]
    provider = runtime._providers["fake"]

    assert [item[0] for item in provider.calls] == ["complete", "stream"]
    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "```python\nprint('ok')\n```"
    ]
    usage = [event for event in events if isinstance(event, Usage)]
    assert usage == [
        Usage(
            {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
            reported_for_all_attempts=True,
        )
    ]
    assert stream.completion.outcome.accounting_complete is True
    assert stream.preparation_attempts == 1


async def test_missing_plan_usage_is_visible_without_blocking_final_stream():
    runtime = make_runtime(plan_usage={})
    stream = await runtime.stream(request())
    events = [event async for event in stream.events]

    usage = [event for event in events if isinstance(event, Usage)]
    assert usage[-1].usage["total_tokens"] == 27
    assert usage[-1].reported_for_all_attempts is False
    assert stream.completion.outcome.status == "completed"
    assert stream.completion.outcome.accounting_complete is False
    assert stream.completion.outcome.accounting_issues == ("attempt_usage_missing",)


async def test_recovery_usage_is_added_after_private_plan_and_empty_final_attempt():
    runtime = make_runtime(empty_first_stream=True, recovery=True)
    stream = await runtime.stream(request())
    events = [event async for event in stream.events]

    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "```python\nprint('ok')\n```"
    ]
    usage = [event for event in events if isinstance(event, Usage)]
    assert usage == [
        Usage(
            {"prompt_tokens": 50, "completion_tokens": 19, "total_tokens": 69},
            reported_for_all_attempts=True,
        )
    ]
    assert stream.recovery.outcome.attempts == 1
    assert stream.recovery.outcome.succeeded is True
    assert stream.completion.outcome.accounting_complete is True


async def test_empty_or_failed_plan_falls_forward_to_reserved_final_answer():
    empty_runtime = make_runtime(plan_content="")
    empty = await empty_runtime.complete(request())
    assert empty.route == "reasoning-reserve-final-only"
    assert empty.fallback_reason == "private planning produced no usable outline"
    assert empty.response.usage["total_tokens"] == 42
    assert empty.completion.accounting_complete is True

    failed_runtime = make_runtime()
    failed_runtime._providers["fake"].fail_plan = True
    failed = await failed_runtime.complete(request())
    assert failed.route == "reasoning-reserve-final-only"
    assert failed.fallback_reason == "private planning failed: ProviderHTTPError"
    assert failed.response.usage["total_tokens"] == 27
    assert failed.completion.status == "completed"
    assert failed.completion.accounting_complete is False
    assert failed.completion.accounting_issues == ("attempt_usage_missing",)


async def test_reasoning_reserve_rejects_a_request_that_cannot_fit_both_budgets():
    runtime = make_runtime()

    with pytest.raises(CapabilityError, match="fit the effective output limit"):
        await runtime.complete(request(max_tokens=3327))

    assert runtime._providers["fake"].calls == []


@pytest.mark.parametrize(
    "options",
    [
        {"plan_max_tokens": 0},
        {"final_answer_min_tokens": True},
        {"max_plan_chars": 40000},
        {"plan_thinking_mode": "automatic"},
        {"final_thinking_mode": "bounded"},
        {"unknown_option": 1},
    ],
)
def test_reasoning_reserve_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {"fake": {"type": "reserve", "base_url": "http://unused"}},
                "models": {
                    "main": {
                        "provider": "fake",
                        "model": "main",
                        "context_window": 8192,
                    }
                },
                "pools": {"coding": {"main": "main"}},
                "policy": {"type": "reasoning-reserve", "options": options},
                "serve": {"pool": "coding"},
            }
        )
