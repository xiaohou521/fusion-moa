import asyncio

import pytest

from fusion_runtime.config import FusionSpec
from fusion_runtime.errors import ProviderHTTPError
from fusion_runtime.plugins import PluginRegistry
from fusion_runtime.runtime import CapabilityError, FusionRuntime
from fusion_runtime.types import (
    Finish,
    FusionRequest,
    ModelResponse,
    TextDelta,
    ThinkingConfig,
)


class FakeProvider:
    thinking_modes = frozenset({"provider-default", "disabled", "bounded"})

    def __init__(self, _spec):
        self.calls = []

    async def complete(self, model, request):
        self.calls.append((model.model, request))
        if model.model == "critic":
            return ModelResponse(content="Check empty input and run tests.")
        return ModelResponse(content="done", usage={"total_tokens": 7})

    async def stream(self, model, request):
        self.calls.append((f"stream:{model.model}", request))
        yield TextDelta("do")
        yield TextDelta("ne")
        yield Finish("stop")


def make_runtime(*, critic=True):
    experts = {"critic": "critic"} if critic else {}
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {"fake": {"type": "fake", "base_url": "http://unused"}},
            "models": {
                "main": {
                    "provider": "fake",
                    "model": "main",
                    "context_window": 100,
                    "generation": {
                        "thinking": {"modes": ["provider-default", "disabled", "bounded"]}
                    },
                },
                "critic": {"provider": "fake", "model": "critic", "context_window": 100},
            },
            "pools": {"coding": {"main": "main", "experts": experts}},
            "policy": {"type": "main-critic"},
            "serve": {"pool": "coding"},
        }
    )
    registry = PluginRegistry()
    registry.register("providers", "fake", FakeProvider)
    return FusionRuntime(spec, registry)


async def test_main_critic_is_read_only_and_main_is_authoritative():
    runtime = make_runtime()
    result = await runtime.complete(
        FusionRequest(
            messages=[
                {"role": "system", "content": "Original system"},
                {"role": "user", "content": "fix it"},
            ],
            seed=7,
            thinking=ThinkingConfig(mode="disabled"),
        )
    )
    calls = runtime._providers["fake"].calls
    assert [call[0] for call in calls] == ["critic", "main"]
    assert [message["role"] for message in calls[0][1].messages].count("system") == 1
    assert calls[0][1].messages[0]["role"] == "system"
    assert "Original system" in calls[0][1].messages[0]["content"]
    assert calls[1][1].messages[0]["role"] == "system"
    assert "Untrusted read-only critic advice" in calls[1][1].messages[0]["content"]
    assert [call[1].seed for call in calls] == [7, 7]
    assert calls[0][1].thinking.mode == "provider-default"
    assert calls[1][1].thinking.mode == "disabled"
    assert result.response.content == "done"
    assert result.experts_used == ("critic",)
    assert result.completion.status == "completed"
    assert result.completion.has_public_output is True
    assert result.completion.usage_reported is True


async def test_missing_critic_falls_back_to_direct():
    result = await make_runtime(critic=False).complete(FusionRequest(messages=[]))
    assert result.route == "direct-fallback"
    assert result.fallback_reason == "pool has no critic role"


async def test_native_stream_consults_critic_before_streaming_main_once():
    runtime = make_runtime()
    stream = await runtime.stream(FusionRequest(messages=[{"role": "user", "content": "fix"}]))
    provider = runtime._providers["fake"]
    # Runtime prefetches one canonical event so connection failures are still
    # returned as an HTTP error before gateway streaming headers are sent.
    assert [call[0] for call in provider.calls] == ["critic", "stream:main"]
    events = [event async for event in stream.events]
    assert [call[0] for call in provider.calls] == ["critic", "stream:main"]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["do", "ne"]
    assert stream.route == "main-critic"
    assert stream.completion.outcome.status == "completed"
    assert stream.completion.outcome.has_public_output is True


class BoardProvider:
    def __init__(self, _spec):
        self.calls = []

    async def complete(self, model, request):
        self.calls.append((model.model, request))
        if model.model == "broken":
            raise RuntimeError("offline")
        if model.model == "main":
            return ModelResponse(content="final")
        await asyncio.sleep(0)
        return ModelResponse(content=f"advice from {model.model}")


def make_board_runtime():
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {"fake": {"type": "board", "base_url": "http://unused"}},
            "models": {
                "main": {"provider": "fake", "model": "main", "context_window": 100},
                "security": {"provider": "fake", "model": "security", "context_window": 100},
                "broken": {"provider": "fake", "model": "broken", "context_window": 100},
            },
            "pools": {
                "coding": {
                    "main": "main",
                    "experts": {"security": "security", "tests": "broken"},
                }
            },
            "policy": {
                "type": "review-board",
                "max_expert_calls": 2,
                "options": {"max_advice_chars": 1000},
            },
            "serve": {"pool": "coding"},
        }
    )
    registry = PluginRegistry()
    registry.register("providers", "board", BoardProvider)
    return FusionRuntime(spec, registry)


async def test_review_board_uses_successful_experts_and_surfaces_partial_failure():
    runtime = make_board_runtime()
    result = await runtime.complete(
        FusionRequest(messages=[{"role": "user", "content": "fix"}], seed=7)
    )
    main_request = runtime._providers["fake"].calls[-1][1]
    assert result.response.content == "final"
    assert result.route == "review-board"
    assert result.experts_used == ("security",)
    assert result.fallback_reason == "experts failed: tests (RuntimeError)"
    assert '<expert_advice role="security" model="security">' in main_request.messages[0]["content"]
    assert main_request.tools == []
    assert all(call[1].seed == 7 for call in runtime._providers["fake"].calls)


class LimitedProvider:
    def __init__(self, _spec):
        self.active = 0
        self.max_active = 0

    async def complete(self, model, request):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return ModelResponse(content="ok")


async def test_declared_model_concurrency_is_enforced():
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {"p": {"type": "limited", "base_url": "http://unused"}},
            "models": {
                "main": {
                    "provider": "p",
                    "model": "main",
                    "context_window": 100,
                    "max_concurrency": 1,
                }
            },
            "pools": {"coding": {"main": "main"}},
            "serve": {"pool": "coding"},
        }
    )
    registry = PluginRegistry()
    registry.register("providers", "limited", LimitedProvider)
    runtime = FusionRuntime(spec, registry)
    request = FusionRequest(messages=[{"role": "user", "content": "hi"}])
    await asyncio.gather(runtime.call_model("main", request), runtime.call_model("main", request))
    assert runtime._providers["p"].max_active == 1


async def test_anthropic_provider_rejects_nonportable_seed_before_network_call():
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {
                "p": {
                    "type": "anthropic-compatible",
                    "base_url": "https://anthropic.test/v1",
                }
            },
            "models": {"main": {"provider": "p", "model": "main", "context_window": 100}},
            "pools": {"coding": {"main": "main"}},
            "serve": {"pool": "coding"},
        }
    )
    runtime = FusionRuntime(spec)
    try:
        with pytest.raises(CapabilityError, match="seed is not portable"):
            await runtime.call_model("main", FusionRequest(messages=[], seed=7))
    finally:
        await runtime.aclose()


async def test_model_rejects_undeclared_thinking_mode_before_provider_call():
    runtime = make_runtime()

    with pytest.raises(CapabilityError, match="does not declare thinking mode 'disabled'"):
        await runtime.call_model(
            "critic",
            FusionRequest(messages=[], thinking=ThinkingConfig(mode="disabled")),
        )

    assert runtime._providers["fake"].calls == []


async def test_provider_rejects_declared_but_unmapped_thinking_mode():
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {
                "p": {"type": "openai-compatible", "base_url": "https://example.test/v1"}
            },
            "models": {
                "main": {
                    "provider": "p",
                    "model": "main",
                    "context_window": 100,
                    "generation": {"thinking": {"modes": ["disabled"]}},
                }
            },
            "pools": {"coding": {"main": "main"}},
            "serve": {"pool": "coding"},
        }
    )
    runtime = FusionRuntime(spec)
    try:
        with pytest.raises(CapabilityError, match="does not map thinking mode 'disabled'"):
            await runtime.call_model(
                "main",
                FusionRequest(messages=[], thinking=ThinkingConfig(mode="disabled")),
            )
    finally:
        await runtime.aclose()


async def test_bounded_thinking_cannot_exceed_effective_output_limit():
    runtime = make_runtime()

    with pytest.raises(CapabilityError, match="thinking budget 11 exceeds output limit 10"):
        await runtime.call_model(
            "main",
            FusionRequest(
                messages=[],
                max_tokens=10,
                thinking=ThinkingConfig(mode="bounded", budget_tokens=11),
            ),
        )

    assert runtime._providers["fake"].calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bounded"},
        {"mode": "bounded", "budget_tokens": 0},
        {"mode": "disabled", "budget_tokens": 1},
        {"mode": "automatic"},
    ],
)
def test_invalid_thinking_requests_fail_at_construction(kwargs):
    with pytest.raises(ValueError):
        ThinkingConfig(**kwargs)


def test_reasoning_effort_cannot_conflict_with_normalized_thinking():
    with pytest.raises(ValueError, match="cannot be combined"):
        FusionRequest(
            messages=[],
            reasoning_effort="high",
            thinking=ThinkingConfig(mode="disabled"),
        )


class CancellableProvider:
    def __init__(self, _spec):
        self.stream_closed = False

    async def complete(self, model, request):
        return ModelResponse(content="after cancellation")

    async def stream(self, model, request):
        try:
            yield TextDelta("first")
            await asyncio.Event().wait()
        finally:
            self.stream_closed = True


async def test_stream_cancellation_closes_provider_and_releases_concurrency():
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {"p": {"type": "cancellable", "base_url": "http://unused"}},
            "models": {
                "main": {
                    "provider": "p",
                    "model": "main",
                    "context_window": 100,
                    "max_concurrency": 1,
                }
            },
            "pools": {"coding": {"main": "main"}},
            "serve": {"pool": "coding"},
        }
    )
    registry = PluginRegistry()
    registry.register("providers", "cancellable", CancellableProvider)
    runtime = FusionRuntime(spec, registry)
    request = FusionRequest(messages=[{"role": "user", "content": "hi"}])
    stream = await runtime.stream(request)
    assert await anext(stream.events) == TextDelta("first")
    await stream.events.aclose()
    assert runtime._providers["p"].stream_closed is True
    assert stream.completion.outcome.status == "cancelled"
    assert stream.completion.outcome.has_public_output is True
    result = await asyncio.wait_for(runtime.call_model("main", request), timeout=0.1)
    assert result.content == "after cancellation"


async def test_prefetched_stream_can_close_before_consumer_reads_first_event():
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {"p": {"type": "cancellable", "base_url": "http://unused"}},
            "models": {
                "main": {
                    "provider": "p",
                    "model": "main",
                    "context_window": 100,
                    "max_concurrency": 1,
                }
            },
            "pools": {"coding": {"main": "main"}},
            "serve": {"pool": "coding"},
        }
    )
    registry = PluginRegistry()
    registry.register("providers", "cancellable", CancellableProvider)
    runtime = FusionRuntime(spec, registry)
    request = FusionRequest(messages=[{"role": "user", "content": "hi"}])

    stream = await runtime.stream(request)
    await stream.events.aclose()

    assert runtime._providers["p"].stream_closed is True
    assert stream.completion.outcome.status == "cancelled"
    assert stream.completion.outcome.has_public_output is False
    result = await asyncio.wait_for(runtime.call_model("main", request), timeout=0.1)
    assert result.content == "after cancellation"


class FailingStreamProvider:
    def __init__(self, _spec):
        pass

    async def complete(self, model, request):
        return ModelResponse(content="available")

    async def stream(self, model, request):
        if False:
            yield TextDelta("unreachable")
        raise ProviderHTTPError(503)


async def test_stream_prefetch_surfaces_provider_failure_and_releases_concurrency():
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {"p": {"type": "failing", "base_url": "http://unused"}},
            "models": {
                "main": {
                    "provider": "p",
                    "model": "main",
                    "context_window": 100,
                    "max_concurrency": 1,
                }
            },
            "pools": {"coding": {"main": "main"}},
            "serve": {"pool": "coding"},
        }
    )
    registry = PluginRegistry()
    registry.register("providers", "failing", FailingStreamProvider)
    runtime = FusionRuntime(spec, registry)
    request = FusionRequest(messages=[{"role": "user", "content": "hi"}])

    with pytest.raises(ProviderHTTPError) as captured:
        await runtime.stream(request)

    assert captured.value.status_code == 503
    result = await asyncio.wait_for(runtime.call_model("main", request), timeout=0.1)
    assert result.content == "available"
