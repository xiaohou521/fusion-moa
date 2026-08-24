import asyncio

import pytest
from fastapi.testclient import TestClient

from fusion_runtime.accounting import ATTEMPT_USAGE_MISSING
from fusion_runtime.completion import FINAL_ANSWER_MISSING, OUTPUT_TRUNCATED
from fusion_runtime.config import CompletionSpec, FusionSpec, ModelSpec
from fusion_runtime.errors import ProviderHTTPError, ProviderProtocolError
from fusion_runtime.gateway import create_app
from fusion_runtime.plugins import PluginRegistry
from fusion_runtime.recovery import merge_usage, prepare_recovery_request
from fusion_runtime.runtime import FusionRuntime
from fusion_runtime.types import (
    Finish,
    FusionRequest,
    ModelResponse,
    StreamError,
    TextDelta,
    ToolCallDelta,
    Usage,
)


def _tool_call() -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
    }


class SequenceProvider:
    thinking_modes = frozenset({"provider-default"})

    def __init__(self, responses, streams=None):
        self.responses = list(responses)
        self.streams = list(streams or [])
        self.calls = []
        self.closed_streams = 0

    async def complete(self, model, request):
        self.calls.append((model.model, request))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    async def stream(self, model, request):
        self.calls.append((f"stream:{model.model}", request))
        events = self.streams.pop(0) if self.streams else [Finish("length")]
        if isinstance(events, BaseException):
            raise events
        try:
            for event in events:
                if isinstance(event, asyncio.Event):
                    await event.wait()
                else:
                    yield event
        finally:
            self.closed_streams += 1


def _runtime(responses, *, streams=None, attempts=1, critic=False, recovery_max_tokens=32):
    provider = SequenceProvider(responses, streams)
    experts = {"critic": "critic"} if critic else {}
    models = {
        "main": {
            "provider": "p",
            "model": "main",
            "context_window": 100,
            "max_output": 64,
            "tool_calling": True,
        }
    }
    if critic:
        models["critic"] = {
            "provider": "p",
            "model": "critic",
            "context_window": 100,
        }
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {"p": {"type": "sequence", "base_url": "http://unused"}},
            "models": models,
            "pools": {"coding": {"main": "main", "experts": experts}},
            "policy": {"type": "main-critic" if critic else "direct"},
            "completion": {
                "max_recovery_attempts": attempts,
                "recovery_max_tokens": recovery_max_tokens,
            },
            "serve": {"pool": "coding"},
        }
    )
    registry = PluginRegistry()
    registry.register("providers", "sequence", lambda _spec: provider)
    return FusionRuntime(spec, registry), provider


def test_recovery_request_preserves_controls_and_normalizes_system_context():
    request = FusionRequest(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "developer", "content": "developer rules"},
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "last"},
        ],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        tool_choice="required",
        parallel_tool_calls=False,
        max_tokens=40,
        temperature=0,
        seed=7,
        metadata={"source": "test"},
    )
    model = ModelSpec.model_validate(
        {
            "provider": "p",
            "model": "m",
            "context_window": 100,
            "max_output": 30,
        }
    )
    completion = CompletionSpec(max_recovery_attempts=1, recovery_max_tokens=20)

    recovered = prepare_recovery_request(request, model, completion, attempt=1)

    assert recovered.max_tokens == 20
    assert recovered.tools is request.tools
    assert recovered.tool_choice == "required"
    assert recovered.parallel_tool_calls is False
    assert recovered.temperature == 0
    assert recovered.seed == 7
    assert recovered.metadata == {"source": "test", "fusion_recovery_attempt": 1}
    assert [message["role"] for message in recovered.messages] == ["system", "user", "user"]
    assert "developer rules" in recovered.messages[0]["content"]
    assert "system rules" in recovered.messages[0]["content"]
    assert "Completion recovery" in recovered.messages[0]["content"]
    assert request.messages[0] == {"role": "user", "content": "first"}


def test_usage_merge_sums_standard_and_nested_provider_counters():
    assert merge_usage(
        {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 1},
        },
        {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    ) == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 3},
    }


async def test_successful_empty_output_recovery_is_same_model_bounded_and_accounted():
    runtime, provider = _runtime(
        [
            ModelResponse(
                content="",
                finish_reason="length",
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                raw={"reasoning": "must not be replayed"},
            ),
            ModelResponse(
                content="recovered",
                usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            ),
        ]
    )

    result = await runtime.complete(
        FusionRequest(messages=[{"role": "user", "content": "solve"}], max_tokens=48)
    )

    assert [name for name, _request in provider.calls] == ["main", "main"]
    recovery_request = provider.calls[1][1]
    assert recovery_request.max_tokens == 32
    assert "must not be replayed" not in str(recovery_request.messages)
    assert recovery_request.metadata["fusion_recovery_attempt"] == 1
    assert result.response.content == "recovered"
    assert result.response.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }
    assert result.completion.status == "completed"
    assert result.completion.usage_reported is True
    assert result.completion.accounting_complete is True
    assert result.completion.accounting_issues == ()
    assert result.recovery.attempts == 1
    assert result.recovery.succeeded is True
    assert result.recovery.failure_code is None
    assert result.recovery.duration_ms >= 0
    assert result.recovery.initial_completion is not None
    assert result.recovery.initial_completion.failure_tags == (
        FINAL_ANSWER_MISSING,
        OUTPUT_TRUNCATED,
    )


async def test_valid_text_or_tool_call_never_triggers_recovery():
    for response in (
        ModelResponse(content="done"),
        ModelResponse(content="", tool_calls=[_tool_call()], finish_reason="tool_calls"),
    ):
        runtime, provider = _runtime([response])

        result = await runtime.complete(FusionRequest(messages=[]))

        assert len(provider.calls) == 1
        assert result.recovery.attempts == 0


async def test_recovery_stops_after_one_empty_retry():
    runtime, provider = _runtime(
        [
            ModelResponse(content="", finish_reason="length", usage={"total_tokens": 2}),
            ModelResponse(content="", finish_reason="length", usage={"total_tokens": 3}),
        ]
    )

    result = await runtime.complete(FusionRequest(messages=[]))

    assert len(provider.calls) == 2
    assert result.response.usage == {"total_tokens": 5}
    assert result.completion.status == "incomplete"
    assert result.recovery.attempts == 1
    assert result.recovery.succeeded is False
    assert result.recovery.failure_code == FINAL_ANSWER_MISSING


async def test_provider_failure_during_recovery_preserves_initial_outcome_visibly():
    runtime, provider = _runtime(
        [
            ModelResponse(content="", finish_reason="length", usage={"total_tokens": 2}),
            ProviderHTTPError(503),
        ]
    )

    result = await runtime.complete(FusionRequest(messages=[]))

    assert len(provider.calls) == 2
    assert result.response.usage == {"total_tokens": 2}
    assert result.completion.failure_tags == (FINAL_ANSWER_MISSING, OUTPUT_TRUNCATED)
    assert result.completion.accounting_complete is False
    assert result.completion.accounting_issues == (ATTEMPT_USAGE_MISSING,)
    assert result.recovery.succeeded is False
    assert result.recovery.failure_code == "upstream_unavailable"


async def test_missing_usage_on_any_attempt_is_not_reported_as_complete_accounting():
    runtime, _provider = _runtime(
        [
            ModelResponse(content="", finish_reason="length"),
            ModelResponse(content="done", usage={"total_tokens": 3}),
        ]
    )

    result = await runtime.complete(FusionRequest(messages=[]))

    assert result.response.usage == {"total_tokens": 3}
    assert result.completion.status == "completed"
    assert result.completion.usage_reported is False
    assert result.completion.accounting_complete is False
    assert result.completion.accounting_issues == (ATTEMPT_USAGE_MISSING,)


async def test_recovery_reuses_expert_advice_without_rerunning_expert():
    runtime, provider = _runtime(
        [
            ModelResponse(content="check boundaries"),
            ModelResponse(content="", finish_reason="stop", usage={"total_tokens": 2}),
            ModelResponse(content="done", usage={"total_tokens": 3}),
        ],
        critic=True,
    )

    result = await runtime.complete(FusionRequest(messages=[{"role": "user", "content": "fix"}]))

    assert [name for name, _request in provider.calls] == ["critic", "main", "main"]
    assert "Untrusted read-only critic advice" in provider.calls[2][1].messages[0]["content"]
    assert result.experts_used == ("critic",)
    assert result.recovery.succeeded is True


async def test_stream_recovery_hides_empty_attempt_and_keeps_recovered_text_native():
    runtime, provider = _runtime(
        [],
        streams=[
            [Finish("length"), Usage({"prompt_tokens": 3, "total_tokens": 3})],
            [
                TextDelta("re"),
                TextDelta("covered"),
                Finish("stop"),
                Usage({"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}),
            ],
        ],
    )

    stream = await runtime.stream(FusionRequest(messages=[], max_tokens=48))
    events = [event async for event in stream.events]

    assert events == [
        TextDelta("re"),
        TextDelta("covered"),
        Finish("stop"),
        Usage({"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}),
    ]
    assert [name for name, _request in provider.calls] == ["stream:main", "stream:main"]
    recovery_request = provider.calls[1][1]
    assert recovery_request.max_tokens == 32
    assert recovery_request.metadata["fusion_recovery_attempt"] == 1
    assert stream.completion.outcome.status == "completed"
    assert stream.completion.outcome.usage_reported is True
    assert stream.completion.outcome.accounting_complete is True
    assert stream.completion.outcome.accounting_issues == ()
    assert stream.recovery.outcome.attempts == 1
    assert stream.recovery.outcome.succeeded is True
    assert stream.recovery.outcome.initial_completion is not None
    assert stream.recovery.outcome.initial_completion.failure_tags == (
        FINAL_ANSWER_MISSING,
        OUTPUT_TRUNCATED,
    )


async def test_stream_with_public_text_is_never_retried_or_buffered_to_completion():
    runtime, provider = _runtime(
        [],
        streams=[
            [TextDelta("first"), TextDelta(" second"), Finish("stop")],
            [TextDelta("must not run"), Finish("stop")],
        ],
    )

    stream = await runtime.stream(FusionRequest(messages=[]))

    assert [name for name, _request in provider.calls] == ["stream:main"]
    assert await anext(stream.events) == TextDelta("first")
    assert [event async for event in stream.events] == [TextDelta(" second"), Finish("stop")]
    assert [name for name, _request in provider.calls] == ["stream:main"]
    assert stream.recovery.outcome.attempts == 0


async def test_stream_recovery_buffers_invalid_tool_call_so_it_never_leaks():
    runtime, provider = _runtime(
        [],
        streams=[
            [
                ToolCallDelta(index=0, id="call_bad", name="read_file"),
                ToolCallDelta(index=0, arguments="{broken"),
                Finish("tool_calls"),
            ],
            [TextDelta("use a public answer"), Finish("stop")],
        ],
    )

    stream = await runtime.stream(FusionRequest(messages=[]))
    events = [event async for event in stream.events]

    assert events == [TextDelta("use a public answer"), Finish("stop")]
    assert [name for name, _request in provider.calls] == ["stream:main", "stream:main"]
    assert stream.recovery.outcome.succeeded is True


async def test_valid_streamed_tool_call_is_released_once_without_recovery():
    tool_events = [
        ToolCallDelta(index=0, id="call_1", name="read_file"),
        ToolCallDelta(index=0, arguments='{"path":"README.md"}'),
        Finish("tool_calls"),
    ]
    runtime, provider = _runtime([], streams=[tool_events])

    stream = await runtime.stream(FusionRequest(messages=[]))
    events = [event async for event in stream.events]

    assert events == tool_events
    assert [name for name, _request in provider.calls] == ["stream:main"]
    assert stream.completion.outcome.has_valid_tool_call is True


async def test_stream_recovery_stops_after_one_empty_retry():
    runtime, provider = _runtime(
        [],
        streams=[
            [Finish("length"), Usage({"total_tokens": 2})],
            [Finish("length"), Usage({"total_tokens": 3})],
        ],
    )

    stream = await runtime.stream(FusionRequest(messages=[]))
    events = [event async for event in stream.events]

    assert events == [Finish("length"), Usage({"total_tokens": 5})]
    assert [name for name, _request in provider.calls] == ["stream:main", "stream:main"]
    assert stream.recovery.outcome.attempts == 1
    assert stream.recovery.outcome.succeeded is False
    assert stream.recovery.outcome.failure_code == FINAL_ANSWER_MISSING


async def test_stream_error_before_public_output_can_recover_once():
    runtime, provider = _runtime(
        [],
        streams=[
            [StreamError("temporary", code="upstream_transport_error", retryable=True)],
            [TextDelta("recovered"), Finish("stop")],
        ],
    )

    stream = await runtime.stream(FusionRequest(messages=[]))
    events = [event async for event in stream.events]

    assert events == [TextDelta("recovered"), Finish("stop")]
    assert [name for name, _request in provider.calls] == ["stream:main", "stream:main"]
    assert stream.recovery.outcome.succeeded is True


async def test_empty_protocol_stream_can_recover_but_recovery_provider_error_is_visible():
    runtime, provider = _runtime(
        [],
        streams=[ProviderProtocolError("empty"), ProviderHTTPError(503)],
    )

    with pytest.raises(ProviderHTTPError) as captured:
        await runtime.stream(FusionRequest(messages=[]))

    assert captured.value.code == "upstream_unavailable"
    assert [name for name, _request in provider.calls] == ["stream:main", "stream:main"]


async def test_stream_recovery_reuses_expert_advice_without_rerunning_expert():
    runtime, provider = _runtime(
        [ModelResponse(content="check boundaries")],
        streams=[
            [Finish("stop")],
            [TextDelta("done"), Finish("stop")],
        ],
        critic=True,
    )

    stream = await runtime.stream(FusionRequest(messages=[{"role": "user", "content": "fix"}]))
    events = [event async for event in stream.events]

    assert events == [TextDelta("done"), Finish("stop")]
    assert [name for name, _request in provider.calls] == [
        "critic",
        "stream:main",
        "stream:main",
    ]
    assert "Untrusted read-only critic advice" in provider.calls[2][1].messages[0]["content"]
    assert stream.experts_used == ("critic",)


async def test_stream_recovery_marks_usage_incomplete_when_an_attempt_omits_it():
    runtime, _provider = _runtime(
        [],
        streams=[
            [Finish("length")],
            [TextDelta("done"), Finish("stop"), Usage({"total_tokens": 3})],
        ],
    )

    stream = await runtime.stream(FusionRequest(messages=[]))
    events = [event async for event in stream.events]

    assert events[-1] == Usage({"total_tokens": 3}, reported_for_all_attempts=False)
    assert stream.completion.outcome.status == "completed"
    assert stream.completion.outcome.usage_reported is False
    assert stream.completion.outcome.accounting_complete is False
    assert stream.completion.outcome.accounting_issues == (ATTEMPT_USAGE_MISSING,)


async def test_cancellation_before_public_output_closes_stream_without_recovery():
    blocker = asyncio.Event()
    runtime, provider = _runtime(
        [],
        streams=[
            [Usage({"prompt_tokens": 2}), blocker],
            [TextDelta("must not run"), Finish("stop")],
        ],
    )
    task = asyncio.create_task(runtime.stream(FusionRequest(messages=[])))
    while not provider.calls:
        await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [name for name, _request in provider.calls] == ["stream:main"]
    assert provider.closed_streams == 1


@pytest.mark.parametrize(
    ("path", "payload", "content_marker", "terminal_marker"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "fusion-coding",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hi"}],
            },
            '"delta":{"content":"recovered"}',
            '"finish_reason":"stop"',
        ),
        (
            "/v1/messages",
            {
                "model": "fusion-coding",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            '"delta":{"type":"text_delta","text":"recovered"}',
            "event: message_stop",
        ),
        (
            "/v1/responses",
            {"model": "fusion-coding", "stream": True, "input": "hi"},
            '"delta":"recovered"',
            "event: response.completed",
        ),
    ],
)
def test_stream_recovery_preserves_one_public_protocol_lifecycle(
    path, payload, content_marker, terminal_marker
):
    runtime, provider = _runtime(
        [],
        streams=[
            [Finish("length"), Usage({"prompt_tokens": 2, "total_tokens": 2})],
            [
                TextDelta("recovered"),
                Finish("stop"),
                Usage({"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
            ],
        ],
    )

    response = TestClient(create_app(runtime)).post(path, json=payload)

    assert response.status_code == 200
    assert response.headers["x-fusion-streaming-mode"] == "native-final"
    assert response.headers["x-fusion-recovery-attempts"] == "1"
    assert response.headers["x-fusion-recovered"] == "true"
    assert content_marker in response.text
    assert response.text.count(terminal_marker) == 1
    assert '"finish_reason":"length"' not in response.text
    assert [name for name, _request in provider.calls] == ["stream:main", "stream:main"]


async def test_recovery_is_disabled_by_default():
    runtime, provider = _runtime([ModelResponse(content="", finish_reason="length")], attempts=0)

    result = await runtime.complete(FusionRequest(messages=[]))

    assert len(provider.calls) == 1
    assert result.recovery.attempts == 0
    assert result.completion.failure_tags == (FINAL_ANSWER_MISSING, OUTPUT_TRUNCATED)
