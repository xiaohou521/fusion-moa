import json

import httpx
import pytest

from fusion_runtime.config import ModelSpec, ProviderSpec
from fusion_runtime.errors import ProviderHTTPError, ProviderProtocolError
from fusion_runtime.providers import AnthropicCompatibleProvider, OpenAICompatibleProvider
from fusion_runtime.types import (
    Finish,
    FusionRequest,
    StreamError,
    TextDelta,
    ToolCallDelta,
    Usage,
)


def data_sse(payload):
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n"


def event_sse(name, payload):
    return f"event: {name}\n" + data_sse(payload)


def model(*, tool_calling=True):
    return ModelSpec.model_validate(
        {
            "provider": "test",
            "model": "test-model",
            "context_window": 1000,
            "max_output": 100,
            "tool_calling": tool_calling,
        }
    )


async def test_openai_provider_preserves_tools_and_tool_calls():
    captured = {}

    async def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "pwd", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://example.test/v1/",
    )
    provider = OpenAICompatibleProvider(
        ProviderSpec(type="openai-compatible", base_url="https://example.test/v1"),
        client,
    )
    response = await provider.complete(
        model(),
        FusionRequest(
            messages=[{"role": "user", "content": "where"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "pwd", "parameters": {"type": "object"}},
                }
            ],
            tool_choice="required",
            parallel_tool_calls=False,
            reasoning_effort="high",
            seed=7,
        ),
    )
    assert captured["tools"][0]["function"]["name"] == "pwd"
    assert captured["tool_choice"] == "required"
    assert captured["parallel_tool_calls"] is False
    assert captured["reasoning_effort"] == "high"
    assert captured["seed"] == 7
    assert response.tool_calls[0]["id"] == "call_1"
    await provider.aclose()


async def test_openai_provider_streams_native_text_tools_finish_and_usage():
    captured = {}

    async def handler(request):
        captured.update(json.loads(request.content))
        sse = "".join(
            (
                data_sse({"choices": [{"delta": {"content": "hel"}}]}),
                data_sse({"choices": [{"delta": {"content": "lo"}}]}),
                data_sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "read",
                                                "arguments": '{"p":',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                data_sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                ),
                data_sse(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 3,
                            "total_tokens": 5,
                        },
                    }
                ),
                data_sse("[DONE]"),
            )
        )
        return httpx.Response(200, content=sse)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test/v1/"
    )
    provider = OpenAICompatibleProvider(
        ProviderSpec(type="openai-compatible", base_url="https://example.test/v1"), client
    )
    events = [
        event
        async for event in provider.stream(
            model(), FusionRequest(messages=[{"role": "user", "content": "hi"}])
        )
    ]
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["hel", "lo"]
    tool_events = [event for event in events if isinstance(event, ToolCallDelta)]
    assert tool_events[0].id == "call_1"
    assert "".join(event.arguments for event in tool_events) == '{"p":1}'
    assert [event.reason for event in events if isinstance(event, Finish)] == ["tool_calls"]
    assert [event.usage["total_tokens"] for event in events if isinstance(event, Usage)] == [5]
    await provider.aclose()


async def test_anthropic_provider_translates_canonical_tool_round_trip():
    captured = {}

    async def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "running"},
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "shell",
                        "input": {"command": "pwd"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://anthropic.test/v1/",
    )
    provider = AnthropicCompatibleProvider(
        ProviderSpec(type="anthropic-compatible", base_url="https://anthropic.test/v1"),
        client,
    )
    response = await provider.complete(
        model(),
        FusionRequest(
            messages=[
                {"role": "system", "content": "safe"},
                {
                    "role": "assistant",
                    "content": "calling",
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_1", "content": "ok"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "description": "run command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "shell"}},
            parallel_tool_calls=False,
        ),
    )
    assert captured["system"] == "safe"
    assert captured["messages"][0]["content"][1]["type"] == "tool_use"
    assert captured["messages"][1]["content"][0]["type"] == "tool_result"
    assert captured["tools"][0]["input_schema"] == {"type": "object"}
    assert captured["tool_choice"] == {
        "type": "tool",
        "name": "shell",
        "disable_parallel_tool_use": True,
    }
    assert response.tool_calls[0]["function"]["arguments"] == '{"command":"pwd"}'
    await provider.aclose()


async def test_anthropic_provider_streams_native_events():
    async def handler(_request):
        sse = "".join(
            (
                event_sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {"usage": {"input_tokens": 4}},
                    },
                ),
                event_sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                event_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "ok"},
                    },
                ),
                event_sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 1},
                    },
                ),
                event_sse("message_stop", {"type": "message_stop"}),
            )
        )
        return httpx.Response(200, content=sse)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://anthropic.test/v1/"
    )
    provider = AnthropicCompatibleProvider(
        ProviderSpec(type="anthropic-compatible", base_url="https://anthropic.test/v1"), client
    )
    events = [
        event
        async for event in provider.stream(
            model(), FusionRequest(messages=[{"role": "user", "content": "hi"}])
        )
    ]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["ok"]
    assert [event.reason for event in events if isinstance(event, Finish)] == ["end_turn"]
    assert [event.usage for event in events if isinstance(event, Usage)] == [
        {"input_tokens": 4},
        {"output_tokens": 1},
    ]
    await provider.aclose()


async def test_openai_stream_http_error_is_typed_retryable_and_secret_safe():
    async def handler(_request):
        return httpx.Response(429, text='{"error":{"message":"token sk-secret"}}')

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test/v1/"
    )
    provider = OpenAICompatibleProvider(
        ProviderSpec(type="openai-compatible", base_url="https://example.test/v1"), client
    )

    with pytest.raises(ProviderHTTPError) as captured:
        await anext(
            provider.stream(model(), FusionRequest(messages=[{"role": "user", "content": "hi"}]))
        )

    assert captured.value.status_code == 429
    assert captured.value.code == "upstream_rate_limited"
    assert captured.value.retryable is True
    assert "sk-secret" not in str(captured.value)
    await provider.aclose()


async def test_openai_initial_malformed_stream_event_is_protocol_error():
    async def handler(_request):
        return httpx.Response(200, content="data: not-json\n\n")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test/v1/"
    )
    provider = OpenAICompatibleProvider(
        ProviderSpec(type="openai-compatible", base_url="https://example.test/v1"), client
    )

    with pytest.raises(ProviderProtocolError):
        await anext(
            provider.stream(model(), FusionRequest(messages=[{"role": "user", "content": "hi"}]))
        )
    await provider.aclose()


async def test_openai_midstream_protocol_failure_becomes_terminal_event():
    async def handler(_request):
        return httpx.Response(
            200,
            content=data_sse({"choices": [{"delta": {"content": "partial"}}]})
            + "data: not-json\n\n",
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test/v1/"
    )
    provider = OpenAICompatibleProvider(
        ProviderSpec(type="openai-compatible", base_url="https://example.test/v1"), client
    )
    events = [
        event
        async for event in provider.stream(
            model(), FusionRequest(messages=[{"role": "user", "content": "hi"}])
        )
    ]

    assert events == [
        TextDelta("partial"),
        StreamError(
            "upstream provider emitted malformed stream JSON",
            code="provider_protocol_error",
        ),
    ]
    await provider.aclose()


async def test_openai_stream_without_finish_becomes_terminal_protocol_error():
    async def handler(_request):
        return httpx.Response(
            200,
            content=data_sse({"choices": [{"delta": {"content": "partial"}}]}) + data_sse("[DONE]"),
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.test/v1/"
    )
    provider = OpenAICompatibleProvider(
        ProviderSpec(type="openai-compatible", base_url="https://example.test/v1"), client
    )
    events = [
        event
        async for event in provider.stream(
            model(), FusionRequest(messages=[{"role": "user", "content": "hi"}])
        )
    ]

    assert events == [
        TextDelta("partial"),
        StreamError(
            "upstream provider ended without a finish event",
            code="provider_protocol_error",
        ),
    ]
    await provider.aclose()


@pytest.mark.parametrize(
    ("provider_type", "base_url", "body"),
    [
        (
            "openai-compatible",
            "https://example.test/v1",
            data_sse({"error": {"message": "do not expose me", "code": "overloaded"}}),
        ),
        (
            "anthropic-compatible",
            "https://anthropic.test/v1",
            event_sse(
                "error",
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "do not expose me"},
                },
            ),
        ),
    ],
)
async def test_provider_body_error_becomes_secret_safe_terminal_event(
    provider_type, base_url, body
):
    async def handler(_request):
        return httpx.Response(200, content=body)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=base_url.rstrip("/") + "/"
    )
    provider_class = (
        OpenAICompatibleProvider
        if provider_type == "openai-compatible"
        else AnthropicCompatibleProvider
    )
    provider = provider_class(ProviderSpec(type=provider_type, base_url=base_url), client)
    events = [
        event
        async for event in provider.stream(
            model(), FusionRequest(messages=[{"role": "user", "content": "hi"}])
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].message == "upstream provider stream failed"
    assert "do not expose me" not in events[0].message
    await provider.aclose()
