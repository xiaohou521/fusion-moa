import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import fusion_runtime.gateway as gateway
from fusion_runtime.config import FusionSpec
from fusion_runtime.errors import ProviderHTTPError
from fusion_runtime.gateway import create_app
from fusion_runtime.types import (
    Finish,
    FusionStream,
    StreamError,
    TextDelta,
    ToolCallDelta,
    Usage,
)


class EventRuntime:
    def __init__(self, events):
        self.events = events
        self.spec = FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {"p": {"type": "openai-compatible", "base_url": "http://unused"}},
                "models": {"m": {"provider": "p", "model": "m", "context_window": 10}},
                "pools": {"coding": {"main": "m"}},
                "serve": {
                    "pool": "coding",
                    "model_name": "fusion-coding",
                    "protocols": ["openai-chat", "openai-responses", "anthropic-messages"],
                },
            }
        )

    async def stream(self, request):
        async def canonical_events():
            for event in self.events:
                yield event

        return FusionStream(events=canonical_events(), route="direct", trace_id="trace123")


def _sse_events(body):
    parsed = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(
            (line.removeprefix("event: ") for line in lines if line.startswith("event: ")),
            "data",
        )
        raw = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        parsed.append((event, raw if raw == "[DONE]" else json.loads(raw)))
    return parsed


@pytest.fixture
def canonical_events():
    return [
        TextDelta("he"),
        TextDelta("llo"),
        ToolCallDelta(index=0, id="call_1", name="read_file"),
        ToolCallDelta(index=0, arguments='{"path":"README.md"}'),
        Finish("tool_calls"),
        Usage({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}),
    ]


def test_openai_chat_frozen_stream_mapping(canonical_events, monkeypatch):
    monkeypatch.setattr(gateway.time, "time", lambda: 1234.0)
    client = TestClient(create_app(EventRuntime(canonical_events)))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fusion-coding",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    events = _sse_events(response.text)

    assert response.status_code == 200
    assert response.headers["x-fusion-streaming-mode"] == "native-final"
    assert [event for event, _ in events] == ["data"] * 8
    assert events[0][1]["choices"][0]["delta"] == {"role": "assistant"}
    assert [
        events[1][1]["choices"][0]["delta"]["content"],
        events[2][1]["choices"][0]["delta"]["content"],
    ] == [
        "he",
        "llo",
    ]
    assert events[3][1]["choices"][0]["delta"]["tool_calls"] == [
        {
            "index": 0,
            "type": "function",
            "id": "call_1",
            "function": {"name": "read_file"},
        }
    ]
    assert events[4][1]["choices"][0]["delta"]["tool_calls"] == [
        {
            "index": 0,
            "type": "function",
            "function": {"arguments": '{"path":"README.md"}'},
        }
    ]
    assert events[5][1]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[6][1]["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
    }
    assert events[7] == ("data", "[DONE]")


def test_anthropic_messages_frozen_stream_mapping(canonical_events):
    client = TestClient(create_app(EventRuntime(canonical_events)))
    response = client.post(
        "/v1/messages",
        json={
            "model": "fusion-coding",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    events = _sse_events(response.text)

    assert [event for event, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1][1]["content_block"] == {"type": "text", "text": ""}
    assert [events[2][1]["delta"]["text"], events[3][1]["delta"]["text"]] == [
        "he",
        "llo",
    ]
    assert events[5][1]["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read_file",
        "input": {},
    }
    assert events[6][1]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"path":"README.md"}',
    }
    assert events[8][1]["delta"]["stop_reason"] == "tool_use"
    assert events[8][1]["usage"] == {"output_tokens": 3}


def test_openai_responses_frozen_stream_mapping(canonical_events, monkeypatch):
    monkeypatch.setattr(gateway.time, "time", lambda: 1234.5)
    monkeypatch.setattr(gateway.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    client = TestClient(create_app(EventRuntime(canonical_events)))
    response = client.post(
        "/v1/responses",
        json={"model": "fusion-coding", "stream": True, "input": "hi"},
    )
    events = _sse_events(response.text)

    assert [event for event, _ in events] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [data["sequence_number"] for _, data in events] == list(range(len(events)))
    assert [events[3][1]["delta"], events[4][1]["delta"]] == ["he", "llo"]
    assert events[6][1]["delta"] == '{"path":"README.md"}'
    completed = events[-1][1]["response"]
    assert completed["status"] == "completed"
    assert completed["created_at"] == 1234.5
    assert completed["output"][0]["content"][0]["text"] == "hello"
    assert completed["output"][1]["call_id"] == "call_1"
    assert completed["output"][1]["name"] == "read_file"
    assert completed["usage"]["total_tokens"] == 7


@pytest.mark.parametrize(
    ("path", "payload", "terminal_event", "forbidden"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "fusion-coding",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "data",
            '"finish_reason":"stop"',
        ),
        (
            "/v1/messages",
            {
                "model": "fusion-coding",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "error",
            "event: message_stop",
        ),
        (
            "/v1/responses",
            {"model": "fusion-coding", "stream": True, "input": "hi"},
            "response.failed",
            "event: response.completed",
        ),
    ],
)
def test_terminal_stream_error_has_protocol_native_mapping(
    path, payload, terminal_event, forbidden
):
    runtime = EventRuntime(
        [StreamError("upstream provider stream failed", code="overloaded", retryable=True)]
    )
    response = TestClient(create_app(runtime)).post(path, json=payload)
    events = _sse_events(response.text)

    assert response.status_code == 200
    assert events[-2 if path == "/v1/chat/completions" else -1][0] == terminal_event
    assert "overloaded" in response.text
    assert forbidden not in response.text


def test_pre_stream_provider_failure_is_json_not_broken_sse():
    class FailingRuntime(EventRuntime):
        async def stream(self, request):
            raise ProviderHTTPError(429)

    response = TestClient(create_app(FailingRuntime([]))).post(
        "/v1/chat/completions",
        json={
            "model": "fusion-coding",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "error": {
            "message": "upstream provider returned HTTP 429",
            "type": "upstream_error",
            "code": "upstream_rate_limited",
            "retryable": True,
        }
    }
