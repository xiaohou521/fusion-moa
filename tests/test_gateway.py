from fastapi.testclient import TestClient

from fusion_runtime.gateway import create_app
from fusion_runtime.types import (
    Finish,
    FusionResult,
    FusionStream,
    ModelResponse,
    TextDelta,
    ToolCallDelta,
    Usage,
)


class StubRuntime:
    def __init__(self, *, protocols=None, response=None):
        from fusion_runtime.config import FusionSpec

        self.spec = FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {"p": {"type": "openai-compatible", "base_url": "http://unused"}},
                "models": {"m": {"provider": "p", "model": "m", "context_window": 10}},
                "pools": {"coding": {"main": "m"}},
                "serve": {
                    "pool": "coding",
                    "model_name": "fusion-coding",
                    "protocols": protocols
                    or ["openai-chat", "openai-responses", "anthropic-messages"],
                },
            }
        )
        self.requests = []
        self.response = response or ModelResponse(content="hello", usage={"total_tokens": 3})

    async def complete(self, request):
        assert request.messages
        self.requests.append(request)
        return FusionResult(
            response=self.response,
            route="direct",
            trace_id="trace123",
        )

    async def stream(self, request):
        assert request.messages
        self.requests.append(request)

        async def events():
            if self.response.content:
                midpoint = max(1, len(self.response.content) // 2)
                yield TextDelta(self.response.content[:midpoint])
                yield TextDelta(self.response.content[midpoint:])
            for index, call in enumerate(self.response.tool_calls):
                function = call.get("function") or {}
                yield ToolCallDelta(
                    index=index,
                    id=call.get("id"),
                    name=function.get("name"),
                )
                yield ToolCallDelta(
                    index=index,
                    arguments=function.get("arguments") or "",
                )
            yield Finish(self.response.finish_reason)
            if self.response.usage:
                yield Usage(self.response.usage)

        return FusionStream(events=events(), route="direct", trace_id="trace123")


def test_openai_chat_contract():
    client = TestClient(create_app(StubRuntime()))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "fusion-coding", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    assert response.headers["x-fusion-trace-id"] == "trace123"


def test_anthropic_messages_contract():
    client = TestClient(create_app(StubRuntime()))
    response = client.post(
        "/v1/messages",
        json={"model": "fusion-coding", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "hello"}]


def test_responses_contract():
    client = TestClient(create_app(StubRuntime()))
    response = client.post("/v1/responses", json={"model": "fusion-coding", "input": "hi"})
    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "hello"


def test_model_discovery_and_unknown_model():
    client = TestClient(create_app(StubRuntime()))
    listing = client.get("/v1/models")
    assert listing.json()["data"][0]["id"] == "fusion-coding"
    missing = client.post(
        "/v1/chat/completions",
        json={"model": "not-fusion", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert missing.status_code == 404


def test_disabled_protocol_is_not_exposed():
    client = TestClient(create_app(StubRuntime(protocols=["openai-chat"])))
    response = client.post(
        "/v1/responses",
        json={"model": "fusion-coding", "input": "hi"},
    )
    assert response.status_code == 404


def test_openai_chat_native_final_stream_contract():
    client = TestClient(create_app(StubRuntime()))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fusion-coding",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["x-fusion-streaming-mode"] == "native-final"
    assert '"delta":{"content":"he"}' in response.text
    assert '"delta":{"content":"llo"}' in response.text
    assert '"usage":{"total_tokens":3}' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_responses_tool_round_trip_and_stream_contract():
    runtime = StubRuntime(
        response=ModelResponse(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }
            ],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        )
    )
    client = TestClient(create_app(runtime))
    response = client.post(
        "/v1/responses",
        json={
            "model": "fusion-coding",
            "stream": True,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "read it"}],
                },
                {"type": "function_call", "call_id": "old_call", "name": "pwd", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "old_call", "output": "/tmp"},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read one file",
                    "parameters": {"type": "object"},
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "reasoning_effort": "high",
            "seed": 7,
        },
    )
    request = runtime.requests[0]
    assert request.messages[-1] == {"role": "tool", "tool_call_id": "old_call", "content": "/tmp"}
    assert request.tools[0]["function"]["name"] == "read_file"
    assert request.tool_choice == "required"
    assert request.parallel_tool_calls is False
    assert request.reasoning_effort == "high"
    assert request.seed == 7
    assert "event: response.function_call_arguments.delta" in response.text
    assert '"call_id":"call_1"' in response.text
    assert "event: response.completed" in response.text


def test_anthropic_tool_round_trip():
    runtime = StubRuntime(
        response=ModelResponse(
            content="checking",
            tool_calls=[
                {
                    "id": "toolu_new",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"command":"pwd"}'},
                }
            ],
            finish_reason="tool_calls",
        )
    )
    client = TestClient(create_app(runtime))
    response = client.post(
        "/v1/messages",
        json={
            "model": "fusion-coding",
            "max_tokens": 100,
            "system": [{"type": "text", "text": "Be careful"}],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calling"},
                        {
                            "type": "tool_use",
                            "id": "toolu_old",
                            "name": "shell",
                            "input": {"command": "ls"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_old", "content": "ok"}
                    ],
                },
            ],
            "tools": [{"name": "shell", "description": "run", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
        },
    )
    request = runtime.requests[0]
    assert request.messages[-1] == {"role": "tool", "tool_call_id": "toolu_old", "content": "ok"}
    assert request.tools[0]["function"]["name"] == "shell"
    assert request.tool_choice == "required"
    assert request.parallel_tool_calls is False
    assert response.json()["stop_reason"] == "tool_use"
    assert response.json()["content"][1]["input"] == {"command": "pwd"}


def test_anthropic_native_final_stream_contract():
    client = TestClient(create_app(StubRuntime()))
    response = client.post(
        "/v1/messages",
        json={
            "model": "fusion-coding",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.headers["x-fusion-streaming-mode"] == "native-final"
    assert '"delta":{"type":"text_delta","text":"he"}' in response.text
    assert "event: message_stop" in response.text


def test_undeclared_tool_capability_fails_visibly():
    from fusion_runtime.runtime import CapabilityError

    class RejectingRuntime(StubRuntime):
        async def complete(self, request):
            raise CapabilityError("model 'm' does not declare tool_calling")

    client = TestClient(create_app(RejectingRuntime()))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fusion-coding",
            "messages": [{"role": "user", "content": "run"}],
            "tools": [{"type": "function", "function": {"name": "shell"}}],
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_capability"
