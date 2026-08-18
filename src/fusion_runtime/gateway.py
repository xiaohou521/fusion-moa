from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .runtime import CapabilityError, FusionRuntime
from .types import (
    Finish,
    FusionRequest,
    FusionResult,
    FusionStream,
    ModelResponse,
    TextDelta,
    ToolCallDelta,
    Usage,
)


def create_app(runtime: FusionRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        close = getattr(runtime, "aclose", None)
        if close is not None:
            await close()

    app = FastAPI(title="Fusion Runtime", version="0.1.0", lifespan=lifespan)
    spec = runtime.spec

    @app.exception_handler(CapabilityError)
    async def capability_error(_request: Request, exc: CapabilityError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "code": "unsupported_capability",
                }
            },
        )

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        env_name = spec.serve.api_key_env
        if not env_name:
            return
        import os

        expected = os.getenv(env_name)
        if not expected or authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": spec.serve.model_name,
            "policy": spec.policy.type,
            "providers": sorted(spec.providers),
            "protocols": sorted(spec.serve.protocols),
        }

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": spec.serve.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "fusion-runtime",
                }
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat(payload: dict[str, Any]) -> Response:
        _require_protocol(spec.serve.protocols, "openai-chat")
        _require_public_model(payload, spec.serve.model_name)
        fusion_request = _chat_request(payload)
        created = int(time.time())
        if payload.get("stream"):
            stream = await runtime.stream(fusion_request)
            completion_id = f"chatcmpl_{stream.trace_id}"
            headers = {**_trace_headers(stream), "x-fusion-streaming-mode": "native-final"}
            stream_options = payload.get("stream_options") or {}
            include_usage = bool(
                isinstance(stream_options, dict) and stream_options.get("include_usage")
            )
            return StreamingResponse(
                _chat_stream(
                    stream,
                    completion_id,
                    spec.serve.model_name,
                    created,
                    include_usage=include_usage,
                ),
                media_type="text/event-stream",
                headers=headers,
            )
        result = await runtime.complete(fusion_request)
        completion_id = f"chatcmpl_{result.trace_id}"
        body = _chat_body(result, completion_id, spec.serve.model_name, created)
        return JSONResponse(body, headers=_trace_headers(result))

    @app.post("/v1/messages", dependencies=[Depends(authorize)])
    async def messages(payload: dict[str, Any]) -> Response:
        _require_protocol(spec.serve.protocols, "anthropic-messages")
        _require_public_model(payload, spec.serve.model_name)
        fusion_request = _anthropic_request(payload)
        if payload.get("stream"):
            stream = await runtime.stream(fusion_request)
            message_id = f"msg_{stream.trace_id}"
            headers = {**_trace_headers(stream), "x-fusion-streaming-mode": "native-final"}
            return StreamingResponse(
                _anthropic_stream(stream, message_id, spec.serve.model_name),
                media_type="text/event-stream",
                headers=headers,
            )
        result = await runtime.complete(fusion_request)
        message_id = f"msg_{result.trace_id}"
        body = _anthropic_body(result, message_id, spec.serve.model_name)
        return JSONResponse(body, headers=_trace_headers(result))

    @app.post("/v1/responses", dependencies=[Depends(authorize)])
    async def responses(payload: dict[str, Any]) -> Response:
        _require_protocol(spec.serve.protocols, "openai-responses")
        _require_public_model(payload, spec.serve.model_name)
        fusion_request = _responses_request(payload)
        if payload.get("stream"):
            stream = await runtime.stream(fusion_request)
            response_id = f"resp_{stream.trace_id}"
            headers = {**_trace_headers(stream), "x-fusion-streaming-mode": "native-final"}
            return StreamingResponse(
                _responses_stream(stream, response_id, spec.serve.model_name),
                media_type="text/event-stream",
                headers=headers,
            )
        result = await runtime.complete(fusion_request)
        response_id = f"resp_{result.trace_id}"
        body = _responses_body(result, response_id, spec.serve.model_name)
        return JSONResponse(body, headers=_trace_headers(result))

    return app


def _require_protocol(enabled: set[str], protocol: str) -> None:
    if protocol not in enabled:
        raise HTTPException(status_code=404, detail=f"protocol {protocol!r} is disabled")


def _require_public_model(payload: dict[str, Any], public_model: str) -> None:
    requested = payload.get("model")
    if requested is not None and requested != public_model:
        raise HTTPException(status_code=404, detail=f"unknown model {requested!r}")


def _chat_request(payload: dict[str, Any]) -> FusionRequest:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=422, detail="messages must be a list")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }:
            raise HTTPException(status_code=422, detail="invalid chat message")
    return _request(
        payload,
        messages=messages,
        tools=_responses_tools(payload.get("tools") or []),
        tool_choice=payload.get("tool_choice"),
    )


def _responses_request(payload: dict[str, Any]) -> FusionRequest:
    messages = _responses_messages(payload.get("input", []), payload.get("instructions"))
    return _request(
        payload,
        messages=messages,
        tools=_responses_tools(payload.get("tools") or []),
        tool_choice=_responses_tool_choice(payload.get("tool_choice")),
    )


def _anthropic_request(payload: dict[str, Any]) -> FusionRequest:
    messages = _anthropic_messages(payload.get("messages"), payload.get("system"))
    tool_choice, parallel_tool_calls = _anthropic_tool_choice(payload.get("tool_choice"))
    return _request(
        payload,
        messages=messages,
        tools=_anthropic_tools(payload.get("tools") or []),
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )


def _request(
    payload: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: Any = None,
    parallel_tool_calls: bool | None = None,
) -> FusionRequest:
    max_tokens = payload.get("max_tokens")
    if max_tokens is None:
        max_tokens = payload.get("max_output_tokens")
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0
    ):
        raise HTTPException(status_code=422, detail="max_tokens must be a positive integer")
    temperature = payload.get("temperature")
    if temperature is not None and (
        not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
    ):
        raise HTTPException(status_code=422, detail="temperature must be numeric")
    if tool_choice is not None and not tools:
        raise HTTPException(status_code=422, detail="tool_choice requires tools")
    if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
        raise HTTPException(status_code=422, detail="tool_choice must be a string or object")
    if parallel_tool_calls is None:
        parallel_tool_calls = payload.get("parallel_tool_calls")
    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        raise HTTPException(status_code=422, detail="parallel_tool_calls must be boolean")
    reasoning_effort = payload.get("reasoning_effort")
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        raise HTTPException(status_code=422, detail="reasoning_effort must be a string")
    return FusionRequest(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _responses_messages(items: Any, instructions: Any) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": _text_content(instructions)}] if instructions else []
    if isinstance(items, str):
        return [*messages, {"role": "user", "content": items}]
    if not isinstance(items, list):
        raise HTTPException(status_code=422, detail="input must be a string or list")
    for item in items:
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail="response input items must be objects")
        item_type = item.get("type", "message")
        if item_type == "message":
            messages.append(
                {
                    "role": item.get("role", "user"),
                    "content": _responses_content(item.get("content", "")),
                }
            )
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.get("call_id")
                            or item.get("id")
                            or f"call_{uuid.uuid4().hex}",
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": _text_content(item.get("output", "")),
                }
            )
        elif item_type == "reasoning":
            # Provider-neutral gateways cannot safely replay opaque reasoning state.
            continue
        else:
            raise HTTPException(
                status_code=422, detail=f"unsupported response input type {item_type!r}"
            )
    return messages


def _responses_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _text_content(content)
    converted: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            converted.append({"type": "text", "text": _text_content(part.get("text", ""))})
        elif part_type == "input_image" and part.get("image_url"):
            converted.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
    if all(part.get("type") == "text" for part in converted):
        return "\n".join(part["text"] for part in converted)
    return converted


def _responses_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise HTTPException(status_code=422, detail="tools must be a list")
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise HTTPException(
                status_code=422, detail="only function tools are portable in fusion/v1"
            )
        if isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {},
                },
            }
        )
    return converted


def _responses_tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict) or choice.get("type") != "function":
        return choice
    return {
        "type": "function",
        "function": {"name": choice.get("name", "")},
    }


def _anthropic_messages(items: Any, system: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise HTTPException(status_code=422, detail="messages must be a list")
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": _anthropic_text(system)})
    for item in items:
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail="messages must contain objects")
        role = item.get("role")
        content = item.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise HTTPException(
                status_code=422, detail="Anthropic message content must be text or blocks"
            )
        text_parts: list[str] = []
        media_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(_text_content(block.get("text", "")))
            elif block_type == "image":
                image = _anthropic_image(block.get("source"))
                if image:
                    media_parts.append(image)
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(
                                block.get("input") or {}, separators=(",", ":")
                            ),
                        },
                    }
                )
            elif block_type == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _anthropic_text(block.get("content", "")),
                    }
                )
        message_content: str | list[dict[str, Any]] = "\n".join(text_parts)
        if media_parts:
            message_content = [
                *({"type": "text", "text": text} for text in text_parts),
                *media_parts,
            ]
        if text_parts or media_parts or tool_calls:
            message: dict[str, Any] = {"role": role, "content": message_content}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
        messages.extend(tool_results)
    return messages


def _anthropic_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise HTTPException(status_code=422, detail="tools must be a list")
    converted = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            raise HTTPException(status_code=422, detail="invalid Anthropic tool")
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {},
                },
            }
        )
    return converted


def _anthropic_tool_choice(choice: Any) -> tuple[Any, bool | None]:
    if choice is None:
        return None, None
    if not isinstance(choice, dict):
        raise HTTPException(status_code=422, detail="Anthropic tool_choice must be an object")
    choice_type = choice.get("type")
    parallel = None
    if "disable_parallel_tool_use" in choice:
        disabled = choice["disable_parallel_tool_use"]
        if not isinstance(disabled, bool):
            raise HTTPException(status_code=422, detail="disable_parallel_tool_use must be boolean")
        parallel = not disabled
    if choice_type == "auto":
        return "auto", parallel
    if choice_type == "any":
        return "required", parallel
    if choice_type == "tool":
        return {"type": "function", "function": {"name": choice.get("name", "")}}, parallel
    if choice_type == "none":
        return "none", parallel
    raise HTTPException(
        status_code=422, detail=f"unsupported Anthropic tool_choice {choice_type!r}"
    )


def _anthropic_image(source: Any) -> dict[str, Any] | None:
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64" and source.get("media_type") and source.get("data"):
        url = f"data:{source['media_type']};base64,{source['data']}"
        return {"type": "image_url", "image_url": {"url": url}}
    if source.get("type") == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _anthropic_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _text_content(content)
    return "\n".join(
        _text_content(part.get("text", part.get("content", "")))
        for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "tool_result"}
    )


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _chat_body(
    result: FusionResult,
    completion_id: str,
    model: str,
    created: int,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.response.content}
    if result.response.tool_calls:
        message["tool_calls"] = result.response.tool_calls
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _chat_finish_reason(result.response),
            }
        ],
        "usage": result.response.usage,
    }


async def _chat_stream(
    stream: FusionStream,
    completion_id: str,
    model: str,
    created: int,
    *,
    include_usage: bool,
) -> AsyncIterator[str]:
    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    yield _data_sse(
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    )
    tool_seen = False
    finish_seen = False
    async for event in stream.events:
        if isinstance(event, TextDelta):
            yield _data_sse(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": event.text},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        elif isinstance(event, ToolCallDelta):
            tool_seen = True
            function: dict[str, str] = {}
            if event.name is not None:
                function["name"] = event.name
            if event.arguments:
                function["arguments"] = event.arguments
            call: dict[str, Any] = {"index": event.index, "type": "function"}
            if event.id is not None:
                call["id"] = event.id
            if function:
                call["function"] = function
            yield _data_sse(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [call]},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        elif isinstance(event, Finish):
            finish_seen = True
            yield _data_sse(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": _chat_finish_reason_value(
                                event.reason, tool_seen=tool_seen
                            ),
                        }
                    ],
                }
            )
        elif isinstance(event, Usage) and include_usage:
            yield _data_sse({**base, "choices": [], "usage": event.usage})
    if not finish_seen:
        yield _data_sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if tool_seen else "stop",
                    }
                ],
            }
        )
    yield "data: [DONE]\n\n"


def _anthropic_body(result: FusionResult, message_id: str, model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if result.response.content:
        content.append({"type": "text", "text": result.response.content})
    for call in result.response.tool_calls:
        function = call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            tool_input = {"_raw": _text_content(arguments)}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex}",
                "name": function.get("name", ""),
                "input": tool_input,
            }
        )
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _anthropic_stop_reason(result.response),
        "stop_sequence": None,
        "usage": _anthropic_usage(result.response.usage),
    }


async def _anthropic_stream(
    stream: FusionStream, message_id: str, model: str
) -> AsyncIterator[str]:
    start = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    yield _event_sse("message_start", {"type": "message_start", "message": start})
    next_content_index = 0
    active_key: tuple[str, int] | None = None
    active_index: int | None = None
    content_indices: dict[tuple[str, int], int] = {}
    tool_seen = False
    finish_reason = "end_turn"
    usage: dict[str, int] = {}

    async for event in stream.events:
        key: tuple[str, int] | None = None
        if isinstance(event, TextDelta):
            key = ("text", 0)
        elif isinstance(event, ToolCallDelta):
            key = ("tool", event.index)
        if key is not None and key != active_key:
            if active_index is not None:
                yield _event_sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": active_index},
                )
            if key not in content_indices:
                content_indices[key] = next_content_index
                next_content_index += 1
            active_key = key
            active_index = content_indices[key]
            if key[0] == "text":
                content_block = {"type": "text", "text": ""}
            else:
                assert isinstance(event, ToolCallDelta)
                tool_seen = True
                content_block = {
                    "type": "tool_use",
                    "id": event.id or f"toolu_{uuid.uuid4().hex}",
                    "name": event.name or "",
                    "input": {},
                }
            yield _event_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": active_index,
                    "content_block": content_block,
                },
            )
        if isinstance(event, TextDelta):
            assert active_index is not None
            yield _event_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": active_index,
                    "delta": {"type": "text_delta", "text": event.text},
                },
            )
        elif isinstance(event, ToolCallDelta) and event.arguments:
            assert active_index is not None
            yield _event_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": active_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": event.arguments,
                    },
                },
            )
        elif isinstance(event, Finish):
            finish_reason = _anthropic_stop_reason_value(event.reason, tool_seen=tool_seen)
        elif isinstance(event, Usage):
            usage.update(event.usage)

    if active_index is not None:
        yield _event_sse(
            "content_block_stop", {"type": "content_block_stop", "index": active_index}
        )
    mapped_usage = _anthropic_usage(usage)
    yield _event_sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": finish_reason, "stop_sequence": None},
            "usage": {"output_tokens": mapped_usage["output_tokens"]},
        },
    )
    yield _event_sse("message_stop", {"type": "message_stop"})


def _responses_body(result: FusionResult, response_id: str, model: str) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if result.response.content or not result.response.tool_calls:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": result.response.content, "annotations": []}
                ],
            }
        )
    for call in result.response.tool_calls:
        function = call.get("function") or {}
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": call.get("id") or f"call_{uuid.uuid4().hex}",
                "name": function.get("name", ""),
                "arguments": function.get("arguments") or "{}",
            }
        )
    return {
        "id": response_id,
        "object": "response",
        "created_at": time.time(),
        "status": "completed",
        "model": model,
        "output": output,
        "error": None,
        "incomplete_details": None,
        "usage": _responses_usage(result.response.usage),
    }


async def _responses_stream(
    stream: FusionStream, response_id: str, model: str
) -> AsyncIterator[str]:
    created_at = time.time()
    pending = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "model": model,
        "output": [],
        "error": None,
        "incomplete_details": None,
        "usage": None,
    }
    yield _event_sse(
        "response.created", {"type": "response.created", "response": pending, "sequence_number": 0}
    )
    sequence = 1
    text_item: dict[str, Any] | None = None
    text = ""
    tool_items: dict[int, dict[str, Any]] = {}
    output_order: list[tuple[str, int]] = []
    usage: dict[str, int] = {}

    async for event in stream.events:
        if isinstance(event, TextDelta):
            if text_item is None:
                output_index = len(output_order)
                output_order.append(("text", 0))
                text_item = {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [],
                    "output_index": output_index,
                }
                added = {key: value for key, value in text_item.items() if key != "output_index"}
                added["status"] = "in_progress"
                yield _event_sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": added,
                        "sequence_number": sequence,
                    },
                )
                sequence += 1
                yield _event_sse(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": text_item["id"],
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                        "sequence_number": sequence,
                    },
                )
                sequence += 1
            text += event.text
            yield _event_sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": text_item["id"],
                    "output_index": text_item["output_index"],
                    "content_index": 0,
                    "delta": event.text,
                    "sequence_number": sequence,
                },
            )
            sequence += 1
        elif isinstance(event, ToolCallDelta):
            item = tool_items.get(event.index)
            new_item = item is None
            if item is None:
                output_index = len(output_order)
                output_order.append(("tool", event.index))
                item = {
                    "id": f"fc_{uuid.uuid4().hex}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": event.id or f"call_{uuid.uuid4().hex}",
                    "name": event.name or "",
                    "arguments": "",
                    "output_index": output_index,
                }
                tool_items[event.index] = item
                added = {key: value for key, value in item.items() if key != "output_index"}
                added["status"] = "in_progress"
                yield _event_sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": added,
                        "sequence_number": sequence,
                    },
                )
                sequence += 1
            if event.id:
                item["call_id"] = event.id
            if event.name and not new_item:
                item["name"] += event.name
            if event.arguments:
                item["arguments"] += event.arguments
                yield _event_sse(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": item["id"],
                        "output_index": item["output_index"],
                        "delta": event.arguments,
                        "sequence_number": sequence,
                    },
                )
                sequence += 1
        elif isinstance(event, Usage):
            usage.update(event.usage)

    if not output_order:
        output_order.append(("text", 0))
        text_item = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [],
            "output_index": 0,
        }
        added = {key: value for key, value in text_item.items() if key != "output_index"}
        added["status"] = "in_progress"
        yield _event_sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": added,
                "sequence_number": sequence,
            },
        )
        sequence += 1
        yield _event_sse(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": text_item["id"],
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
                "sequence_number": sequence,
            },
        )
        sequence += 1

    output: list[dict[str, Any]] = []
    for kind, canonical_index in output_order:
        if kind == "text":
            assert text_item is not None
            output_index = text_item["output_index"]
            part = {"type": "output_text", "text": text, "annotations": []}
            final_item = {key: value for key, value in text_item.items() if key != "output_index"}
            final_item["content"] = [part]
            yield _event_sse(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": text_item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                    "sequence_number": sequence,
                },
            )
            sequence += 1
            yield _event_sse(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": text_item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "part": part,
                    "sequence_number": sequence,
                },
            )
            sequence += 1
        else:
            item = tool_items[canonical_index]
            output_index = item["output_index"]
            final_item = {key: value for key, value in item.items() if key != "output_index"}
            yield _event_sse(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "arguments": item["arguments"],
                    "sequence_number": sequence,
                },
            )
            sequence += 1
        yield _event_sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": final_item,
                "sequence_number": sequence,
            },
        )
        sequence += 1
        output.append(final_item)
    body = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model,
        "output": output,
        "error": None,
        "incomplete_details": None,
        "usage": _responses_usage(usage),
    }
    yield _event_sse(
        "response.completed",
        {"type": "response.completed", "response": body, "sequence_number": sequence},
    )


def _anthropic_stop_reason(response: ModelResponse) -> str:
    return _anthropic_stop_reason_value(response.finish_reason, tool_seen=bool(response.tool_calls))


def _anthropic_stop_reason_value(reason: str, *, tool_seen: bool) -> str:
    if tool_seen:
        return "tool_use"
    return {
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "tool_calls": "tool_use",
        "stop": "end_turn",
    }.get(reason, reason or "end_turn")


def _chat_finish_reason(response: ModelResponse) -> str:
    return _chat_finish_reason_value(response.finish_reason, tool_seen=bool(response.tool_calls))


def _chat_finish_reason_value(reason: str, *, tool_seen: bool) -> str:
    if tool_seen:
        return "tool_calls"
    return {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(reason, reason or "stop")


def _anthropic_usage(usage: dict[str, int]) -> dict[str, int]:
    return {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
    }


def _responses_usage(usage: dict[str, int]) -> dict[str, Any]:
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": usage.get("input_tokens_details", {"cached_tokens": 0}),
        "output_tokens": output_tokens,
        "output_tokens_details": usage.get("output_tokens_details", {"reasoning_tokens": 0}),
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
    }


def _data_sse(data: dict[str, Any]) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n\n"


def _event_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\n" + _data_sse(data)


def _trace_headers(result: FusionResult | FusionStream) -> dict[str, str]:
    headers = {"x-fusion-trace-id": result.trace_id, "x-fusion-route": result.route}
    if result.fallback_reason:
        headers["x-fusion-fallback"] = result.fallback_reason[:200]
    return headers
