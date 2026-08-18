from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx

from .config import ModelSpec, ProviderSpec
from .errors import ProviderError, ProviderHTTPError, ProviderProtocolError, ProviderTransportError
from .types import (
    Finish,
    FusionRequest,
    ModelResponse,
    ModelStreamEvent,
    StreamError,
    TextDelta,
    ToolCallDelta,
    Usage,
)


class Provider(Protocol):
    async def complete(self, model: ModelSpec, request: FusionRequest) -> ModelResponse: ...
    def stream(
        self, model: ModelSpec, request: FusionRequest
    ) -> AsyncIterator[ModelStreamEvent]: ...
    async def aclose(self) -> None: ...


class OpenAICompatibleProvider:
    def __init__(self, spec: ProviderSpec, client: httpx.AsyncClient | None = None) -> None:
        headers = dict(spec.headers)
        if spec.api_key():
            headers["authorization"] = f"Bearer {spec.api_key()}"
        self._client = client or httpx.AsyncClient(
            base_url=spec.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=spec.timeout_seconds,
        )

    async def complete(self, model: ModelSpec, request: FusionRequest) -> ModelResponse:
        payload = self._payload(model, request)
        payload["stream"] = False
        try:
            response = await self._client.post("chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderTransportError() from exc
        _raise_for_status(response)
        body = _response_json(response)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderProtocolError()
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProviderProtocolError()
        return ModelResponse(
            content=message.get("content") or "",
            tool_calls=message.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason") or "stop",
            usage=body.get("usage") or {},
            raw=body,
        )

    async def stream(
        self, model: ModelSpec, request: FusionRequest
    ) -> AsyncIterator[ModelStreamEvent]:
        payload = self._payload(model, request)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        emitted = False
        finished = False
        try:
            async with self._client.stream("POST", "chat/completions", json=payload) as response:
                _raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        body = json.loads(data)
                    except json.JSONDecodeError as exc:
                        if finished:
                            return
                        if emitted:
                            yield StreamError(
                                "upstream provider emitted malformed stream JSON",
                                code="provider_protocol_error",
                            )
                            return
                        raise ProviderProtocolError(
                            "upstream provider emitted malformed stream JSON"
                        ) from exc
                    if not isinstance(body, dict):
                        if finished:
                            return
                        if emitted:
                            yield StreamError(
                                "upstream provider emitted an invalid stream event",
                                code="provider_protocol_error",
                            )
                            return
                        raise ProviderProtocolError(
                            "upstream provider emitted an invalid stream event"
                        )
                    if body.get("error"):
                        if finished:
                            return
                        yield _canonical_stream_error(body["error"])
                        return
                    for choice in body.get("choices") or []:
                        if not isinstance(choice, dict):
                            continue
                        if finished:
                            continue
                        delta = choice.get("delta") or {}
                        if not isinstance(delta, dict):
                            continue
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            emitted = True
                            yield TextDelta(content)
                        for call in delta.get("tool_calls") or []:
                            if not isinstance(call, dict):
                                continue
                            function = call.get("function") or {}
                            if not isinstance(function, dict):
                                function = {}
                            event = ToolCallDelta(
                                index=_non_negative_index(call.get("index")),
                                id=_optional_string(call.get("id")),
                                name=_optional_string(function.get("name")),
                                arguments=_optional_string(function.get("arguments")) or "",
                            )
                            if event.id is not None or event.name is not None or event.arguments:
                                emitted = True
                                yield event
                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            emitted = True
                            finished = True
                            yield Finish(str(finish_reason))
                    usage = body.get("usage")
                    if isinstance(usage, dict) and usage:
                        emitted = True
                        yield Usage(_integer_usage(usage))
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            if finished:
                return
            if emitted:
                yield StreamError(
                    "upstream provider transport failed",
                    code="upstream_transport_error",
                    retryable=True,
                )
                return
            raise ProviderTransportError() from exc
        if finished:
            return
        if emitted:
            yield StreamError(
                "upstream provider ended without a finish event",
                code="provider_protocol_error",
            )
            return
        raise ProviderProtocolError("upstream provider returned an empty stream")

    @staticmethod
    def _payload(model: ModelSpec, request: FusionRequest) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model.model,
            "messages": request.messages,
            "max_tokens": min(request.max_tokens or model.max_output, model.max_output),
        }
        if request.tools and model.tool_calling:
            payload["tools"] = request.tools
            payload["tool_choice"] = request.tool_choice or "auto"
            if request.parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = request.parallel_tool_calls
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort
        return payload

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicCompatibleProvider:
    def __init__(self, spec: ProviderSpec, client: httpx.AsyncClient | None = None) -> None:
        headers = {"anthropic-version": "2023-06-01", **spec.headers}
        if spec.api_key():
            headers["x-api-key"] = spec.api_key() or ""
        self._client = client or httpx.AsyncClient(
            base_url=spec.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=spec.timeout_seconds,
        )

    async def complete(self, model: ModelSpec, request: FusionRequest) -> ModelResponse:
        payload = self._payload(model, request)
        payload["stream"] = False
        try:
            response = await self._client.post("messages", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderTransportError() from exc
        _raise_for_status(response)
        body = _response_json(response)
        content = body.get("content")
        if not isinstance(content, list):
            raise ProviderProtocolError()
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        tool_calls = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}, separators=(",", ":")),
                    },
                }
            )
        return ModelResponse(
            content=text,
            tool_calls=tool_calls,
            finish_reason=body.get("stop_reason") or "stop",
            usage=body.get("usage") or {},
            raw=body,
        )

    async def stream(
        self, model: ModelSpec, request: FusionRequest
    ) -> AsyncIterator[ModelStreamEvent]:
        payload = self._payload(model, request)
        payload["stream"] = True
        emitted = False
        finished = False
        try:
            async with self._client.stream("POST", "messages", json=payload) as response:
                _raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        if finished:
                            return
                        if emitted:
                            yield StreamError(
                                "upstream provider emitted malformed stream JSON",
                                code="provider_protocol_error",
                            )
                            return
                        raise ProviderProtocolError(
                            "upstream provider emitted malformed stream JSON"
                        ) from exc
                    if not isinstance(event, dict):
                        if finished:
                            return
                        if emitted:
                            yield StreamError(
                                "upstream provider emitted an invalid stream event",
                                code="provider_protocol_error",
                            )
                            return
                        raise ProviderProtocolError(
                            "upstream provider emitted an invalid stream event"
                        )
                    event_type = event.get("type")
                    if event_type == "error":
                        if finished:
                            return
                        yield _canonical_stream_error(event.get("error"))
                        return
                    if finished and event_type != "message_stop":
                        continue
                    if event_type == "message_start":
                        message = event.get("message") or {}
                        usage = message.get("usage") if isinstance(message, dict) else None
                        if isinstance(usage, dict) and usage:
                            emitted = True
                            yield Usage(_integer_usage(usage))
                    elif event_type == "content_block_start":
                        block = event.get("content_block") or {}
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            initial_input = block.get("input") or {}
                            emitted = True
                            yield ToolCallDelta(
                                index=_non_negative_index(event.get("index")),
                                id=_optional_string(block.get("id")),
                                name=_optional_string(block.get("name")),
                                arguments=(
                                    json.dumps(initial_input, separators=(",", ":"))
                                    if initial_input
                                    else ""
                                ),
                            )
                    elif event_type == "content_block_delta":
                        delta = event.get("delta") or {}
                        if not isinstance(delta, dict):
                            continue
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            emitted = True
                            yield TextDelta(str(delta["text"]))
                        elif delta.get("type") == "input_json_delta":
                            arguments = str(delta.get("partial_json") or "")
                            if arguments:
                                emitted = True
                                yield ToolCallDelta(
                                    index=_non_negative_index(event.get("index")),
                                    arguments=arguments,
                                )
                    elif event_type == "message_delta":
                        usage = event.get("usage")
                        if isinstance(usage, dict) and usage:
                            emitted = True
                            yield Usage(_integer_usage(usage))
                        delta = event.get("delta") or {}
                        stop_reason = delta.get("stop_reason") if isinstance(delta, dict) else None
                        if stop_reason is not None:
                            emitted = True
                            finished = True
                            yield Finish(str(stop_reason))
                    elif event_type == "message_stop":
                        break
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            if finished:
                return
            if emitted:
                yield StreamError(
                    "upstream provider transport failed",
                    code="upstream_transport_error",
                    retryable=True,
                )
                return
            raise ProviderTransportError() from exc
        if finished:
            return
        if emitted:
            yield StreamError(
                "upstream provider ended without a finish event",
                code="provider_protocol_error",
            )
            return
        raise ProviderProtocolError("upstream provider returned an empty stream")

    @staticmethod
    def _payload(model: ModelSpec, request: FusionRequest) -> dict[str, object]:
        system = "\n\n".join(
            _content_text(item.get("content", ""))
            for item in request.messages
            if item.get("role") in {"system", "developer"}
        )
        messages = _anthropic_messages(request.messages)
        payload: dict[str, object] = {
            "model": model.model,
            "messages": messages,
            "max_tokens": min(request.max_tokens or model.max_output, model.max_output),
        }
        if system:
            payload["system"] = system
        if request.tools and model.tool_calling:
            payload["tools"] = _anthropic_tools(request.tools)
            tool_choice = _anthropic_tool_choice(request.tool_choice)
            if request.parallel_tool_calls is not None:
                tool_choice["disable_parallel_tool_use"] = not request.parallel_tool_calls
            if tool_choice["type"] != "none":
                payload["tool_choice"] = tool_choice
            else:
                payload.pop("tools", None)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    async def aclose(self) -> None:
        await self._client.aclose()


def _anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in {"system", "developer"}:
            continue
        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": _content_text(message.get("content", "")),
                        }
                    ],
                }
            )
            continue
        blocks = _anthropic_content_blocks(message.get("content", ""))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            try:
                tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (TypeError, json.JSONDecodeError):
                tool_input = {"_raw": _content_text(arguments)}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id", ""),
                    "name": function.get("name", ""),
                    "input": tool_input,
                }
            )
        converted.append({"role": role, "content": blocks})
    return converted


def _anthropic_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": _content_text(content)}]
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": _content_text(part.get("text", ""))})
        elif part.get("type") == "image_url":
            image = part.get("image_url") or {}
            url = image.get("url") if isinstance(image, dict) else image
            if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                prefix, data = url.split(",", 1)
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": prefix[5:].split(";", 1)[0],
                            "data": data,
                        },
                    }
                )
            elif isinstance(url, str):
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for tool in tools:
        function = tool.get("function") or {}
        converted.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters") or {},
            }
        )
    return converted


def _anthropic_tool_choice(choice: str | dict[str, Any] | None) -> dict[str, Any]:
    if choice in (None, "auto"):
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    if isinstance(choice, dict):
        function = choice.get("function") or {}
        return {"type": "tool", "name": function.get("name", "")}
    return {"type": "auto"}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if content is None:
        return ""
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return str(content)


def _integer_usage(usage: dict[str, Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in usage.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        raise ProviderHTTPError(response.status_code)


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise ProviderProtocolError() from exc
    if not isinstance(body, dict):
        raise ProviderProtocolError()
    return body


def _canonical_stream_error(error: Any) -> StreamError:
    code = "provider_stream_error"
    retryable = False
    if isinstance(error, dict):
        candidate = error.get("code") or error.get("type")
        if isinstance(candidate, str):
            normalized = "".join(
                character
                for character in candidate[:64]
                if character.isalnum() or character in "-_"
            )
            if normalized:
                code = normalized
        status = error.get("status") or error.get("status_code")
        if isinstance(status, int) and not isinstance(status, bool):
            retryable = status in {408, 409, 425, 429} or status >= 500
    return StreamError("upstream provider stream failed", code=code, retryable=retryable)


def _non_negative_index(value: Any) -> int:
    try:
        index = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(index, 0)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
