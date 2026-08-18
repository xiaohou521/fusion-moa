from __future__ import annotations

import asyncio
import contextvars
import inspect
import uuid
from collections.abc import AsyncIterator

from .config import FusionSpec
from .errors import ProviderProtocolError
from .plugins import PluginRegistry
from .policies import DirectPolicy, MainCriticPolicy, ReviewBoardPolicy
from .providers import AnthropicCompatibleProvider, OpenAICompatibleProvider
from .types import FusionRequest, FusionResult, FusionStream, ModelResponse, ModelStreamEvent

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


class CapabilityError(ValueError):
    """The selected model does not declare a capability required by a request."""


class FusionRuntime:
    def __init__(self, spec: FusionSpec, registry: PluginRegistry | None = None) -> None:
        self.spec = spec
        self.registry = registry or PluginRegistry()
        self.registry.register("providers", "openai-compatible", OpenAICompatibleProvider)
        self.registry.register("providers", "llama.cpp", OpenAICompatibleProvider)
        self.registry.register("providers", "anthropic-compatible", AnthropicCompatibleProvider)
        self.registry.register("policies", "direct", lambda _spec: DirectPolicy())
        self.registry.register("policies", "main-critic", MainCriticPolicy)
        self.registry.register("policies", "review-board", ReviewBoardPolicy)
        self.registry.discover()
        self._providers = {
            name: self.registry.create("providers", provider.type, provider)
            for name, provider in spec.providers.items()
        }
        self._limits = {
            name: asyncio.Semaphore(model.max_concurrency) for name, model in spec.models.items()
        }
        self._policy = self.registry.create("policies", spec.policy.type, spec.policy)

    def trace_id(self) -> str:
        return _trace_id.get()

    def _validate_request(self, name: str, request: FusionRequest) -> None:
        model = self.spec.models[name]
        if request.tools and not model.tool_calling:
            raise CapabilityError(f"model {name!r} does not declare tool_calling")
        if request.reasoning_effort is not None and not model.reasoning:
            raise CapabilityError(f"model {name!r} does not declare reasoning")
        provider_type = self.spec.providers[model.provider].type
        if request.reasoning_effort is not None and provider_type == "anthropic-compatible":
            raise CapabilityError(
                "reasoning_effort is not portable to the built-in anthropic-compatible provider"
            )

    async def call_model(self, name: str, request: FusionRequest) -> ModelResponse:
        self._validate_request(name, request)
        model = self.spec.models[name]
        async with self._limits[name]:
            return await self._providers[model.provider].complete(model, request)

    async def stream_model(
        self, name: str, request: FusionRequest
    ) -> AsyncIterator[ModelStreamEvent]:
        self._validate_request(name, request)
        model = self.spec.models[name]
        provider = self._providers[model.provider]
        stream = getattr(provider, "stream", None)
        if stream is None:
            raise CapabilityError(
                f"provider {model.provider!r} does not implement native final-model streaming"
            )
        async with self._limits[name]:
            events = stream(model, request)
            if inspect.isawaitable(events):
                events = await events
            try:
                async for event in events:
                    yield event
            finally:
                close = getattr(events, "aclose", None)
                if close is not None:
                    await close()

    async def complete(self, request: FusionRequest, pool: str | None = None) -> FusionResult:
        token = _trace_id.set(uuid.uuid4().hex)
        try:
            prepared = await self._policy.prepare(self, pool or self.spec.serve.pool, request)
            response = await self.call_model(prepared.model_name, prepared.request)
            return FusionResult(
                response=response,
                route=prepared.route,
                experts_used=prepared.experts_used,
                fallback_reason=prepared.fallback_reason,
                trace_id=self.trace_id(),
            )
        finally:
            _trace_id.reset(token)

    async def stream(self, request: FusionRequest, pool: str | None = None) -> FusionStream:
        token = _trace_id.set(uuid.uuid4().hex)
        try:
            prepared = await self._policy.prepare(self, pool or self.spec.serve.pool, request)
            self._validate_request(prepared.model_name, prepared.request)
            provider = self._providers[self.spec.models[prepared.model_name].provider]
            if not hasattr(provider, "stream"):
                raise CapabilityError(
                    "selected provider does not implement native final-model streaming"
                )
            events = self.stream_model(prepared.model_name, prepared.request)
            try:
                first = await anext(events)
            except StopAsyncIteration as exc:
                await events.aclose()
                raise ProviderProtocolError("upstream provider returned an empty stream") from exc
            except BaseException:
                await events.aclose()
                raise
            return FusionStream(
                events=_PrefetchedStream(first, events),
                route=prepared.route,
                experts_used=prepared.experts_used,
                fallback_reason=prepared.fallback_reason,
                trace_id=self.trace_id(),
            )
        finally:
            _trace_id.reset(token)

    async def aclose(self) -> None:
        seen: set[int] = set()
        for provider in self._providers.values():
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            close = getattr(provider, "aclose", None)
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await result


class _PrefetchedStream:
    def __init__(self, first: ModelStreamEvent, events: AsyncIterator[ModelStreamEvent]) -> None:
        self._first = first
        self._events = events
        self._first_pending = True
        self._closed = False

    def __aiter__(self) -> _PrefetchedStream:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if self._closed:
            raise StopAsyncIteration
        if self._first_pending:
            self._first_pending = False
            return self._first
        return await anext(self._events)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._events.aclose()
