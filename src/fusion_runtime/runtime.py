from __future__ import annotations

import asyncio
import contextvars
import inspect
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace

from .accounting import assess_usage
from .completion import CompletionTracker, classify_response
from .config import FusionSpec
from .errors import CapabilityError, ProviderError, ProviderProtocolError
from .plugins import PluginRegistry
from .policies import (
    AdaptiveReasoningReservePolicy,
    DirectPolicy,
    MainCriticPolicy,
    ReasoningReservePolicy,
    ReviewBoardPolicy,
)
from .providers import AnthropicCompatibleProvider, OpenAICompatibleProvider
from .recovery import merge_usage, prepare_recovery_request, requires_recovery
from .types import (
    THINKING_MODES,
    CompletionOutcome,
    CompletionRecord,
    FusionRequest,
    FusionResult,
    FusionStream,
    ModelResponse,
    ModelStreamEvent,
    RecoveryOutcome,
    RecoveryRecord,
    StreamError,
    TextDelta,
    Usage,
)

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


class FusionRuntime:
    def __init__(self, spec: FusionSpec, registry: PluginRegistry | None = None) -> None:
        self.spec = spec
        self.registry = registry or PluginRegistry()
        self.registry.register("providers", "openai-compatible", OpenAICompatibleProvider)
        self.registry.register("providers", "llama.cpp", OpenAICompatibleProvider)
        self.registry.register("providers", "anthropic-compatible", AnthropicCompatibleProvider)
        self.registry.register("policies", "direct", lambda _spec: DirectPolicy())
        self.registry.register("policies", "reasoning-reserve", ReasoningReservePolicy)
        self.registry.register(
            "policies", "adaptive-reasoning-reserve", AdaptiveReasoningReservePolicy
        )
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
        if request.seed is not None and provider_type == "anthropic-compatible":
            raise CapabilityError(
                "seed is not portable to the built-in anthropic-compatible provider"
            )
        thinking = request.thinking
        if thinking.mode not in model.generation.thinking.modes:
            raise CapabilityError(
                f"model {name!r} does not declare thinking mode {thinking.mode!r}"
            )
        if thinking.mode == "bounded" and thinking.budget_tokens is not None:
            requested_limit = (
                request.max_tokens if request.max_tokens is not None else model.max_output
            )
            output_limit = min(requested_limit, model.max_output)
            if thinking.budget_tokens > output_limit:
                raise CapabilityError(
                    f"thinking budget {thinking.budget_tokens} exceeds output limit "
                    f"{output_limit} for model {name!r}"
                )
        provider = self._providers[model.provider]
        provider_modes = getattr(provider, "thinking_modes", frozenset({"provider-default"}))
        if (
            not isinstance(provider_modes, (set, frozenset))
            or not provider_modes
            or not provider_modes <= THINKING_MODES
        ):
            raise CapabilityError(
                f"provider {model.provider!r} has an invalid thinking_modes declaration"
            )
        if thinking.mode not in provider_modes:
            raise CapabilityError(
                f"provider {model.provider!r} does not map thinking mode {thinking.mode!r}"
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
            initial_attempt = await self.call_model(prepared.model_name, prepared.request)
            initial_attempt_usage_reported = bool(initial_attempt.usage)
            initial_response = replace(
                initial_attempt,
                usage=merge_usage(prepared.preparation_usage, initial_attempt.usage),
            )
            initial_completion = _classify_attempts(
                initial_response,
                all_usage_reported=(
                    prepared.preparation_usage_complete and initial_attempt_usage_reported
                ),
            )
            completion_spec = self.spec.completion
            if completion_spec.max_recovery_attempts == 0 or not requires_recovery(
                initial_completion, completion_spec
            ):
                return FusionResult(
                    response=initial_response,
                    route=prepared.route,
                    experts_used=prepared.experts_used,
                    fallback_reason=prepared.fallback_reason,
                    trace_id=self.trace_id(),
                    completion=initial_completion,
                    preparation_attempts=prepared.preparation_attempts,
                    preparation_duration_ms=prepared.preparation_duration_ms,
                    preparation_usage=dict(prepared.preparation_usage),
                )

            model = self.spec.models[prepared.model_name]
            recovery_request = prepare_recovery_request(
                prepared.request,
                model,
                completion_spec,
                attempt=1,
            )
            started = time.perf_counter()
            try:
                recovery_response = await self.call_model(prepared.model_name, recovery_request)
            except ProviderError as exc:
                duration_ms = (time.perf_counter() - started) * 1000
                usage_reported, accounting_issues = assess_usage(
                    initial_response.usage,
                    report_seen=bool(initial_response.usage),
                    reported_for_all_attempts=False,
                )
                return FusionResult(
                    response=initial_response,
                    route=prepared.route,
                    experts_used=prepared.experts_used,
                    fallback_reason=prepared.fallback_reason,
                    trace_id=self.trace_id(),
                    completion=replace(
                        initial_completion,
                        usage_reported=usage_reported,
                        accounting_complete=not accounting_issues,
                        accounting_issues=accounting_issues,
                    ),
                    recovery=RecoveryOutcome(
                        attempts=1,
                        duration_ms=duration_ms,
                        failure_code=exc.code,
                        initial_completion=initial_completion,
                    ),
                    preparation_attempts=prepared.preparation_attempts,
                    preparation_duration_ms=prepared.preparation_duration_ms,
                    preparation_usage=dict(prepared.preparation_usage),
                )

            duration_ms = (time.perf_counter() - started) * 1000
            combined_usage = merge_usage(initial_response.usage, recovery_response.usage)
            final_response = replace(recovery_response, usage=combined_usage)
            final_completion = classify_response(final_response)
            all_usage_reported = (
                prepared.preparation_usage_complete
                and initial_attempt_usage_reported
                and bool(recovery_response.usage)
            )
            usage_reported, accounting_issues = assess_usage(
                combined_usage,
                report_seen=bool(combined_usage),
                reported_for_all_attempts=all_usage_reported,
            )
            final_completion = replace(
                final_completion,
                usage_reported=usage_reported,
                accounting_complete=not accounting_issues,
                accounting_issues=accounting_issues,
            )
            succeeded = not requires_recovery(final_completion, completion_spec)
            failure_code = None
            if not succeeded:
                failure_code = (
                    final_completion.failure_tags[0]
                    if final_completion.failure_tags
                    else "recovery_output_missing"
                )
            return FusionResult(
                response=final_response,
                route=prepared.route,
                experts_used=prepared.experts_used,
                fallback_reason=prepared.fallback_reason,
                trace_id=self.trace_id(),
                completion=final_completion,
                recovery=RecoveryOutcome(
                    attempts=1,
                    succeeded=succeeded,
                    duration_ms=duration_ms,
                    failure_code=failure_code,
                    usage=dict(recovery_response.usage),
                    initial_completion=initial_completion,
                ),
                preparation_attempts=prepared.preparation_attempts,
                preparation_duration_ms=prepared.preparation_duration_ms,
                preparation_usage=dict(prepared.preparation_usage),
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
            recovery = RecoveryRecord()
            events = self._stream_with_recovery(
                prepared.model_name,
                prepared.request,
                recovery,
                preparation_usage=prepared.preparation_usage,
                preparation_usage_complete=prepared.preparation_usage_complete,
            )
            try:
                first = await anext(events)
            except StopAsyncIteration as exc:
                await events.aclose()
                raise ProviderProtocolError("upstream provider returned an empty stream") from exc
            except BaseException:
                await events.aclose()
                raise
            completion = CompletionRecord()
            observed_events = _ObservedStream(
                _PrefetchedStream(first, events), CompletionTracker(completion)
            )
            return FusionStream(
                events=observed_events,
                route=prepared.route,
                experts_used=prepared.experts_used,
                fallback_reason=prepared.fallback_reason,
                trace_id=self.trace_id(),
                completion=completion,
                recovery=recovery,
                preparation_attempts=prepared.preparation_attempts,
                preparation_duration_ms=prepared.preparation_duration_ms,
                preparation_usage=dict(prepared.preparation_usage),
            )
        finally:
            _trace_id.reset(token)

    async def _stream_with_recovery(
        self,
        model_name: str,
        request: FusionRequest,
        recovery: RecoveryRecord,
        *,
        preparation_usage: dict[str, object],
        preparation_usage_complete: bool,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Keep the final answer native-streamed while hiding one empty attempt."""

        initial = _BufferedAttempt()
        initial_error: ProviderProtocolError | None = None
        initial_events = self.stream_model(model_name, request)
        try:
            try:
                async for event in initial_events:
                    for ready in initial.observe(event):
                        yield ready
            except ProviderProtocolError as exc:
                initial_error = exc
        finally:
            await initial_events.aclose()
        initial.finish()

        if initial.committed:
            if initial_error is not None:
                yield StreamError(str(initial_error), code=initial_error.code)
                return
            combined_usage = merge_usage(preparation_usage, initial.usage)
            if combined_usage:
                yield Usage(
                    combined_usage,
                    reported_for_all_attempts=(
                        preparation_usage_complete
                        and initial.usage_seen
                        and initial.all_usage_reported
                    ),
                )
            return

        completion_spec = self.spec.completion
        if completion_spec.max_recovery_attempts == 0 or not requires_recovery(
            initial.outcome, completion_spec
        ):
            if initial_error is not None:
                raise initial_error
            for event in initial.release_pending():
                yield event
            combined_usage = merge_usage(preparation_usage, initial.usage)
            if combined_usage:
                yield Usage(
                    combined_usage,
                    reported_for_all_attempts=(
                        preparation_usage_complete
                        and initial.usage_seen
                        and initial.all_usage_reported
                    ),
                )
            return

        model = self.spec.models[model_name]
        recovery_request = prepare_recovery_request(
            request,
            model,
            completion_spec,
            attempt=1,
        )
        started = time.perf_counter()
        recovery.outcome = RecoveryOutcome(
            attempts=1,
            initial_completion=initial.outcome,
        )
        retried = _BufferedAttempt()
        recovery_error: ProviderError | None = None
        recovery_events = self.stream_model(model_name, recovery_request)
        try:
            try:
                async for event in recovery_events:
                    ready_events = retried.observe(event)
                    if ready_events and retried.committed:
                        recovery.outcome = RecoveryOutcome(
                            attempts=1,
                            succeeded=True,
                            duration_ms=(time.perf_counter() - started) * 1000,
                            usage=dict(retried.usage),
                            initial_completion=initial.outcome,
                        )
                    for ready in ready_events:
                        yield ready
            except ProviderError as exc:
                recovery_error = exc
        finally:
            await recovery_events.aclose()
        retried.finish()

        if recovery_error is not None:
            recovery.outcome = RecoveryOutcome(
                attempts=1,
                succeeded=retried.committed,
                duration_ms=(time.perf_counter() - started) * 1000,
                failure_code=None if retried.committed else recovery_error.code,
                usage=dict(retried.usage),
                initial_completion=initial.outcome,
            )
            if retried.committed:
                yield StreamError(
                    str(recovery_error),
                    code=recovery_error.code,
                    retryable=recovery_error.retryable,
                )
                return
            raise recovery_error

        if not retried.committed:
            for event in retried.release_pending():
                yield event

        succeeded = retried.committed and not requires_recovery(retried.outcome, completion_spec)
        failure_code = None if succeeded else retried.failure_code
        recovery.outcome = RecoveryOutcome(
            attempts=1,
            succeeded=succeeded,
            duration_ms=(time.perf_counter() - started) * 1000,
            failure_code=failure_code,
            usage=dict(retried.usage),
            initial_completion=initial.outcome,
        )
        combined_usage = merge_usage(preparation_usage, initial.usage, retried.usage)
        if combined_usage:
            yield Usage(
                combined_usage,
                reported_for_all_attempts=(
                    preparation_usage_complete
                    and initial.usage_seen
                    and initial.all_usage_reported
                    and retried.usage_seen
                    and retried.all_usage_reported
                ),
            )

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


def _classify_attempts(
    response: ModelResponse,
    *,
    all_usage_reported: bool,
) -> CompletionOutcome:
    outcome = classify_response(response)
    usage_reported, accounting_issues = assess_usage(
        response.usage,
        report_seen=bool(response.usage),
        reported_for_all_attempts=all_usage_reported,
    )
    return replace(
        outcome,
        usage_reported=usage_reported,
        accounting_complete=not accounting_issues,
        accounting_issues=accounting_issues,
    )


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


class _ObservedStream:
    def __init__(self, events: _PrefetchedStream, tracker: CompletionTracker) -> None:
        self._events = events
        self._tracker = tracker
        self._closed = False

    def __aiter__(self) -> _ObservedStream:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if self._closed:
            raise StopAsyncIteration
        try:
            event = await anext(self._events)
        except StopAsyncIteration:
            self._tracker.end()
            raise
        except asyncio.CancelledError:
            self._tracker.cancel()
            raise
        except BaseException:
            self._tracker.end()
            raise
        self._tracker.observe(event)
        return event

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._tracker.cancel()
        await self._events.aclose()


class _BufferedAttempt:
    """Buffer only the events that cannot yet be exposed exactly once."""

    def __init__(self) -> None:
        self._record = CompletionRecord()
        self._tracker = CompletionTracker(self._record)
        self._pending: list[ModelStreamEvent] = []
        self.usage: dict[str, object] = {}
        self.usage_seen = False
        self.all_usage_reported = True
        self.committed = False
        self._stream_error_code: str | None = None

    @property
    def outcome(self):
        return self._record.outcome

    @property
    def failure_code(self) -> str:
        if self._stream_error_code:
            return self._stream_error_code
        if self.outcome.failure_tags:
            return self.outcome.failure_tags[0]
        return "recovery_output_missing"

    def observe(self, event: ModelStreamEvent) -> list[ModelStreamEvent]:
        self._tracker.observe(event)
        if isinstance(event, Usage):
            self.usage_seen = True
            self.all_usage_reported = self.all_usage_reported and event.reported_for_all_attempts
            self.usage.update(event.usage)
            return []
        if isinstance(event, StreamError):
            self._stream_error_code = event.code
        if self.committed:
            return [event]
        self._pending.append(event)
        if isinstance(event, TextDelta) and self.outcome.has_text_output:
            return self._commit()
        if self._tracker.terminal_seen and self.outcome.has_valid_tool_call:
            return self._commit()
        return []

    def finish(self) -> None:
        self._tracker.end()

    def release_pending(self) -> list[ModelStreamEvent]:
        pending = self._pending
        self._pending = []
        return pending

    def _commit(self) -> list[ModelStreamEvent]:
        self.committed = True
        return self.release_pending()
