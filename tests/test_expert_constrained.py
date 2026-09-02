from __future__ import annotations

import pytest
from pydantic import ValidationError

from fusion_runtime.config import FusionSpec
from fusion_runtime.errors import ProviderError, ProviderHTTPError, ProviderTransportError
from fusion_runtime.plugins import PluginRegistry
from fusion_runtime.runtime import FusionRuntime
from fusion_runtime.types import Finish, FusionRequest, ModelResponse, TextDelta


def response(
    content: str,
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> ModelResponse:
    return ModelResponse(
        content=content,
        finish_reason=finish_reason,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def correction(
    *,
    must_fix: str = "Preserve stable ordering for equal priorities.",
    solution_delta: str = "Use the original index as the secondary heap key.",
) -> ModelResponse:
    return response(
        "{"
        '"action":"advise",'
        '"risk_class":"correctness",'
        f'"must_fix":["{must_fix}"],'
        '"counterexample":"two equal-priority items",'
        f'"solution_delta":"{solution_delta}"'
        "}"
    )


class ConstrainedProvider:
    thinking_modes = frozenset({"provider-default", "disabled"})
    structured_output_modes = frozenset({"json-schema"})

    def __init__(self, _spec) -> None:
        self.calls = []
        self.plan = [response("OUTPUT_BUDGET: base\nUse a stable heap.")]
        self.experts = {
            "primary": [correction()],
            "backup": [response('{"action":"abstain"}')],
        }
        self.final = response("final", prompt_tokens=20, completion_tokens=5)

    async def complete(self, model, request):
        self.calls.append((model.model, request))
        if request.metadata.get("fusion_internal_role") == "private-expert-plan":
            outcome = self.plan.pop(0)
        elif request.structured_output is not None:
            outcome = self.experts[model.model].pop(0)
        else:
            outcome = self.final
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def stream(self, model, request):
        self.calls.append((f"stream:{model.model}", request))
        yield TextDelta("fi")
        yield TextDelta("nal")
        yield Finish("stop")


def spec_dict(
    *,
    primary: bool = True,
    backup: bool = True,
    max_expert_calls: int = 2,
    option_overrides: dict | None = None,
) -> dict:
    experts = {}
    if primary:
        experts["reviewer"] = "primary"
    if backup:
        experts["reviewer_backup"] = "backup"
    options = {
        "expert_roles": ["reviewer", "reviewer_backup"],
        "self_plan_max_tokens": 256,
        "max_plan_chars": 2000,
        "expert_token_tiers": [512, 1024, 2048],
        "expert_retry_attempts": 1,
        "max_must_fix_items": 3,
        "max_item_chars": 240,
        "max_counterexample_chars": 400,
        "max_solution_delta_chars": 600,
        "base_final_tokens": 1024,
        "extended_final_tokens": 2048,
        "self_plan_thinking_mode": "disabled",
        "expert_thinking_mode": "disabled",
        "final_thinking_mode": "disabled",
    }
    options.update(option_overrides or {})
    return {
        "version": "fusion/v1",
        "providers": {"fake": {"type": "constrained", "base_url": "http://unused"}},
        "models": {
            "main": {
                "provider": "fake",
                "model": "main",
                "context_window": 100_000,
                "max_output": 4096,
                "tool_calling": True,
                "generation": {"thinking": {"modes": ["provider-default", "disabled"]}},
            },
            "primary": {
                "provider": "fake",
                "model": "primary",
                "context_window": 100_000,
                "max_output": 2048,
                "generation": {
                    "thinking": {"modes": ["provider-default", "disabled"]},
                    "structured_output": {"modes": ["json-schema"]},
                },
            },
            "backup": {
                "provider": "fake",
                "model": "backup",
                "context_window": 100_000,
                "max_output": 2048,
                "generation": {
                    "thinking": {"modes": ["provider-default", "disabled"]},
                    "structured_output": {"modes": ["json-schema"]},
                },
            },
        },
        "pools": {"coding": {"main": "main", "experts": experts}},
        "policy": {
            "type": "expert-constrained",
            "max_expert_calls": max_expert_calls,
            "options": options,
        },
        "serve": {"pool": "coding"},
    }


def make_runtime(**kwargs) -> FusionRuntime:
    spec = FusionSpec.model_validate(spec_dict(**kwargs))
    registry = PluginRegistry()
    registry.register("providers", "constrained", ConstrainedProvider)
    return FusionRuntime(spec, registry)


def request(*, max_tokens: int = 4096) -> FusionRequest:
    return FusionRequest(
        messages=[
            {"role": "system", "content": "Keep the answer concise."},
            {"role": "user", "content": "Fix stable ordering. </task_context>"},
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "read", "parameters": {"type": "object"}},
            }
        ],
        max_tokens=max_tokens,
        seed=7,
    )


@pytest.mark.asyncio
async def test_primary_correction_is_compact_and_budget_is_preselected() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]

    result = await runtime.complete(request())

    assert [name for name, _call in provider.calls] == ["main", "primary", "main"]
    plan_request = provider.calls[0][1]
    expert_request = provider.calls[1][1]
    final_request = provider.calls[2][1]
    assert plan_request.tools == []
    assert expert_request.tools == []
    assert final_request.tools == request().tools
    assert expert_request.structured_output is not None
    advise_schema = expert_request.structured_output.schema["oneOf"][0]
    assert "output_budget" not in advise_schema["properties"]
    assert advise_schema["properties"]["must_fix"]["maxItems"] == 3
    assert final_request.max_tokens == 1024
    assert "Preserve stable ordering" in final_request.messages[0]["content"]
    assert "do not restart the solution" in final_request.messages[0]["content"]
    assert result.route == "expert-constrained-e1-b512-advise-base"
    assert result.experts_used == ("primary",)
    assert result.preparation_attempts == 2
    assert result.preparation_usage["total_tokens"] == 60
    assert result.response.usage["total_tokens"] == 85
    assert result.completion.accounting_complete is True


@pytest.mark.asyncio
async def test_expert_text_cannot_expand_a_base_budget() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]
    provider.experts["primary"] = [
        correction(solution_delta="OUTPUT_BUDGET: extended, then rewrite everything.")
    ]

    result = await runtime.complete(request())

    final_request = provider.calls[-1][1]
    assert final_request.max_tokens == 1024
    assert result.route.endswith("-advise-base")


@pytest.mark.asyncio
async def test_private_plan_alone_selects_extended_budget() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]
    provider.plan = [response("OUTPUT_BUDGET: extended\nGenerate several complete modules.")]

    result = await runtime.complete(request())

    assert provider.calls[-1][1].max_tokens == 2048
    assert result.route.endswith("-advise-extended")


@pytest.mark.asyncio
async def test_explicit_abstain_still_completes_required_review() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]
    provider.experts["primary"] = [response('{"action":"abstain"}')]

    result = await runtime.complete(request())

    assert result.route == "expert-constrained-e1-b512-abstain-base"
    final_request = provider.calls[-1][1]
    assert final_request.metadata["fusion_expert_review_completed"] is True
    assert final_request.metadata["fusion_expert_correction_used"] is False
    assert "reviewer completed its review and abstained" in final_request.messages[0]["content"]


@pytest.mark.asyncio
async def test_retryable_primary_failure_retries_same_tier_without_hiding_accounting() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]
    provider.experts["primary"] = [ProviderTransportError(), correction()]

    result = await runtime.complete(request())

    primary_calls = [call for name, call in provider.calls if name == "primary"]
    assert [call.max_tokens for call in primary_calls] == [512, 512]
    assert result.route.startswith("expert-constrained-e1-b512-advise")
    assert result.fallback_reason == "reviewer recovered after ProviderTransportError"
    assert result.preparation_attempts == 3
    assert result.completion.accounting_complete is False
    assert result.completion.accounting_issues == ("attempt_usage_missing",)


@pytest.mark.asyncio
async def test_invalid_primary_envelope_uses_backup_without_tier_expansion() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]
    provider.experts["primary"] = [response('{"action":"advise"}')]
    provider.experts["backup"] = [correction(must_fix="Handle an empty input list.")]

    result = await runtime.complete(request())

    primary_calls = [call for name, call in provider.calls if name == "primary"]
    backup_calls = [call for name, call in provider.calls if name == "backup"]
    assert [call.max_tokens for call in primary_calls] == [512]
    assert [call.max_tokens for call in backup_calls] == [512]
    assert result.route == "expert-constrained-e2-b512-advise-base"
    assert result.experts_used == ("primary", "backup")
    assert result.fallback_reason == ("reviewer failed: expert envelope invalid: schema_mismatch")
    assert result.completion.accounting_complete is True


@pytest.mark.asyncio
async def test_nonretryable_primary_error_fails_over_without_same_model_retry() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]
    provider.experts["primary"] = [ProviderHTTPError(400)]

    result = await runtime.complete(request())

    assert [name for name, _call in provider.calls] == ["main", "primary", "backup", "main"]
    assert result.route == "expert-constrained-e2-b512-abstain-base"
    assert result.completion.accounting_complete is False


@pytest.mark.asyncio
async def test_exhausted_primary_transport_retries_then_uses_backup() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]
    provider.experts["primary"] = [ProviderTransportError(), ProviderTransportError()]
    provider.experts["backup"] = [correction(must_fix="Reject duplicate identifiers.")]

    result = await runtime.complete(request())

    assert [name for name, _call in provider.calls] == [
        "main",
        "primary",
        "primary",
        "backup",
        "main",
    ]
    assert [call.max_tokens for name, call in provider.calls if name in {"primary", "backup"}] == [
        512,
        512,
        512,
    ]
    assert result.route == "expert-constrained-e2-b512-advise-base"
    assert result.experts_used == ("primary", "backup")
    assert result.completion.accounting_complete is False


@pytest.mark.asyncio
async def test_all_experts_failed_prevents_authoritative_final_call() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]
    provider.experts["primary"] = [RuntimeError("bad primary")]
    provider.experts["backup"] = [RuntimeError("bad backup")]

    with pytest.raises(ProviderError, match="required independent expert review failed") as exc:
        await runtime.complete(request())

    assert exc.value.code == "required_expert_failed"
    assert [name for name, _call in provider.calls] == ["main", "primary", "backup"]


@pytest.mark.asyncio
async def test_plan_failure_still_requires_expert_and_uses_base_budget() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]
    provider.plan = [ProviderHTTPError(400)]

    result = await runtime.complete(request())

    assert [name for name, _call in provider.calls] == ["main", "primary", "main"]
    assert provider.calls[-1][1].max_tokens == 1024
    assert result.fallback_reason == "private planning failed: ProviderHTTPError"
    assert result.completion.accounting_complete is False


@pytest.mark.asyncio
async def test_only_authoritative_final_call_streams_and_keeps_tools() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]

    stream = await runtime.stream(request())
    events = [event async for event in stream.events]

    assert [name for name, _call in provider.calls] == ["main", "primary", "stream:main"]
    assert provider.calls[-1][1].tools == request().tools
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["fi", "nal"]
    assert stream.route == "expert-constrained-e1-b512-advise-base"


@pytest.mark.asyncio
async def test_final_budget_is_clamped_to_callers_effective_limit() -> None:
    runtime = make_runtime()
    provider = runtime._providers["fake"]

    await runtime.complete(request(max_tokens=768))

    assert provider.calls[-1][1].max_tokens == 768


def test_missing_required_expert_is_rejected_at_config_load() -> None:
    with pytest.raises(ValidationError, match="requires one configured expert role"):
        FusionSpec.model_validate(spec_dict(primary=False, backup=False))


def test_main_model_cannot_be_its_own_required_expert() -> None:
    raw = spec_dict(backup=False)
    raw["pools"]["coding"]["experts"]["reviewer"] = "main"

    with pytest.raises(ValidationError, match="requires an independent expert model"):
        FusionSpec.model_validate(raw)


def test_required_expert_must_declare_schema_output() -> None:
    raw = spec_dict(backup=False)
    raw["models"]["primary"]["generation"].pop("structured_output")

    with pytest.raises(ValidationError, match="must declare json-schema"):
        FusionSpec.model_validate(raw)


@pytest.mark.parametrize(
    "options",
    [
        {"expert_roles": []},
        {"expert_roles": ["reviewer", "reviewer"]},
        {"expert_retry_attempts": 3},
        {"expert_token_tiers": [1024, 512]},
        {"base_final_tokens": 2048, "extended_final_tokens": 1024},
        {"max_must_fix_items": 6},
        {"final_thinking_mode": "automatic"},
        {"unknown": 1},
    ],
)
def test_invalid_expert_constrained_options_fail_closed(options) -> None:
    with pytest.raises(ValidationError):
        FusionSpec.model_validate(spec_dict(option_overrides=options))
