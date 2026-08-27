from pathlib import Path

import pytest
from pydantic import ValidationError

from fusion_runtime.config import FusionSpec, load_spec


def test_example_recipe_is_valid() -> None:
    spec = load_spec(Path(__file__).parents[1] / "recipes/local-main-critic.yaml")
    assert spec.serve.model_name == "fusion-coding"
    assert spec.pools["coding"].experts["critic"] == "critic"


def test_all_shipped_recipes_are_valid() -> None:
    recipe_dir = Path(__file__).parents[1] / "recipes"
    for recipe in recipe_dir.glob("*.yaml"):
        assert load_spec(recipe).version == "fusion/v1"


def test_unknown_model_reference_fails_closed() -> None:
    with pytest.raises(ValidationError, match="unknown models"):
        FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {
                    "p": {"type": "openai-compatible", "base_url": "http://localhost/v1"}
                },
                "models": {"main": {"provider": "p", "model": "m", "context_window": 10}},
                "pools": {"coding": {"main": "missing"}},
                "serve": {"pool": "coding"},
            }
        )


def test_literal_credential_header_is_rejected() -> None:
    with pytest.raises(ValidationError, match="credential headers"):
        FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {
                    "p": {
                        "type": "openai-compatible",
                        "base_url": "http://localhost/v1",
                        "headers": {"Authorization": "Bearer secret"},
                    }
                },
                "models": {"main": {"provider": "p", "model": "m", "context_window": 10}},
                "pools": {"coding": {"main": "main"}},
                "serve": {"pool": "coding"},
            }
        )


def test_unknown_protocol_is_rejected() -> None:
    with pytest.raises(ValidationError, match="openai-chat"):
        FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {
                    "p": {"type": "openai-compatible", "base_url": "http://localhost/v1"}
                },
                "models": {"main": {"provider": "p", "model": "m", "context_window": 10}},
                "pools": {"coding": {"main": "main"}},
                "serve": {"pool": "coding", "protocols": ["made-up"]},
            }
        )


@pytest.mark.parametrize(
    "base_url",
    ["localhost:8000/v1", "file:///tmp/model.sock", "https://user:pass@example.test/v1"],
)
def test_unsafe_or_ambiguous_provider_url_is_rejected(base_url) -> None:
    with pytest.raises(ValidationError, match="base_url"):
        FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {"p": {"type": "openai-compatible", "base_url": base_url}},
                "models": {"main": {"provider": "p", "model": "m", "context_window": 10}},
                "pools": {"coding": {"main": "main"}},
                "serve": {"pool": "coding"},
            }
        )


def test_invalid_advice_budget_is_rejected_at_load_time() -> None:
    with pytest.raises(ValidationError, match="max_advice_chars"):
        FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {
                    "p": {"type": "openai-compatible", "base_url": "http://localhost/v1"}
                },
                "models": {"main": {"provider": "p", "model": "m", "context_window": 10}},
                "pools": {"coding": {"main": "main"}},
                "policy": {"type": "review-board", "options": {"max_advice_chars": True}},
                "serve": {"pool": "coding"},
            }
        )


def test_invalid_expert_token_budget_is_rejected_at_load_time() -> None:
    with pytest.raises(ValidationError, match="expert_max_tokens"):
        FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {
                    "p": {"type": "openai-compatible", "base_url": "http://localhost/v1"}
                },
                "models": {"main": {"provider": "p", "model": "m", "context_window": 10}},
                "pools": {"coding": {"main": "main"}},
                "policy": {"type": "review-board", "options": {"expert_max_tokens": 0}},
                "serve": {"pool": "coding"},
            }
        )


@pytest.mark.parametrize(
    "options",
    [
        {"expert_token_tiers": []},
        {"expert_token_tiers": [512, 512]},
        {"expert_token_tiers": [1024, 512]},
        {"expert_token_tiers": [512, 0]},
        {"expert_temperature": True},
        {"expert_temperature": 2.1},
        {"expert_thinking_mode": "automatic"},
        {"expert_role": ""},
        {"unknown": 1},
    ],
)
def test_invalid_adaptive_self_review_options_fail_closed(options) -> None:
    with pytest.raises(ValidationError):
        FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {
                    "p": {"type": "openai-compatible", "base_url": "http://localhost/v1"}
                },
                "models": {"main": {"provider": "p", "model": "m", "context_window": 10}},
                "pools": {"coding": {"main": "main"}},
                "policy": {
                    "type": "adaptive-self-review",
                    "max_expert_calls": 1,
                    "options": options,
                },
                "serve": {"pool": "coding"},
            }
        )


def test_generation_capabilities_are_explicit_and_strict() -> None:
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {"p": {"type": "openai-compatible", "base_url": "http://localhost/v1"}},
            "models": {
                "main": {
                    "provider": "p",
                    "model": "m",
                    "context_window": 100,
                    "generation": {
                        "thinking": {"modes": ["provider-default", "disabled", "bounded"]},
                        "final_answer_reserve": True,
                    },
                }
            },
            "pools": {"coding": {"main": "main"}},
            "serve": {"pool": "coding"},
        }
    )

    generation = spec.models["main"].generation
    assert generation.thinking.modes == {"provider-default", "disabled", "bounded"}
    assert generation.final_answer_reserve is True


@pytest.mark.parametrize(
    "thinking",
    [
        {"modes": []},
        {"modes": ["automatic"]},
        {"modes": ["provider-default"], "enable_thinking": False},
    ],
)
def test_invalid_generation_capabilities_fail_closed(thinking) -> None:
    with pytest.raises(ValidationError):
        FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {
                    "p": {
                        "type": "openai-compatible",
                        "base_url": "http://localhost/v1",
                    }
                },
                "models": {
                    "main": {
                        "provider": "p",
                        "model": "m",
                        "context_window": 100,
                        "generation": {"thinking": thinking},
                    }
                },
                "pools": {"coding": {"main": "main"}},
                "serve": {"pool": "coding"},
            }
        )


def test_empty_output_recovery_is_bounded_and_opt_in() -> None:
    spec = FusionSpec.model_validate(
        {
            "version": "fusion/v1",
            "providers": {"p": {"type": "openai-compatible", "base_url": "http://localhost/v1"}},
            "models": {"main": {"provider": "p", "model": "m", "context_window": 100}},
            "pools": {"coding": {"main": "main"}},
            "completion": {
                "require_public_output": True,
                "require_tool_or_text": True,
                "max_recovery_attempts": 1,
                "recovery_max_tokens": 512,
            },
            "serve": {"pool": "coding"},
        }
    )

    assert spec.completion.max_recovery_attempts == 1
    assert spec.completion.recovery_max_tokens == 512


@pytest.mark.parametrize(
    "completion",
    [
        {"max_recovery_attempts": 2},
        {"max_recovery_attempts": 1, "recovery_max_tokens": 0},
        {
            "max_recovery_attempts": 1,
            "require_public_output": False,
            "require_tool_or_text": False,
        },
        {"max_recovery_attempts": 1, "retry_forever": True},
    ],
)
def test_invalid_completion_policy_fails_closed(completion) -> None:
    with pytest.raises(ValidationError):
        FusionSpec.model_validate(
            {
                "version": "fusion/v1",
                "providers": {
                    "p": {
                        "type": "openai-compatible",
                        "base_url": "http://localhost/v1",
                    }
                },
                "models": {"main": {"provider": "p", "model": "m", "context_window": 100}},
                "pools": {"coding": {"main": "main"}},
                "completion": completion,
                "serve": {"pool": "coding"},
            }
        )
