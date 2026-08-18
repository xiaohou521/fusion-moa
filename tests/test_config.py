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
