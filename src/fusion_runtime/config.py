from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .types import ThinkingMode

ProtocolName = Literal["openai-chat", "openai-responses", "anthropic-messages"]
_ENV_NAME = r"^[A-Za-z_][A-Za-z0-9_]*$"
_SECRET_HEADERS = {"authorization", "proxy-authorization", "x-api-key", "api-key"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderSpec(StrictModel):
    type: str
    base_url: str
    api_key_env: str | None = Field(default=None, pattern=_ENV_NAME)
    timeout_seconds: float = Field(default=120, gt=0)
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain credentials")
        if parsed.fragment:
            raise ValueError("base_url must not contain a fragment")
        return value

    @field_validator("headers")
    @classmethod
    def reject_literal_credentials(cls, headers: dict[str, str]) -> dict[str, str]:
        forbidden = sorted(name for name in headers if name.lower() in _SECRET_HEADERS)
        if forbidden:
            raise ValueError(
                "credential headers must use api_key_env, not literal headers: "
                + ", ".join(forbidden)
            )
        return headers

    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) if self.api_key_env else None


class ThinkingCapabilities(StrictModel):
    modes: set[ThinkingMode] = Field(default_factory=lambda: {"provider-default"}, min_length=1)


class GenerationCapabilities(StrictModel):
    thinking: ThinkingCapabilities = Field(default_factory=ThinkingCapabilities)
    final_answer_reserve: bool = False


class ModelCapabilities(StrictModel):
    context_window: int = Field(gt=0)
    max_output: int = Field(default=8192, gt=0)
    modalities: set[Literal["text", "image", "audio"]] = Field(default_factory=lambda: {"text"})
    capabilities: set[str] = Field(default_factory=set)
    tool_calling: bool = False
    reasoning: bool = False
    generation: GenerationCapabilities = Field(default_factory=GenerationCapabilities)
    prefix_cache: bool = False
    estimated_tps: float | None = Field(default=None, gt=0)
    max_concurrency: int = Field(default=1, gt=0)


class ModelSpec(ModelCapabilities):
    provider: str
    model: str


class PoolSpec(StrictModel):
    main: str
    experts: dict[str, str] = Field(default_factory=dict)


class PolicySpec(StrictModel):
    type: str = "direct"
    max_expert_calls: int = Field(default=2, ge=0)
    options: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_common_options(self) -> PolicySpec:
        max_advice = self.options.get("max_advice_chars")
        if max_advice is not None and (
            not isinstance(max_advice, int)
            or isinstance(max_advice, bool)
            or not 0 < max_advice <= 100_000
        ):
            raise ValueError("options.max_advice_chars must be an integer from 1 to 100000")
        expert_max_tokens = self.options.get("expert_max_tokens")
        if expert_max_tokens is not None and (
            not isinstance(expert_max_tokens, int)
            or isinstance(expert_max_tokens, bool)
            or not 0 < expert_max_tokens <= 32_768
        ):
            raise ValueError("options.expert_max_tokens must be an integer from 1 to 32768")
        if self.type == "reasoning-reserve":
            allowed = {
                "plan_max_tokens",
                "final_answer_min_tokens",
                "max_plan_chars",
                "plan_thinking_mode",
                "final_thinking_mode",
            }
            unknown = sorted(self.options.keys() - allowed)
            if unknown:
                raise ValueError("unknown reasoning-reserve options: " + ", ".join(unknown))
            for name, default in (
                ("plan_max_tokens", 256),
                ("final_answer_min_tokens", 3072),
                ("max_plan_chars", 4000),
            ):
                value = self.options.get(name, default)
                if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= 32_768:
                    raise ValueError(f"options.{name} must be an integer from 1 to 32768")
            for name in ("plan_thinking_mode", "final_thinking_mode"):
                value = self.options.get(name, "disabled")
                if value not in {"provider-default", "disabled"}:
                    raise ValueError(f"options.{name} must be provider-default or disabled")
        return self


class CompletionSpec(StrictModel):
    require_public_output: bool = True
    require_tool_or_text: bool = True
    max_recovery_attempts: int = Field(default=0, ge=0, le=1)
    recovery_max_tokens: int = Field(default=2048, gt=0, le=32_768)

    @model_validator(mode="after")
    def validate_recovery_gate(self) -> CompletionSpec:
        if (
            self.max_recovery_attempts > 0
            and not self.require_public_output
            and not self.require_tool_or_text
        ):
            raise ValueError("recovery requires require_public_output or require_tool_or_text")
        return self


class ServeSpec(StrictModel):
    model_name: str = Field(default="fusion-coding", min_length=1)
    pool: str
    protocols: set[ProtocolName] = Field(
        default_factory=lambda: {"openai-chat", "openai-responses", "anthropic-messages"}
    )
    api_key_env: str | None = Field(default=None, pattern=_ENV_NAME)


class FusionSpec(StrictModel):
    version: Literal["fusion/v1"]
    providers: dict[str, ProviderSpec]
    models: dict[str, ModelSpec]
    pools: dict[str, PoolSpec]
    policy: PolicySpec = Field(default_factory=PolicySpec)
    completion: CompletionSpec = Field(default_factory=CompletionSpec)
    serve: ServeSpec

    @model_validator(mode="after")
    def references_exist(self) -> FusionSpec:
        for name, model in self.models.items():
            if model.provider not in self.providers:
                raise ValueError(f"model {name!r} references unknown provider {model.provider!r}")
        for name, pool in self.pools.items():
            refs = {pool.main, *pool.experts.values()}
            missing = refs - self.models.keys()
            if missing:
                raise ValueError(f"pool {name!r} references unknown models: {sorted(missing)}")
        if self.serve.pool not in self.pools:
            raise ValueError(f"serve.pool references unknown pool {self.serve.pool!r}")
        return self


def load_spec(path: str | Path) -> FusionSpec:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return FusionSpec.model_validate(raw)
