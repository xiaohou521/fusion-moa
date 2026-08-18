from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationSummary(StrictModel):
    """Aggregate from one frozen, reproducible evaluation run."""

    recipe: str = Field(min_length=1)
    task_set_digest: str = Field(min_length=1)
    environment_digest: str = Field(min_length=1)
    seed: int
    attempted: int = Field(gt=0)
    passed: int = Field(ge=0)
    infrastructure_failures: int = Field(default=0, ge=0)
    p95_latency_ms: float = Field(gt=0)
    mean_cost_usd: float = Field(ge=0)

    @model_validator(mode="after")
    def counts_are_possible(self) -> EvaluationSummary:
        if self.passed > self.attempted:
            raise ValueError("passed cannot exceed attempted")
        if self.infrastructure_failures > self.attempted:
            raise ValueError("infrastructure_failures cannot exceed attempted")
        return self

    @computed_field
    @property
    def pass_rate(self) -> float:
        return self.passed / self.attempted

    @computed_field
    @property
    def infrastructure_failure_rate(self) -> float:
        return self.infrastructure_failures / self.attempted


class PromotionPolicy(StrictModel):
    min_attempts: int = Field(default=20, gt=0)
    min_pass_rate_delta: float = 0.0
    max_p95_latency_ratio: float = Field(default=1.25, gt=0)
    max_mean_cost_ratio: float = Field(default=1.25, gt=0)
    max_infrastructure_failure_rate_delta: float = Field(default=0.0, ge=0)
    require_any_improvement: bool = True


class PromotionDecision(StrictModel):
    promote: bool
    reasons: list[str]
    pass_rate_delta: float
    p95_latency_ratio: float
    mean_cost_ratio: float
    infrastructure_failure_rate_delta: float


def evaluate_promotion(
    baseline: EvaluationSummary,
    candidate: EvaluationSummary,
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    """Fail closed unless a candidate clears every frozen comparison gate."""

    policy = policy or PromotionPolicy()
    reasons: list[str] = []
    if baseline.task_set_digest != candidate.task_set_digest:
        reasons.append("task_set_digest differs")
    if baseline.environment_digest != candidate.environment_digest:
        reasons.append("environment_digest differs")
    if baseline.seed != candidate.seed:
        reasons.append("seed differs")
    if baseline.attempted != candidate.attempted:
        reasons.append("attempted count differs")
    if baseline.attempted < policy.min_attempts:
        reasons.append("baseline has too few attempts")
    if candidate.attempted < policy.min_attempts:
        reasons.append("candidate has too few attempts")

    pass_delta = candidate.pass_rate - baseline.pass_rate
    latency_ratio = candidate.p95_latency_ms / baseline.p95_latency_ms
    cost_ratio = _ratio(candidate.mean_cost_usd, baseline.mean_cost_usd)
    infra_delta = candidate.infrastructure_failure_rate - baseline.infrastructure_failure_rate
    if pass_delta < policy.min_pass_rate_delta:
        reasons.append(
            f"pass-rate delta {pass_delta:.6f} is below {policy.min_pass_rate_delta:.6f}"
        )
    if latency_ratio > policy.max_p95_latency_ratio:
        reasons.append(
            f"p95 latency ratio {latency_ratio:.6f} exceeds {policy.max_p95_latency_ratio:.6f}"
        )
    if cost_ratio > policy.max_mean_cost_ratio:
        reasons.append(f"mean cost ratio {cost_ratio:.6f} exceeds {policy.max_mean_cost_ratio:.6f}")
    if infra_delta > policy.max_infrastructure_failure_rate_delta:
        reasons.append(
            "infrastructure-failure-rate delta "
            f"{infra_delta:.6f} exceeds {policy.max_infrastructure_failure_rate_delta:.6f}"
        )
    if policy.require_any_improvement and not (
        pass_delta > 0 or latency_ratio < 1 or cost_ratio < 1 or infra_delta < 0
    ):
        reasons.append("candidate improves no gated metric")
    return PromotionDecision(
        promote=not reasons,
        reasons=reasons,
        pass_rate_delta=pass_delta,
        p95_latency_ratio=latency_ratio,
        mean_cost_ratio=cost_ratio,
        infrastructure_failure_rate_delta=infra_delta,
    )


def _ratio(candidate: float, baseline: float) -> float:
    if baseline == 0:
        # Keep CLI JSON standards-compliant while still making the gate fail closed.
        return 1.0 if candidate == 0 else 1e300
    return candidate / baseline


def _load(path: str, model: type[StrictModel]) -> StrictModel:
    return model.model_validate_json(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a candidate Fusion recipe with a frozen baseline"
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--policy")
    args = parser.parse_args()
    baseline = _load(args.baseline, EvaluationSummary)
    candidate = _load(args.candidate, EvaluationSummary)
    policy = _load(args.policy, PromotionPolicy) if args.policy else PromotionPolicy()
    decision = evaluate_promotion(baseline, candidate, policy)  # type: ignore[arg-type]
    print(json.dumps(decision.model_dump(), ensure_ascii=False, indent=2))
    raise SystemExit(0 if decision.promote else 2)


if __name__ == "__main__":
    main()
