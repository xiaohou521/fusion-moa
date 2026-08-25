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
    mean_total_tokens: float = Field(default=0, ge=0)
    reproducibility_issues: list[str] = Field(default_factory=list)

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
    max_mean_token_ratio: float | None = Field(default=None, gt=0)
    max_infrastructure_failure_rate_delta: float = Field(default=0.0, ge=0)
    require_any_improvement: bool = True


class PromotionDecision(StrictModel):
    promote: bool
    reasons: list[str]
    pass_rate_delta: float
    p95_latency_ratio: float
    mean_cost_ratio: float
    mean_total_token_ratio: float
    infrastructure_failure_rate_delta: float


class CompletionSeedSummary(StrictModel):
    """Counts from one seed of a completion-focused screen."""

    seed: int
    attempted: int = Field(gt=0)
    passed: int = Field(ge=0)
    empty_outputs: int = Field(ge=0)
    unextractable_answers: int = Field(ge=0)
    truncated_outputs: int = Field(default=0, ge=0)
    infrastructure_failures: int = Field(default=0, ge=0)
    recovery_attempts: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def counts_are_possible(self) -> CompletionSeedSummary:
        for name in (
            "passed",
            "empty_outputs",
            "unextractable_answers",
            "truncated_outputs",
            "infrastructure_failures",
        ):
            if getattr(self, name) > self.attempted:
                raise ValueError(f"{name} cannot exceed attempted")
        return self


class CompletionVariantSummary(StrictModel):
    """Secret-free aggregate for one variant in a repeated completion screen."""

    recipe: str = Field(min_length=1)
    recipe_digest: str = Field(min_length=1)
    task_set_digest: str = Field(min_length=1)
    environment_digest: str = Field(min_length=1)
    grader_digest: str = Field(min_length=1)
    seeds: list[CompletionSeedSummary] = Field(min_length=1)
    p95_latency_ms: float = Field(gt=0)
    mean_prompt_tokens: float = Field(default=0, ge=0)
    mean_completion_tokens: float = Field(default=0, ge=0)
    mean_total_tokens: float = Field(ge=0)
    accounting_issues: list[str] = Field(default_factory=list)
    reproducibility_issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def seeds_are_unique(self) -> CompletionVariantSummary:
        values = [item.seed for item in self.seeds]
        if len(values) != len(set(values)):
            raise ValueError("seed summaries must be unique")
        return self

    @computed_field
    @property
    def attempted(self) -> int:
        return sum(item.attempted for item in self.seeds)

    @computed_field
    @property
    def passed(self) -> int:
        return sum(item.passed for item in self.seeds)

    @computed_field
    @property
    def empty_output_rate(self) -> float:
        return sum(item.empty_outputs for item in self.seeds) / self.attempted

    @computed_field
    @property
    def unextractable_answer_rate(self) -> float:
        return sum(item.unextractable_answers for item in self.seeds) / self.attempted

    @computed_field
    @property
    def truncated_output_rate(self) -> float:
        return sum(item.truncated_outputs for item in self.seeds) / self.attempted

    @computed_field
    @property
    def infrastructure_failure_rate(self) -> float:
        return sum(item.infrastructure_failures for item in self.seeds) / self.attempted


class CompletionScreenPolicy(StrictModel):
    """Frozen gates for a small repeated screen, never for production promotion."""

    expected_seed_count: int = Field(default=3, gt=0)
    expected_attempts_per_seed: int = Field(default=15, gt=0)
    min_non_regressing_seeds: int = Field(default=2, gt=0)
    min_empty_output_reduction: float = Field(default=0.5, ge=0, le=1)
    max_p95_latency_ratio: float = Field(default=1.25, gt=0)
    max_mean_token_ratio: float = Field(default=1.25, gt=0)
    max_infrastructure_failure_rate_delta: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def seed_gate_is_possible(self) -> CompletionScreenPolicy:
        if self.min_non_regressing_seeds > self.expected_seed_count:
            raise ValueError("min_non_regressing_seeds cannot exceed expected_seed_count")
        return self


class CompletionScreenDecision(StrictModel):
    """Decision to continue to a larger evaluation, not a promotion decision."""

    advance_to_larger_evaluation: bool
    reasons: list[str]
    non_regressing_seeds: int
    pass_deltas_by_seed: dict[str, int]
    empty_output_reduction: float
    p95_latency_ratio: float
    mean_total_token_ratio: float
    infrastructure_failure_rate_delta: float


class CodeReserveScreenPolicy(StrictModel):
    """Frozen gates for a final-code-reserve screen, never for promotion."""

    expected_seed_count: int = Field(default=3, gt=0)
    expected_attempts_per_seed: int = Field(default=15, gt=0)
    min_non_regressing_seeds: int = Field(default=2, gt=0)
    min_aggregate_pass_delta: int = Field(default=0)
    min_unextractable_reduction: float = Field(default=0.25, ge=0, le=1)
    max_p95_latency_ratio: float = Field(default=1.25, gt=0)
    max_mean_token_ratio: float = Field(default=1.25, gt=0)
    max_infrastructure_failure_rate_delta: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def seed_gate_is_possible(self) -> CodeReserveScreenPolicy:
        if self.min_non_regressing_seeds > self.expected_seed_count:
            raise ValueError("min_non_regressing_seeds cannot exceed expected_seed_count")
        return self


class CodeReserveScreenDecision(StrictModel):
    """Decision to continue a reserve candidate, not a promotion decision."""

    advance_to_larger_evaluation: bool
    reasons: list[str]
    aggregate_pass_delta: int
    non_regressing_seeds: int
    pass_deltas_by_seed: dict[str, int]
    unextractable_answer_reduction: float
    p95_latency_ratio: float
    mean_total_token_ratio: float
    infrastructure_failure_rate_delta: float


class BudgetAblationPolicy(StrictModel):
    """Frozen gates for selecting the smallest sufficient output budget."""

    expected_seed_count: int = Field(default=3, gt=0)
    expected_attempts_per_seed: int = Field(default=15, gt=0)
    min_non_regressing_seeds: int = Field(default=2, gt=0)
    min_aggregate_pass_delta: int = Field(default=0)
    min_meaningful_pass_gain: int = Field(default=2, ge=0)
    min_meaningful_unextractable_reduction: float = Field(default=0.25, ge=0, le=1)
    max_p95_latency_ratio: float = Field(default=1.5, gt=0)
    max_mean_token_ratio: float = Field(default=1.5, gt=0)
    max_infrastructure_failure_rate_delta: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def seed_gate_is_possible(self) -> BudgetAblationPolicy:
        if self.min_non_regressing_seeds > self.expected_seed_count:
            raise ValueError("min_non_regressing_seeds cannot exceed expected_seed_count")
        return self


class BudgetStepDecision(StrictModel):
    """One adjacent comparison in a sequential output-budget ablation."""

    baseline_recipe: str
    candidate_recipe: str
    advance_to_higher_budget: bool
    reasons: list[str]
    aggregate_pass_delta: int
    non_regressing_seeds: int
    pass_deltas_by_seed: dict[str, int]
    unextractable_answer_delta: int
    unextractable_answer_reduction: float
    meaningful_gain: bool
    p95_latency_ratio: float
    mean_total_token_ratio: float
    infrastructure_failure_rate_delta: float


class BudgetAblationDecision(StrictModel):
    """Sequential selection result; it is not a production promotion decision."""

    selected_recipe: str
    selected_index: int = Field(ge=0)
    selected_higher_budget: bool
    stopped_early: bool
    reasons: list[str]
    evaluated_steps: list[BudgetStepDecision]


def evaluate_budget_ablation(
    variants: list[CompletionVariantSummary],
    policy: BudgetAblationPolicy | None = None,
) -> BudgetAblationDecision:
    """Walk adjacent budgets and stop at the first gate failure."""

    if len(variants) < 2:
        raise ValueError("budget ablation requires at least two ordered variants")
    policy = policy or BudgetAblationPolicy()
    evaluated_steps: list[BudgetStepDecision] = []
    selected_index = 0
    for candidate_index in range(1, len(variants)):
        step = _evaluate_budget_step(
            variants[candidate_index - 1],
            variants[candidate_index],
            policy,
        )
        evaluated_steps.append(step)
        if not step.advance_to_higher_budget:
            return BudgetAblationDecision(
                selected_recipe=variants[selected_index].recipe,
                selected_index=selected_index,
                selected_higher_budget=selected_index > 0,
                stopped_early=True,
                reasons=step.reasons,
                evaluated_steps=evaluated_steps,
            )
        selected_index = candidate_index

    return BudgetAblationDecision(
        selected_recipe=variants[selected_index].recipe,
        selected_index=selected_index,
        selected_higher_budget=selected_index > 0,
        stopped_early=False,
        reasons=[],
        evaluated_steps=evaluated_steps,
    )


def _evaluate_budget_step(
    baseline: CompletionVariantSummary,
    candidate: CompletionVariantSummary,
    policy: BudgetAblationPolicy,
) -> BudgetStepDecision:
    reasons: list[str] = []
    for field_name in ("task_set_digest", "environment_digest", "grader_digest"):
        if getattr(baseline, field_name) != getattr(candidate, field_name):
            reasons.append(f"{field_name} differs")
    if baseline.recipe_digest == candidate.recipe_digest:
        reasons.append("recipe_digest does not differ")

    baseline_by_seed = {item.seed: item for item in baseline.seeds}
    candidate_by_seed = {item.seed: item for item in candidate.seeds}
    baseline_seed_set = set(baseline_by_seed)
    candidate_seed_set = set(candidate_by_seed)
    if baseline_seed_set != candidate_seed_set:
        reasons.append("seed set differs")
    if len(baseline.seeds) != policy.expected_seed_count:
        reasons.append("baseline seed count differs from policy")
    if len(candidate.seeds) != policy.expected_seed_count:
        reasons.append("candidate seed count differs from policy")

    pass_deltas: dict[str, int] = {}
    non_regressing = 0
    for seed in sorted(baseline_seed_set & candidate_seed_set):
        baseline_seed = baseline_by_seed[seed]
        candidate_seed = candidate_by_seed[seed]
        if baseline_seed.attempted != policy.expected_attempts_per_seed:
            reasons.append(f"baseline seed {seed} attempt count differs from policy")
        if candidate_seed.attempted != policy.expected_attempts_per_seed:
            reasons.append(f"candidate seed {seed} attempt count differs from policy")
        if baseline_seed.attempted != candidate_seed.attempted:
            reasons.append(f"attempted count differs for seed {seed}")
        delta = candidate_seed.passed - baseline_seed.passed
        pass_deltas[str(seed)] = delta
        if delta >= 0:
            non_regressing += 1

    aggregate_pass_delta = candidate.passed - baseline.passed
    if aggregate_pass_delta < policy.min_aggregate_pass_delta:
        reasons.append(
            f"aggregate pass delta {aggregate_pass_delta} is below "
            f"{policy.min_aggregate_pass_delta}"
        )
    if non_regressing < policy.min_non_regressing_seeds:
        reasons.append(
            f"only {non_regressing} seeds are non-regressing; "
            f"policy requires {policy.min_non_regressing_seeds}"
        )

    baseline_unextractable = sum(item.unextractable_answers for item in baseline.seeds)
    candidate_unextractable = sum(item.unextractable_answers for item in candidate.seeds)
    unextractable_delta = candidate_unextractable - baseline_unextractable
    unextractable_reduction = _rate_reduction(
        baseline.unextractable_answer_rate,
        candidate.unextractable_answer_rate,
    )
    if unextractable_delta > 0:
        reasons.append(f"unextractable answers increase by {unextractable_delta}")

    meaningful_gain = (
        aggregate_pass_delta >= policy.min_meaningful_pass_gain
        or unextractable_reduction >= policy.min_meaningful_unextractable_reduction
    )
    if not meaningful_gain:
        reasons.append(
            "candidate has no meaningful gain: pass delta "
            f"{aggregate_pass_delta} is below {policy.min_meaningful_pass_gain} and "
            f"unextractable-answer reduction {unextractable_reduction:.6f} is below "
            f"{policy.min_meaningful_unextractable_reduction:.6f}"
        )

    latency_ratio = candidate.p95_latency_ms / baseline.p95_latency_ms
    token_ratio = _ratio(candidate.mean_total_tokens, baseline.mean_total_tokens)
    infra_delta = candidate.infrastructure_failure_rate - baseline.infrastructure_failure_rate
    if latency_ratio > policy.max_p95_latency_ratio:
        reasons.append(
            f"p95 latency ratio {latency_ratio:.6f} exceeds {policy.max_p95_latency_ratio:.6f}"
        )
    if token_ratio > policy.max_mean_token_ratio:
        reasons.append(
            f"mean token ratio {token_ratio:.6f} exceeds {policy.max_mean_token_ratio:.6f}"
        )
    if infra_delta > policy.max_infrastructure_failure_rate_delta:
        reasons.append(
            "infrastructure-failure-rate delta "
            f"{infra_delta:.6f} exceeds "
            f"{policy.max_infrastructure_failure_rate_delta:.6f}"
        )
    for label, summary in (("baseline", baseline), ("candidate", candidate)):
        if summary.accounting_issues:
            reasons.append(
                f"{label} accounting is incomplete: " + "; ".join(summary.accounting_issues)
            )
        if summary.reproducibility_issues:
            reasons.append(
                f"{label} reproducibility is not verified: "
                + "; ".join(summary.reproducibility_issues)
            )

    return BudgetStepDecision(
        baseline_recipe=baseline.recipe,
        candidate_recipe=candidate.recipe,
        advance_to_higher_budget=not reasons,
        reasons=reasons,
        aggregate_pass_delta=aggregate_pass_delta,
        non_regressing_seeds=non_regressing,
        pass_deltas_by_seed=pass_deltas,
        unextractable_answer_delta=unextractable_delta,
        unextractable_answer_reduction=unextractable_reduction,
        meaningful_gain=meaningful_gain,
        p95_latency_ratio=latency_ratio,
        mean_total_token_ratio=token_ratio,
        infrastructure_failure_rate_delta=infra_delta,
    )


def evaluate_code_reserve_screen(
    baseline: CompletionVariantSummary,
    candidate: CompletionVariantSummary,
    policy: CodeReserveScreenPolicy | None = None,
) -> CodeReserveScreenDecision:
    """Fail closed when a final-code-reserve candidate misses a frozen gate."""

    policy = policy or CodeReserveScreenPolicy()
    reasons: list[str] = []
    for field_name in ("task_set_digest", "environment_digest", "grader_digest"):
        if getattr(baseline, field_name) != getattr(candidate, field_name):
            reasons.append(f"{field_name} differs")
    if baseline.recipe_digest == candidate.recipe_digest:
        reasons.append("recipe_digest does not differ")

    baseline_by_seed = {item.seed: item for item in baseline.seeds}
    candidate_by_seed = {item.seed: item for item in candidate.seeds}
    baseline_seed_set = set(baseline_by_seed)
    candidate_seed_set = set(candidate_by_seed)
    if baseline_seed_set != candidate_seed_set:
        reasons.append("seed set differs")
    if len(baseline.seeds) != policy.expected_seed_count:
        reasons.append("baseline seed count differs from policy")
    if len(candidate.seeds) != policy.expected_seed_count:
        reasons.append("candidate seed count differs from policy")

    pass_deltas: dict[str, int] = {}
    non_regressing = 0
    for seed in sorted(baseline_seed_set & candidate_seed_set):
        baseline_seed = baseline_by_seed[seed]
        candidate_seed = candidate_by_seed[seed]
        if baseline_seed.attempted != policy.expected_attempts_per_seed:
            reasons.append(f"baseline seed {seed} attempt count differs from policy")
        if candidate_seed.attempted != policy.expected_attempts_per_seed:
            reasons.append(f"candidate seed {seed} attempt count differs from policy")
        if baseline_seed.attempted != candidate_seed.attempted:
            reasons.append(f"attempted count differs for seed {seed}")
        delta = candidate_seed.passed - baseline_seed.passed
        pass_deltas[str(seed)] = delta
        if delta >= 0:
            non_regressing += 1

    aggregate_pass_delta = candidate.passed - baseline.passed
    if aggregate_pass_delta < policy.min_aggregate_pass_delta:
        reasons.append(
            f"aggregate pass delta {aggregate_pass_delta} is below "
            f"{policy.min_aggregate_pass_delta}"
        )
    if non_regressing < policy.min_non_regressing_seeds:
        reasons.append(
            f"only {non_regressing} seeds are non-regressing; "
            f"policy requires {policy.min_non_regressing_seeds}"
        )

    unextractable_reduction = _rate_reduction(
        baseline.unextractable_answer_rate,
        candidate.unextractable_answer_rate,
    )
    latency_ratio = candidate.p95_latency_ms / baseline.p95_latency_ms
    token_ratio = _ratio(candidate.mean_total_tokens, baseline.mean_total_tokens)
    infra_delta = candidate.infrastructure_failure_rate - baseline.infrastructure_failure_rate
    if unextractable_reduction < policy.min_unextractable_reduction:
        reasons.append(
            f"unextractable-answer reduction {unextractable_reduction:.6f} is below "
            f"{policy.min_unextractable_reduction:.6f}"
        )
    if latency_ratio > policy.max_p95_latency_ratio:
        reasons.append(
            f"p95 latency ratio {latency_ratio:.6f} exceeds {policy.max_p95_latency_ratio:.6f}"
        )
    if token_ratio > policy.max_mean_token_ratio:
        reasons.append(
            f"mean token ratio {token_ratio:.6f} exceeds {policy.max_mean_token_ratio:.6f}"
        )
    if infra_delta > policy.max_infrastructure_failure_rate_delta:
        reasons.append(
            "infrastructure-failure-rate delta "
            f"{infra_delta:.6f} exceeds "
            f"{policy.max_infrastructure_failure_rate_delta:.6f}"
        )
    for label, summary in (("baseline", baseline), ("candidate", candidate)):
        if summary.accounting_issues:
            reasons.append(
                f"{label} accounting is incomplete: " + "; ".join(summary.accounting_issues)
            )
        if summary.reproducibility_issues:
            reasons.append(
                f"{label} reproducibility is not verified: "
                + "; ".join(summary.reproducibility_issues)
            )

    return CodeReserveScreenDecision(
        advance_to_larger_evaluation=not reasons,
        reasons=reasons,
        aggregate_pass_delta=aggregate_pass_delta,
        non_regressing_seeds=non_regressing,
        pass_deltas_by_seed=pass_deltas,
        unextractable_answer_reduction=unextractable_reduction,
        p95_latency_ratio=latency_ratio,
        mean_total_token_ratio=token_ratio,
        infrastructure_failure_rate_delta=infra_delta,
    )


def evaluate_completion_screen(
    baseline: CompletionVariantSummary,
    candidate: CompletionVariantSummary,
    policy: CompletionScreenPolicy | None = None,
) -> CompletionScreenDecision:
    """Fail closed when a completion candidate misses any frozen screen gate."""

    policy = policy or CompletionScreenPolicy()
    reasons: list[str] = []
    for field_name in ("task_set_digest", "environment_digest", "grader_digest"):
        if getattr(baseline, field_name) != getattr(candidate, field_name):
            reasons.append(f"{field_name} differs")
    if baseline.recipe_digest == candidate.recipe_digest:
        reasons.append("recipe_digest does not differ")

    baseline_by_seed = {item.seed: item for item in baseline.seeds}
    candidate_by_seed = {item.seed: item for item in candidate.seeds}
    baseline_seed_set = set(baseline_by_seed)
    candidate_seed_set = set(candidate_by_seed)
    if baseline_seed_set != candidate_seed_set:
        reasons.append("seed set differs")
    if len(baseline.seeds) != policy.expected_seed_count:
        reasons.append("baseline seed count differs from policy")
    if len(candidate.seeds) != policy.expected_seed_count:
        reasons.append("candidate seed count differs from policy")

    shared_seeds = sorted(baseline_seed_set & candidate_seed_set)
    pass_deltas: dict[str, int] = {}
    non_regressing = 0
    for seed in shared_seeds:
        baseline_seed = baseline_by_seed[seed]
        candidate_seed = candidate_by_seed[seed]
        if baseline_seed.attempted != policy.expected_attempts_per_seed:
            reasons.append(f"baseline seed {seed} attempt count differs from policy")
        if candidate_seed.attempted != policy.expected_attempts_per_seed:
            reasons.append(f"candidate seed {seed} attempt count differs from policy")
        if baseline_seed.attempted != candidate_seed.attempted:
            reasons.append(f"attempted count differs for seed {seed}")
        delta = candidate_seed.passed - baseline_seed.passed
        pass_deltas[str(seed)] = delta
        if delta >= 0:
            non_regressing += 1

    if non_regressing < policy.min_non_regressing_seeds:
        reasons.append(
            f"only {non_regressing} seeds are non-regressing; "
            f"policy requires {policy.min_non_regressing_seeds}"
        )

    empty_reduction = _rate_reduction(
        baseline.empty_output_rate,
        candidate.empty_output_rate,
    )
    latency_ratio = candidate.p95_latency_ms / baseline.p95_latency_ms
    token_ratio = _ratio(candidate.mean_total_tokens, baseline.mean_total_tokens)
    infra_delta = candidate.infrastructure_failure_rate - baseline.infrastructure_failure_rate
    if empty_reduction < policy.min_empty_output_reduction:
        reasons.append(
            f"empty-output reduction {empty_reduction:.6f} is below "
            f"{policy.min_empty_output_reduction:.6f}"
        )
    if latency_ratio > policy.max_p95_latency_ratio:
        reasons.append(
            f"p95 latency ratio {latency_ratio:.6f} exceeds {policy.max_p95_latency_ratio:.6f}"
        )
    if token_ratio > policy.max_mean_token_ratio:
        reasons.append(
            f"mean token ratio {token_ratio:.6f} exceeds {policy.max_mean_token_ratio:.6f}"
        )
    if infra_delta > policy.max_infrastructure_failure_rate_delta:
        reasons.append(
            "infrastructure-failure-rate delta "
            f"{infra_delta:.6f} exceeds "
            f"{policy.max_infrastructure_failure_rate_delta:.6f}"
        )
    for label, summary in (("baseline", baseline), ("candidate", candidate)):
        if summary.accounting_issues:
            reasons.append(
                f"{label} accounting is incomplete: " + "; ".join(summary.accounting_issues)
            )
        if summary.reproducibility_issues:
            reasons.append(
                f"{label} reproducibility is not verified: "
                + "; ".join(summary.reproducibility_issues)
            )

    return CompletionScreenDecision(
        advance_to_larger_evaluation=not reasons,
        reasons=reasons,
        non_regressing_seeds=non_regressing,
        pass_deltas_by_seed=pass_deltas,
        empty_output_reduction=empty_reduction,
        p95_latency_ratio=latency_ratio,
        mean_total_token_ratio=token_ratio,
        infrastructure_failure_rate_delta=infra_delta,
    )


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
    if baseline.reproducibility_issues:
        reasons.append(
            "baseline reproducibility is not verified: "
            + "; ".join(baseline.reproducibility_issues)
        )
    if candidate.reproducibility_issues:
        reasons.append(
            "candidate reproducibility is not verified: "
            + "; ".join(candidate.reproducibility_issues)
        )

    pass_delta = candidate.pass_rate - baseline.pass_rate
    latency_ratio = candidate.p95_latency_ms / baseline.p95_latency_ms
    cost_ratio = _ratio(candidate.mean_cost_usd, baseline.mean_cost_usd)
    token_ratio = _ratio(candidate.mean_total_tokens, baseline.mean_total_tokens)
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
    if policy.max_mean_token_ratio is not None and token_ratio > policy.max_mean_token_ratio:
        reasons.append(
            f"mean token ratio {token_ratio:.6f} exceeds {policy.max_mean_token_ratio:.6f}"
        )
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
        mean_total_token_ratio=token_ratio,
        infrastructure_failure_rate_delta=infra_delta,
    )


def _ratio(candidate: float, baseline: float) -> float:
    if baseline == 0:
        # Keep CLI JSON standards-compliant while still making the gate fail closed.
        return 1.0 if candidate == 0 else 1e300
    return candidate / baseline


def _rate_reduction(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else -1.0
    return (baseline - candidate) / baseline


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
