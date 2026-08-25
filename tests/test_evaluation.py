import pytest

from fusion_runtime.evaluation import (
    BudgetAblationPolicy,
    CodeReserveScreenPolicy,
    CompletionScreenPolicy,
    CompletionSeedSummary,
    CompletionVariantSummary,
    EvaluationSummary,
    PromotionPolicy,
    evaluate_budget_ablation,
    evaluate_code_reserve_screen,
    evaluate_completion_screen,
    evaluate_promotion,
)


def summary(
    recipe,
    *,
    passed,
    latency=1000,
    cost=1,
    tokens=1000,
    infra=0,
    digest="tasks",
    reproducibility_issues=None,
):
    return EvaluationSummary(
        recipe=recipe,
        task_set_digest=digest,
        environment_digest="env",
        seed=7,
        attempted=100,
        passed=passed,
        infrastructure_failures=infra,
        p95_latency_ms=latency,
        mean_cost_usd=cost,
        mean_total_tokens=tokens,
        reproducibility_issues=reproducibility_issues or [],
    )


def test_candidate_clearing_all_gates_is_promotable():
    decision = evaluate_promotion(
        summary("direct", passed=60),
        summary("review-board", passed=66, latency=1100, cost=1.1),
        PromotionPolicy(min_pass_rate_delta=0.05),
    )
    assert decision.promote is True
    assert decision.reasons == []


def test_comparison_fails_closed_on_regression_or_dataset_drift():
    decision = evaluate_promotion(
        summary("direct", passed=60),
        summary("candidate", passed=59, latency=1400, digest="other"),
    )
    assert decision.promote is False
    assert "task_set_digest differs" in decision.reasons
    assert any("pass-rate delta" in reason for reason in decision.reasons)
    assert any("latency ratio" in reason for reason in decision.reasons)


def test_new_cost_from_zero_baseline_fails_cost_gate():
    decision = evaluate_promotion(
        summary("direct", passed=60, cost=0),
        summary("candidate", passed=61, cost=0.01),
    )
    assert decision.promote is False
    assert any("cost ratio" in reason for reason in decision.reasons)


def test_identical_candidate_is_not_recursive_improvement():
    decision = evaluate_promotion(
        summary("direct", passed=60),
        summary("renamed", passed=60),
    )
    assert decision.promote is False
    assert "candidate improves no gated metric" in decision.reasons


def test_unverified_reproducibility_fails_closed():
    decision = evaluate_promotion(
        summary("direct", passed=60),
        summary(
            "candidate",
            passed=66,
            reproducibility_issues=["provider did not honor a generation seed"],
        ),
    )

    assert decision.promote is False
    assert any("candidate reproducibility" in reason for reason in decision.reasons)


def test_optional_token_ratio_gate_accounts_for_expert_calls():
    decision = evaluate_promotion(
        summary("direct", passed=60, tokens=1000),
        summary("review-board", passed=66, tokens=3000),
        PromotionPolicy(max_mean_token_ratio=2.0),
    )

    assert decision.promote is False
    assert decision.mean_total_token_ratio == 3.0
    assert any("mean token ratio" in reason for reason in decision.reasons)


def test_emitted_summary_round_trips_without_computed_fields():
    original = summary("direct", passed=60)
    payload = original.model_dump(exclude_computed_fields=True)
    assert "pass_rate" not in payload
    restored = EvaluationSummary.model_validate(payload)
    assert restored.pass_rate == 0.6


def completion_variant(
    recipe,
    *,
    recipe_digest,
    passed=(7, 7, 7),
    empty=(8, 8, 8),
    unextractable=None,
    truncated=(0, 0, 0),
    infra=(0, 0, 0),
    latency=1000,
    tokens=1000,
    accounting_issues=None,
    reproducibility_issues=None,
):
    if unextractable is None:
        unextractable = empty
    return CompletionVariantSummary(
        recipe=recipe,
        recipe_digest=recipe_digest,
        task_set_digest="tasks",
        environment_digest="environment",
        grader_digest="grader",
        seeds=[
            CompletionSeedSummary(
                seed=seed,
                attempted=15,
                passed=seed_passed,
                empty_outputs=seed_empty,
                unextractable_answers=seed_unextractable,
                truncated_outputs=seed_truncated,
                infrastructure_failures=seed_infra,
                recovery_attempts=0,
            )
            for (
                seed,
                seed_passed,
                seed_empty,
                seed_unextractable,
                seed_truncated,
                seed_infra,
            ) in zip(
                (101, 202, 303),
                passed,
                empty,
                unextractable,
                truncated,
                infra,
                strict=True,
            )
        ],
        p95_latency_ms=latency,
        mean_total_tokens=tokens,
        accounting_issues=accounting_issues or [],
        reproducibility_issues=reproducibility_issues or [],
    )


def test_completion_screen_advances_without_claiming_promotion():
    decision = evaluate_completion_screen(
        completion_variant("default direct", recipe_digest="parent"),
        completion_variant(
            "disabled thinking with recovery",
            recipe_digest="candidate",
            passed=(7, 8, 6),
            empty=(4, 4, 4),
            latency=1240,
            tokens=1250,
        ),
    )

    assert decision.advance_to_larger_evaluation is True
    assert decision.non_regressing_seeds == 2
    assert decision.empty_output_reduction == 0.5
    assert decision.reasons == []


def test_completion_screen_fails_closed_on_missing_evidence_and_budget_regression():
    baseline = completion_variant("default direct", recipe_digest="parent")
    candidate = completion_variant(
        "candidate",
        recipe_digest="candidate",
        passed=(6, 6, 6),
        empty=(7, 7, 7),
        infra=(1, 0, 0),
        latency=1300,
        tokens=1300,
        accounting_issues=["attempt_usage_missing"],
    )
    candidate = candidate.model_copy(
        update={"grader_digest": "different-grader", "seeds": candidate.seeds[:2]}
    )

    decision = evaluate_completion_screen(baseline, candidate)

    assert decision.advance_to_larger_evaluation is False
    assert "grader_digest differs" in decision.reasons
    assert "candidate seed count differs from policy" in decision.reasons
    assert any("only 0 seeds" in reason for reason in decision.reasons)
    assert any("empty-output reduction" in reason for reason in decision.reasons)
    assert any("p95 latency ratio" in reason for reason in decision.reasons)
    assert any("mean token ratio" in reason for reason in decision.reasons)
    assert any("infrastructure-failure-rate" in reason for reason in decision.reasons)
    assert any("accounting is incomplete" in reason for reason in decision.reasons)


def test_completion_screen_rejects_zero_failure_baseline_as_unproven_hypothesis():
    decision = evaluate_completion_screen(
        completion_variant("default direct", recipe_digest="parent", empty=(0, 0, 0)),
        completion_variant("candidate", recipe_digest="candidate", empty=(0, 0, 0)),
        CompletionScreenPolicy(min_empty_output_reduction=0.5),
    )

    assert decision.advance_to_larger_evaluation is False
    assert decision.empty_output_reduction == 0


def test_code_reserve_screen_advances_on_quality_and_complete_code_gain():
    decision = evaluate_code_reserve_screen(
        completion_variant(
            "disabled direct",
            recipe_digest="parent",
            passed=(7, 7, 6),
            empty=(0, 0, 0),
            unextractable=(7, 8, 8),
            truncated=(7, 8, 7),
            latency=290726.804,
            tokens=2863.911,
        ),
        completion_variant(
            "reasoning reserve",
            recipe_digest="candidate",
            passed=(10, 10, 10),
            empty=(0, 0, 0),
            unextractable=(4, 2, 5),
            truncated=(4, 2, 5),
            latency=282420.965,
            tokens=3102.022,
        ),
    )

    assert decision.advance_to_larger_evaluation is True
    assert decision.aggregate_pass_delta == 10
    assert decision.pass_deltas_by_seed == {"101": 3, "202": 3, "303": 4}
    assert decision.non_regressing_seeds == 3
    assert decision.unextractable_answer_reduction == pytest.approx(12 / 23)
    assert decision.reasons == []


def test_code_reserve_screen_fails_closed_on_quality_cost_and_evidence():
    decision = evaluate_code_reserve_screen(
        completion_variant(
            "disabled direct",
            recipe_digest="parent",
            passed=(7, 7, 7),
            empty=(0, 0, 0),
            unextractable=(6, 6, 6),
        ),
        completion_variant(
            "candidate",
            recipe_digest="candidate",
            passed=(6, 6, 6),
            empty=(0, 0, 0),
            unextractable=(6, 6, 6),
            latency=1300,
            tokens=1300,
            accounting_issues=["attempt_usage_missing"],
        ),
        CodeReserveScreenPolicy(),
    )

    assert decision.advance_to_larger_evaluation is False
    assert any("aggregate pass delta" in reason for reason in decision.reasons)
    assert any("only 0 seeds" in reason for reason in decision.reasons)
    assert any("unextractable-answer reduction" in reason for reason in decision.reasons)
    assert any("p95 latency ratio" in reason for reason in decision.reasons)
    assert any("mean token ratio" in reason for reason in decision.reasons)
    assert any("accounting is incomplete" in reason for reason in decision.reasons)


def test_budget_ablation_stops_at_first_failed_adjacent_gate():
    decision = evaluate_budget_ablation(
        [
            completion_variant(
                "reserve-4096",
                recipe_digest="4k",
                passed=(10, 10, 10),
                empty=(0, 0, 0),
                unextractable=(4, 2, 5),
                latency=282420.965,
                tokens=3102.022,
            ),
            completion_variant(
                "reserve-8192",
                recipe_digest="8k",
                passed=(12, 11, 11),
                empty=(0, 0, 0),
                unextractable=(1, 0, 2),
                latency=729769.466,
                tokens=3611.644,
            ),
            completion_variant(
                "reserve-16384",
                recipe_digest="16k",
                passed=(12, 11, 12),
                empty=(0, 0, 0),
                unextractable=(0, 0, 0),
                latency=1026325.835,
                tokens=3831.044,
            ),
        ]
    )

    assert decision.selected_recipe == "reserve-4096"
    assert decision.selected_index == 0
    assert decision.selected_higher_budget is False
    assert decision.stopped_early is True
    assert len(decision.evaluated_steps) == 1
    step = decision.evaluated_steps[0]
    assert step.aggregate_pass_delta == 4
    assert step.pass_deltas_by_seed == {"101": 2, "202": 1, "303": 1}
    assert step.unextractable_answer_delta == -8
    assert step.unextractable_answer_reduction == pytest.approx(8 / 11)
    assert step.meaningful_gain is True
    assert step.p95_latency_ratio == pytest.approx(2.5839776661)
    assert any("p95 latency ratio" in reason for reason in step.reasons)


def test_budget_ablation_selects_highest_when_every_adjacent_gate_passes():
    decision = evaluate_budget_ablation(
        [
            completion_variant(
                "4k",
                recipe_digest="4k",
                passed=(8, 8, 8),
                empty=(0, 0, 0),
                unextractable=(4, 4, 4),
            ),
            completion_variant(
                "8k",
                recipe_digest="8k",
                passed=(9, 9, 9),
                empty=(0, 0, 0),
                unextractable=(3, 3, 3),
                latency=1200,
                tokens=1200,
            ),
            completion_variant(
                "16k",
                recipe_digest="16k",
                passed=(10, 10, 10),
                empty=(0, 0, 0),
                unextractable=(2, 2, 2),
                latency=1400,
                tokens=1400,
            ),
        ]
    )

    assert decision.selected_recipe == "16k"
    assert decision.selected_index == 2
    assert decision.selected_higher_budget is True
    assert decision.stopped_early is False
    assert decision.reasons == []
    assert len(decision.evaluated_steps) == 2


def test_budget_ablation_fails_closed_on_weak_gain_and_incomplete_evidence():
    decision = evaluate_budget_ablation(
        [
            completion_variant(
                "4k",
                recipe_digest="4k",
                passed=(10, 10, 10),
                empty=(0, 0, 0),
                unextractable=(1, 1, 1),
            ),
            completion_variant(
                "8k",
                recipe_digest="8k",
                passed=(10, 10, 10),
                empty=(0, 0, 0),
                unextractable=(1, 1, 1),
                accounting_issues=["attempt_usage_missing"],
            ),
        ],
        BudgetAblationPolicy(),
    )

    assert decision.selected_recipe == "4k"
    assert decision.selected_higher_budget is False
    assert any("no meaningful gain" in reason for reason in decision.reasons)
    assert any("accounting is incomplete" in reason for reason in decision.reasons)


def test_budget_ablation_requires_two_ordered_variants():
    with pytest.raises(ValueError, match="at least two"):
        evaluate_budget_ablation(
            [completion_variant("4k", recipe_digest="4k")],
        )
