from fusion_runtime.evaluation import (
    EvaluationSummary,
    PromotionPolicy,
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
