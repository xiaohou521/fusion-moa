import json
from pathlib import Path

from fusion_runtime.cli import main as cli_main
from fusion_runtime.evaluation import (
    BudgetAblationPolicy,
    CodeReserveScreenPolicy,
    CompletionScreenPolicy,
    CompletionVariantSummary,
    EvaluationSummary,
    PromotionPolicy,
    evaluate_budget_ablation,
    evaluate_code_reserve_screen,
    evaluate_completion_screen,
    evaluate_promotion,
)

ROOT = Path(__file__).parents[1]


def test_deepseek_harness_bundle_manifest_and_patch():
    integration = ROOT / "integrations" / "deepseek-harness"
    manifest = json.loads((integration / "package.json").read_text(encoding="utf-8"))
    assert "dsh-plugin" in manifest["keywords"]
    assert manifest["dsh"]["bundle"]["patch"] == "./cordis.patch.yml"
    patch = (integration / "cordis.patch.yml").read_text(encoding="utf-8")
    assert "@deepseek-ai/dsh-llm-pi-ai" in patch
    assert "FUSION_RUNTIME_API_KEY" in patch
    assert "fusion-coding" in patch


def test_cli_check_prints_secret_free_summary(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion-runtime",
            "--config",
            str(ROOT / "recipes" / "local-main-critic.yaml"),
            "--check",
        ],
    )
    cli_main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "valid"
    assert summary["public_model"] == "fusion-coding"
    assert "api_key" not in summary


def test_published_pilot_card_matches_the_promotion_gate():
    card = ROOT / "benchmarks" / "cards" / "lcb-pilot-2026-08-19"
    baseline = EvaluationSummary.model_validate_json(
        (card / "direct-summary.json").read_text(encoding="utf-8")
    )
    candidate = EvaluationSummary.model_validate_json(
        (card / "review-board-summary.json").read_text(encoding="utf-8")
    )
    policy = PromotionPolicy.model_validate_json(
        (card / "promotion-policy.json").read_text(encoding="utf-8")
    )
    stored = json.loads((card / "decision.json").read_text(encoding="utf-8"))

    assert evaluate_promotion(baseline, candidate, policy).model_dump() == stored
    assert stored["promote"] is False


def test_published_completion_card_matches_the_screen_gate():
    card = ROOT / "benchmarks" / "cards" / "completion-screen-2026-08-24"
    baseline = CompletionVariantSummary.model_validate_json(
        (card / "provider-default-direct-summary.json").read_text(encoding="utf-8")
    )
    candidate = CompletionVariantSummary.model_validate_json(
        (card / "thinking-disabled-recovery-summary.json").read_text(encoding="utf-8")
    )
    policy = CompletionScreenPolicy.model_validate_json(
        (card / "screen-policy.json").read_text(encoding="utf-8")
    )
    stored = json.loads((card / "decision.json").read_text(encoding="utf-8"))

    assert evaluate_completion_screen(baseline, candidate, policy).model_dump() == stored
    assert stored["advance_to_larger_evaluation"] is True


def test_published_reasoning_reserve_card_matches_the_screen_gate():
    card = ROOT / "benchmarks" / "cards" / "reasoning-reserve-screen-2026-08-25"
    baseline = CompletionVariantSummary.model_validate_json(
        (card / "thinking-disabled-direct-summary.json").read_text(encoding="utf-8")
    )
    candidate = CompletionVariantSummary.model_validate_json(
        (card / "reasoning-reserve-plan256-summary.json").read_text(encoding="utf-8")
    )
    policy = CodeReserveScreenPolicy.model_validate_json(
        (card / "screen-policy.json").read_text(encoding="utf-8")
    )
    stored = json.loads((card / "decision.json").read_text(encoding="utf-8"))

    assert evaluate_code_reserve_screen(baseline, candidate, policy).model_dump() == stored
    assert stored["advance_to_larger_evaluation"] is True


def test_published_output_budget_card_matches_the_sequential_gate():
    card = ROOT / "benchmarks" / "cards" / "output-budget-ablation-2026-08-25"
    variants = [
        CompletionVariantSummary.model_validate_json(
            (card / f"reasoning-reserve-{budget}-summary.json").read_text(encoding="utf-8")
        )
        for budget in (4096, 8192, 16384)
    ]
    policy = BudgetAblationPolicy.model_validate_json(
        (card / "budget-policy.json").read_text(encoding="utf-8")
    )
    stored = json.loads((card / "decision.json").read_text(encoding="utf-8"))

    assert evaluate_budget_ablation(variants, policy).model_dump() == stored
    assert stored["selected_recipe"] == "reasoning-reserve-4096"
    assert stored["selected_higher_budget"] is False
