import json
from pathlib import Path

from fusion_runtime.cli import main as cli_main
from fusion_runtime.evaluation import EvaluationSummary, PromotionPolicy, evaluate_promotion

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
