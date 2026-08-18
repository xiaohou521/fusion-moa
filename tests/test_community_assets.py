import json
from pathlib import Path

from fusion_runtime.cli import main as cli_main

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
