# Contributing

Fusion Runtime welcomes provider plugins, policy plugins, protocol fixes,
recipes, evaluation tooling, and reproducible performance cards.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install '.[dev]'
pytest
```

The test configuration imports directly from `src`, so local source edits are
picked up without an editable install. Re-run `pip install '.[dev]'` before
testing the generated console commands after changing package code.

Keep changes model- and hardware-independent. New core behavior needs CPU-only
contract tests with fake or mocked providers; tests must not require an API key,
GPU, private endpoint, or network connection.

## Change rules

- Preserve the provider/policy/protocol separation enforced by the runtime interfaces.
- Add recipe fields only with strict validation and documented failure behavior.
- Never infer capability from a model name when it can be declared.
- Never commit secrets, private endpoint credentials, raw user traces, or model
  licenses/weights that cannot be redistributed.
- Bound fan-out, output injection, retries, and repair rounds.
- Fail visibly on unsupported protocol features; do not silently drop them.

## Recipe and performance contributions

A recipe PR must state its intended workload and direct fallback. A performance
claim must include direct and fusion runs on the same task-set digest, seed, and
environment, plus latency, token/cost, cache, concurrency, retry, and
infrastructure-failure accounting. A winning subset alone is not a valid card.

## Design changes

Open an issue or short RFC for public schema changes, new plugin groups, trust
boundary changes, or compatibility breaks. Maintainers should record the
decision and migration path before merging implementation.
