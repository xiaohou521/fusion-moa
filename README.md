# Fusion MoA

[简体中文](README.zh-CN.md)

One model API for coding agents, backed by your main model and optional expert models.

Fusion MoA is an open, model- and GPU-independent Mixture-of-Agents runtime. You connect the model
endpoints you already use, choose an orchestration recipe, and expose one OpenAI- or
Anthropic-compatible model to Claude Code, Codex, OpenCode, DeepSeek Harness, or another coding
agent.

```text
Coding agent
    │  OpenAI Chat / Responses / Anthropic Messages
    ▼
Fusion MoA  ── policy, budgets, fallback, accounting
    │
    ├── authoritative main model ──► native final stream
    └── optional read-only experts ─► private advice only
```

The main model is always the only public writer. Experts receive no coding tools, their output is
bounded and treated as untrusted, and a failed expert falls forward to the main model. Fusion MoA
does not require a particular model family, inference server, cloud, GPU, or coding-agent harness.

## What works today

The current community release provides:

- one `fusion/v1` YAML recipe for providers, models, pools, policies, completion rules, and serving;
- OpenAI-compatible and Anthropic-compatible upstream providers;
- OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages client endpoints;
- native streaming from the authoritative final model, including incremental tool-call arguments;
- portable function/tool-call round trips and `/v1/models` discovery;
- `direct`, reasoning-reserve, critic, review-board, and adaptive self-review policies;
- explicit capability checks for thinking controls, tools, and schema-constrained output;
- bounded fallback, usage aggregation, and completeness flags across all model calls;
- Python entry points for third-party provider and policy plugins;
- a version-pinned [DeepSeek Harness integration](integrations/deepseek-harness/README.md).

Fusion MoA is early-stage software. Expert orchestration can cost more or perform worse than direct
inference on some workloads; start with `direct`, measure on your own tasks, and add experts only
when the evidence supports them.

## Quick start

Requirements: Python 3.11+ and at least one model endpoint. A local vLLM, SGLang, llama.cpp, or any
service with an OpenAI-compatible Chat Completions API is enough for the first run.

### 1. Install

```bash
git clone https://github.com/xiaohou521/fusion-moa.git
cd fusion-moa
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

### 2. Configure one main model

```bash
cp recipes/direct.yaml fusion.yaml
```

Edit these values in `fusion.yaml`:

```yaml
providers:
  main_api:
    base_url: http://127.0.0.1:8000/v1

models:
  main:
    model: your-coding-model-id
```

Secrets are referenced by environment-variable name and never written into the recipe:

```bash
export MAIN_MODEL_API_KEY='your-upstream-key'
export FUSION_RUNTIME_API_KEY='choose-a-key-for-coding-agents'
```

If the local upstream does not require authentication, remove `api_key_env` from that provider.

### 3. Validate and start

```bash
fusion-runtime --config fusion.yaml --check
fusion-runtime --config fusion.yaml --host 127.0.0.1 --port 18888
```

Check the gateway:

```bash
curl http://127.0.0.1:18888/health
```

Send a native streaming request:

```bash
curl http://127.0.0.1:18888/v1/chat/completions \
  -H "Authorization: Bearer $FUSION_RUNTIME_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "fusion-coding",
    "stream": true,
    "messages": [{"role": "user", "content": "Write a Python binary search."}]
  }'
```

### 4. Connect a coding agent

For an OpenAI-compatible client:

```text
Base URL: http://127.0.0.1:18888/v1
API key:  value of FUSION_RUNTIME_API_KEY
Model:    fusion-coding
```

For Claude Code through CC Switch, create a custom Anthropic-compatible provider:

```text
Base URL: http://127.0.0.1:18888
API key:  value of FUSION_RUNTIME_API_KEY
Model:    fusion-coding
```

The Anthropic client adds `/v1/messages`; OpenAI clients use the `/v1` base URL. Keep the runtime on
loopback unless you have added TLS, network access controls, and a strong API key.

## Add an expert reviewer

Start from the adaptive self-review recipe:

```bash
cp recipes/adaptive-self-review.yaml fusion.yaml
export MAIN_MODEL_KEY='your-main-model-key'
export EXPERT_MODEL_KEY='your-expert-model-key'
export FUSION_RUNTIME_API_KEY='choose-a-key-for-coding-agents'
```

Edit both endpoint URLs and model IDs, then validate and start as above. The request path becomes:

```text
bounded main-model plan
        ▼
one read-only structured review
        ▼
authoritative main-model native stream
```

The reviewer must support provider-enforced JSON Schema. It returns either `advise` or `abstain`.
The default token tiers are `512 → 1024 → 2048`; a higher tier is used only when the response was
actually truncated. Schema, semantic, provider, and transport errors fail closed instead of buying
more tokens. Only the final main-model call retains the coding agent's tools.

Inspect response headers while testing:

- `x-fusion-route` shows the policy route, selected tier, and review action;
- `x-fusion-fallback` explains a safe degradation;
- `x-fusion-streaming-mode: native-final` confirms final-model streaming.

## Choose a recipe

| Starting point | Use it when |
| --- | --- |
| [`recipes/direct.yaml`](recipes/direct.yaml) | You want the simplest and cheapest baseline. |
| [`recipes/local-main-critic.yaml`](recipes/local-main-critic.yaml) | One bounded critic should review the main model. |
| [`recipes/review-board.yaml`](recipes/review-board.yaml) | Multiple role-specific experts should advise in parallel. |
| [`recipes/adaptive-reasoning-reserve.yaml`](recipes/adaptive-reasoning-reserve.yaml) | One model should reserve final-answer space and select an output tier. |
| [`recipes/adaptive-self-review.yaml`](recipes/adaptive-self-review.yaml) | A separate structured reviewer should inspect a bounded main-model plan. |

Every recipe has six sections:

1. `providers`: endpoint transports and environment-variable credential references;
2. `models`: upstream model IDs, limits, tools, and declared generation capabilities;
3. `pools`: one authoritative main model and role-named experts;
4. `policy`: orchestration and hard expert budgets;
5. `completion`: public-output requirements and optional bounded recovery;
6. `serve`: the public model name and enabled client protocols.

Capabilities are declarations, not guesses based on model names. Only declare `disabled` or
`bounded` thinking when the provider plugin really maps that control. Only declare `json-schema`
when the upstream endpoint enforces it. Unsupported combinations fail visibly before inference.

## Plugins and integrations

Third-party Python packages can register providers under `fusion_runtime.providers` and policies
under `fusion_runtime.policies`. Provider plugins own upstream protocol translation; policy plugins
own bounded orchestration; the gateway owns client-protocol translation and the final stream.

The repository also contains a community-maintained
[DeepSeek Harness profile bundle](integrations/deepseek-harness/README.md). It uses the Harness's
generic OpenAI-compatible model seam and does not imply upstream endorsement.

## In development

The active roadmap focuses on:

- checkpointed, model-independent evaluation and public Evidence Cards;
- direct and matched-compute controls that separate expert value from extra inference cost;
- selective routing so experts run only where their expected value exceeds their cost;
- privacy-safe outcome storage, failure clustering, and reusable failure taxonomies;
- offline, evaluation-gated recipe evolution with lineage, rejection, promotion, and rollback;
- more provider, coding-agent, and community expert-pool plugins;
- optional training/model-adapter plugins that remain outside the core runtime.

This roadmap does **not** mean a production model can currently rewrite runtime code, modify its own
weights, or promote itself online. Fusion MoA uses “RSI” to mean a controlled offline loop:
observe failures → propose a bounded recipe candidate → run frozen evaluation → explicitly accept or
reject it.

## Evidence, security, and contributing

Fusion is not automatically better than direct inference. Published aggregate experiments live in
[`benchmarks/cards`](benchmarks/cards); they document both accepted and rejected hypotheses and do
not contain private endpoints, credentials, raw prompts, or model responses.

- Keep credentials in environment variables, never YAML or Git.
- Treat recipes and installed plugins as trusted deployment code.
- Treat user input, model output, and expert advice as untrusted data.
- Read [SECURITY.md](SECURITY.md) before exposing the gateway outside localhost.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for provider, policy, recipe, and evaluation contributions.

## License

Apache-2.0.
