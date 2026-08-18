# Fusion MoA

[简体中文](README.zh-CN.md)

Fusion MoA is a model-, GPU-, and harness-independent Mixture-of-Agents runtime
for coding agents. Bring a main model and an optional pool of read-only experts,
configure them in one recipe, and expose one stable API to Codex, Claude Code,
OpenCode, DeepSeek Harness, or another client.

```text
coding agents / DeepSeek Harness
               |
 OpenAI Chat + Responses + Anthropic Messages
               |
       protocol-neutral runtime
    policy + experts + budgets + fallback
               |
 vLLM / llama.cpp / cloud APIs / plugins
```

## Status

The v0.1 contract includes:

- strict `fusion/v1` recipes with cross-reference and secret validation;
- declared model capabilities and per-model concurrency limits;
- OpenAI-compatible, llama.cpp, and Anthropic-compatible providers;
- `direct`, `main-critic`, and parallel `review-board` policies;
- OpenAI Chat, OpenAI Responses, and Anthropic Messages endpoints;
- portable function/tool-call round trips and `/v1/models` discovery;
- native final-model SSE for all three public protocols;
- Python entry points for third-party providers and policies;
- an evaluation-gated promotion command for controlled RSI;
- a version-pinned DeepSeek Harness profile bundle.

Native final-model streaming means expert orchestration completes first, then
text and tool-call deltas from the one authoritative main-model call are
forwarded without buffering the full answer. Expert output is never exposed as
the public stream. Non-function built-in tools, multimodal parity across every
provider, and online training are not claimed in v0.1.

## Quick start

```bash
git clone https://github.com/xiaohou521/fusion-moa.git
cd fusion-moa
python -m venv .venv
. .venv/bin/activate
pip install .

cp recipes/local-main-critic.yaml my-recipe.yaml
# Edit endpoints/model ids and export referenced keys. Never put keys in YAML.
fusion-runtime --config my-recipe.yaml --port 18888
```

Point an OpenAI-compatible client at `http://127.0.0.1:18888/v1` and select
`fusion-coding`:

```bash
curl http://127.0.0.1:18888/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"fusion-coding","messages":[{"role":"user","content":"Review this patch"}]}'
```

Use [`recipes/review-board.yaml`](recipes/review-board.yaml) to mix providers
and define role-named experts. A pool can use local models, hosted APIs, or both;
the runtime never inspects GPU type or guesses capability from model names.

## Configuration model

A recipe has five explicit layers:

1. `providers`: transport endpoints and environment-variable credential refs;
2. `models`: provider model ids plus declared capacity and capabilities;
3. `pools`: one authoritative main model and role-to-expert assignments;
4. `policy`: orchestration, expert-call budget, and policy-specific options;
5. `serve`: one public model name and enabled client protocols.

Experts are advisory-only. They receive no coding tools, their output is
bounded and marked untrusted, and only the main model can produce the public
answer or tool call. A failed expert is surfaced through `x-fusion-fallback`;
it cannot silently become the writer.

## Plugin contract

Provider packages register a factory under `fusion_runtime.providers`; policy
packages use `fusion_runtime.policies`:

```toml
[project.entry-points."fusion_runtime.providers"]
my-provider = "my_package:MyProvider"

[project.entry-points."fusion_runtime.policies"]
my-policy = "my_package:MyPolicy"
```

A provider implements `async complete(model, request) -> ModelResponse` and
`stream(model, request) -> AsyncIterator[ModelStreamEvent]`. A policy implements
`async prepare(runtime, pool_name, request) -> PreparedCall`: experts finish in
`prepare`, while the runtime owns the sole final call in complete or streaming
mode. Protocol translation stays at the gateway boundary and must not own
routing. A provider without `stream` fails a streaming request visibly instead
of silently falling back to buffered output.

The canonical stream is provider-neutral: emit non-empty `TextDelta` and
`ToolCallDelta` events, then exactly one `Finish`; an optional `Usage` may
follow. If an error happens after deltas have started, emit one terminal,
secret-safe `StreamError` instead of throwing and breaking the client SSE. If
the request fails before the first event, raise a typed `ProviderError`. Plugin
authors can freeze this contract in their own test suite:

```python
from fusion_runtime.conformance import assert_stream_conforms, collect_stream

events = await collect_stream(provider.stream(model, request))
assert_stream_conforms(events)
```

The runtime prefetches one canonical event before sending HTTP streaming
headers. This preserves real main-model streaming while allowing connection,
authentication, rate-limit, and initial protocol failures to return normal JSON
errors. The OpenAI Chat, OpenAI Responses, and Anthropic Messages mappings are
covered by deterministic protocol tests, including cancellation and terminal
errors.

`FusionRequest.seed` is forwarded by OpenAI-compatible providers and preserved
through the built-in policies, including expert calls. The built-in Anthropic
provider rejects seeded requests visibly because that parameter is not portable
to Anthropic Messages; a frozen card must record this as a reproducibility issue
or use a provider that honors the seed.

## DeepSeek Harness

[`integrations/deepseek-harness`](integrations/deepseek-harness) is a real
DeepSeek Harness profile bundle using its official generic OpenAI-compatible
LLM seam. It is pinned to the current developer-preview contract; see that
directory for installation and compatibility notes. The integration is
community-maintained and does not claim upstream endorsement.

## Controlled RSI

RSI in this project means evaluation-gated improvement of recipes, prompts,
budgets, stopping rules, and completion gates. It does not mean an online model
may rewrite production code, configuration, or weights.

After evaluating a candidate and baseline on the same frozen task set, seed,
and environment, compare their JSON summaries:

```bash
fusion-runtime-gate \
  --baseline cards/direct-summary.json \
  --candidate cards/review-board-summary.json
```

The command exits `0` only if every quality, latency, cost, infrastructure, and
reproducibility gate passes; otherwise it exits `2` with explicit reasons.
Summaries can record `mean_total_tokens` across the main and all expert calls,
and a policy can set `max_mean_token_ratio`. Any declared
`reproducibility_issues` make promotion fail closed even when aggregate quality
improves.

Provider-neutral runtime code belongs in core. Vendor SDKs, custom routers,
training backends, and additional harness adapters belong in plugins. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md).

## Performance claims

Fusion is not automatically better than direct inference. Publish a recipe only
with a reproducible card comparing direct and fusion modes on identical tasks,
seed, environment, latency, token cost, and infrastructure-failure accounting.
If a candidate does not clear its declared objective, keep the direct route.

The first [frozen LiveCodeBench pilot](benchmarks/cards/lcb-pilot-2026-08-19/CARD.md)
did exactly that: the default review board was rejected and the direct route was
kept. The card contains aggregate evidence and limitations, never private
deployment configuration or raw model answers.

## License

Apache-2.0.
