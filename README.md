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
- bounded pre-public-output recovery on the same native stream;
- Python entry points for third-party providers and policies;
- an evaluation-gated promotion command for controlled RSI;
- a version-pinned DeepSeek Harness profile bundle.

Native final-model streaming means expert orchestration completes first, then
public text deltas from the authoritative main model are forwarded without
buffering the full answer. Expert output is never exposed as the public stream.
If the first main-model stream produces no usable public output, an explicitly
enabled policy may replace it with one bounded same-model stream before any
terminal event reaches the client. Non-function built-in tools, multimodal
parity across every provider, and online training are not claimed in v0.1.

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
  -d '{"model":"fusion-coding","stream":true,"messages":[{"role":"user","content":"Review this patch"}]}'
```

Use [`recipes/review-board.yaml`](recipes/review-board.yaml) to mix providers
and define role-named experts. A pool can use local models, hosted APIs, or both;
the runtime never inspects GPU type or guesses capability from model names.

## Configuration model

A recipe has six explicit layers:

1. `providers`: transport endpoints and environment-variable credential refs;
2. `models`: provider model ids plus declared capacity and capabilities;
3. `pools`: one authoritative main model and role-to-expert assignments;
4. `policy`: orchestration, expert-call budget, and policy-specific options;
5. `completion`: output requirements and a bounded recovery budget;
6. `serve`: one public model name and enabled client protocols.

Thinking behavior is a declared generation capability, never inferred from a
model id:

```yaml
models:
  main:
    # provider, model, context_window, ...
    generation:
      thinking:
        modes: [provider-default, disabled, bounded]
      final_answer_reserve: true
```

`provider-default` sends no normalized thinking override. `disabled` and
`bounded` may be used only when both the model declaration and its provider
plugin support them; bounded requests also carry a positive token budget no
larger than the effective output limit. This capability is separate from the
OpenAI-specific `reasoning_effort` passthrough.

Empty-output recovery is explicit and bounded:

```yaml
completion:
  require_public_output: true
  require_tool_or_text: true
  max_recovery_attempts: 1
  recovery_max_tokens: 2048
```

The default `max_recovery_attempts` is `0`, so an existing recipe does not
change behavior until it opts in. When enabled, only an attempt without usable
text or a valid tool call is replaced. The runtime calls the same authoritative
main model once with a bounded recovery instruction; experts are not run again,
hidden provider reasoning is not replayed, and the client/model output limits
remain hard caps.

For a streaming request, leading control events, whitespace-only text, terminal
events, and incomplete tool calls remain behind a small public-output gate. A
first non-whitespace text delta commits the stream immediately. Tool-call
deltas are committed only after the call is valid. Once committed, the runtime
never retries, so clients cannot receive duplicated text or tool execution.
When an empty attempt is replaced, the recovered deltas continue inside the
single original Chat, Responses, or Messages SSE lifecycle. The complete answer
is never buffered.

Known usage from both attempts is summed. If either attempt omits usage,
completion accounting remains marked incomplete. Responses expose
`x-fusion-recovery-attempts` and `x-fusion-recovered` when the recovery decision
is known before headers, plus a secret-safe `x-fusion-recovery-failure` when a
bounded recovery finishes without public output. Later stream failures remain
visible through the client protocol's native error event.

Output quality and accounting evidence are classified independently. After a
non-streaming call, inspect `result.completion`; after a streaming call, consume
`stream.events` to its terminal event and then inspect `stream.completion.outcome`.
`accounting_complete` is true only when every attempt reports recognizable,
non-negative token counters without a contradictory total. Stable
`accounting_issues` include `usage_missing`, `attempt_usage_missing`,
`usage_tokens_missing`, `usage_value_invalid`, and `usage_total_mismatch`.
Missing usage is never interpreted as zero cost. Protocol-required placeholder
zeros in a compatibility response are not accounting evidence.

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
`stream(model, request) -> AsyncIterator[ModelStreamEvent]`. It publishes the
normalized modes it actually maps, for example
`thinking_modes = frozenset({"provider-default", "bounded"})`; declaring a mode
is a promise to translate `request.thinking` to the upstream protocol. An
unsupported mode must raise `CapabilityError`, not be dropped. The built-in
generic providers intentionally publish only `provider-default`.

A policy implements
`async prepare(runtime, pool_name, request) -> PreparedCall`: experts finish in
`prepare`, while the runtime owns the authoritative final call and any configured
one-shot same-model recovery. Protocol translation stays at the gateway
boundary and must not own routing. A provider without `stream` fails a streaming
request visibly instead of silently falling back to buffered output.

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
