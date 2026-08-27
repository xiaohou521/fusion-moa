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
- `direct`, fixed/adaptive `reasoning-reserve`, structured adaptive
  `self-review`, `main-critic`, and parallel `review-board` policies;
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
For a single main model with length-aware budget routing, start from
[`recipes/adaptive-reasoning-reserve.yaml`](recipes/adaptive-reasoning-reserve.yaml).
To add one read-only reviewer after a bounded main-model plan, start from
[`recipes/adaptive-self-review.yaml`](recipes/adaptive-self-review.yaml).

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
      structured_output:
        modes: [json-schema]
      final_answer_reserve: true
```

`provider-default` sends no normalized thinking override. `disabled` and
`bounded` may be used only when both the model declaration and its provider
plugin support them; bounded requests also carry a positive token budget no
larger than the effective output limit. This capability is separate from the
OpenAI-specific `reasoning_effort` passthrough.

Structured output is explicit as well. `json-schema` means the model accepts a
provider-enforced JSON Schema request and the provider adapter maps the
normalized `StructuredOutputConfig`; it is not a claim that prompt-only JSON is
reliable. The built-in OpenAI-compatible provider maps this mode to
`response_format.type=json_schema`. The built-in Anthropic-compatible provider
does not currently map it and fails before making an upstream request. Declare
the mode only when the model's actual endpoint supports the contract.

When a provider cannot enforce a hidden-reasoning budget, the built-in
`reasoning-reserve` policy can make the budget split explicit:

```yaml
policy:
  type: reasoning-reserve
  options:
    plan_max_tokens: 256
    final_answer_min_tokens: 3072
    max_plan_chars: 4000
    plan_thinking_mode: disabled
    final_thinking_mode: disabled
```

The authoritative main model first produces one bounded private outline with
tools removed. The remaining output budget is then reserved for a second call
to the same main model, whose final answer or tool call is the only call exposed
as the native public stream. The outline is bounded, escaped, marked as
non-authoritative context, and never persisted by the policy. If planning fails
or returns no outline, the reserved final call still runs and the fallback is
visible. `plan_max_tokens + final_answer_min_tokens` must fit the request/model
output limit; the policy never silently shrinks the declared final reserve.

Both calls' usage is merged into the public total and missing usage from either
call leaves accounting incomplete. The output-token split does not make the
two-call total compute free: the final call repeats the input context, so frozen
evaluation must still gate aggregate input tokens and latency. A provider plugin
must explicitly map the configured thinking modes; the built-in generic
OpenAI-compatible provider does not guess a model-specific disable switch.

For requests whose required answer length varies widely, the adaptive form lets
the same bounded plan select between two aggregate ceilings:

```yaml
policy:
  type: adaptive-reasoning-reserve
  options:
    plan_max_tokens: 256
    final_answer_min_tokens: 3072
    base_total_tokens: 4096
    extended_total_tokens: 16384
    max_plan_chars: 4000
    plan_thinking_mode: disabled
    final_thinking_mode: disabled
```

The plan's first non-empty line must be exactly `OUTPUT_BUDGET: base` or
`OUTPUT_BUDGET: extended`. Missing, malformed, duplicate, or contradictory
markers fail closed to the base tier. Planning failures do the same. The marker
is removed before the bounded outline is passed to the final call, and the plan
is never persisted by the policy.

`base_total_tokens` and `extended_total_tokens` include both the private plan
and the final answer. The selected total is always capped by both the model's
declared `max_output` and the client request's `max_tokens`; neither setting can
override those hard limits. If the extended signal cannot increase the budget,
the base route is used and the reason is surfaced. The stable routes are
`adaptive-reasoning-reserve-base` and
`adaptive-reasoning-reserve-extended`, visible through `x-fusion-route`; a
fail-closed reason is visible through `x-fusion-fallback`. Tools remain removed
from planning and available to the authoritative native-streamed final call.

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

For tasks where one fixed expert ceiling is either wasteful or too short, the
experimental `adaptive-self-review` policy combines a bounded main-model plan
with one structured independent review:

```yaml
policy:
  type: adaptive-self-review
  max_expert_calls: 1
  options:
    expert_role: reviewer
    self_plan_max_tokens: 256
    expert_token_tiers: [512, 1024, 2048]
    max_advice_chars: 1600
    self_plan_thinking_mode: provider-default
    expert_thinking_mode: provider-default
    final_thinking_mode: provider-default
```

The expert must return exactly one schema-constrained `advise` or `abstain`
envelope. A higher tier is attempted only after an explicit length finish, or
after invalid JSON whose reported output-token count has reached the requested
tier. Schema, semantic, capability, transport, and provider errors fail closed
without buying more tokens. The configured tiers are bounded by the expert
model's declared `max_output`; for example, `[512, 1024, 2048]` becomes
`[512, 768]` when that hard limit is 768. At most one call is made per usable
tier, so the default aggregate expert completion ceiling is 3584 tokens.

The self-plan and review are private preparation calls with tools removed. The
review is escaped and injected as untrusted context; only the authoritative
main model retains the original tools and emits the native public stream.
Routes such as `adaptive-self-review-b512-advise` and
`adaptive-self-review-b1024-abstain` expose the selected tier and action.
Preparation usage is merged across the plan and every expert attempt; missing
or contradictory usage keeps completion accounting incomplete. This policy is
a candidate mechanism, not a general quality claim; deploy and validate it
against direct and matched-compute routes on your own frozen workload.

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

Providers that accept normalized schema-constrained requests also publish, for
example, `structured_output_modes = frozenset({"json-schema"})`, and translate
`request.structured_output` or raise `CapabilityError`. They must not silently
drop an unsupported structured-output request.

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

The follow-up [completion screen](benchmarks/cards/completion-screen-2026-08-24/CARD.md)
compares provider-default direct inference with disabled thinking and bounded
empty-output recovery across 15 tasks and three seeds. It passed its frozen
screening gate, which means only that the candidate may advance to a larger
held-out evaluation; the card explicitly does not claim production promotion.

The next [final-code reserve screen](benchmarks/cards/reasoning-reserve-screen-2026-08-25/CARD.md)
compares thinking-disabled direct inference with a 256-token private outline and
a hard-reserved native-streamed final call. The development screen improved
execution pass@1 and answer completeness, while reporting the repeated-prompt
overhead and disclosing that two diagnostic tasks were used to choose the plan
budget. Its result still requires genuinely held-out and coding-agent validation.

The follow-on [output-budget ablation](benchmarks/cards/output-budget-ablation-2026-08-25/CARD.md)
holds that policy fixed and compares 4K, 8K, and 16K aggregate output ceilings.
Higher ceilings recover more complete and correct code, but the frozen sequential
gate retains 4K because the 4K-to-8K p95-latency ratio exceeds its limit. The card
reports the full quality/latency tradeoff and does not treat 4096 as a runtime
requirement or a universal optimum.

## License

Apache-2.0.
