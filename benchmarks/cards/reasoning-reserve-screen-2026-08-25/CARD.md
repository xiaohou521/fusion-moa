# Frozen final-code reserve screen

Status: **passed screen — advance to held-out evaluation, not production promotion**

This paired 15-task LiveCodeBench screen compared a thinking-disabled direct
main-model call with a two-phase call to the same main model. The candidate used
at most 256 output tokens for a private explicit outline, then native-streamed
the authoritative final answer with the remaining 3840-token output budget.
Recovery and experts were disabled.

## Results

| Variant | pass@1 by seed | Total pass@1 | Unextractable code | Truncated | Mean prompt tokens | Mean completion tokens | Mean total tokens | p95 latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thinking-disabled direct | 7 / 7 / 6 | 20/45 (44.44%) | 23/45 | 22/45 | 498.533 | 2,365.378 | 2,863.911 | 290.727 s |
| Reasoning reserve, plan 256 | 10 / 10 / 10 | 30/45 (66.67%) | 11/45 | 11/45 | 1,344.311 | 1,757.711 | 3,102.022 | 282.421 s |

The candidate gained ten execution passes, and every seed improved (+3, +3,
+4). Unextractable-code incidence fell by 52.17%. Mean completion tokens fell,
but repeating the prompt for the second call raised mean prompt tokens; aggregate
mean tokens increased by 8.31%. End-to-end p95 latency fell by 2.86%. Both
variants had zero infrastructure failures, zero empty public outputs, complete
all-attempt usage, and complete paired evidence.

The pre-frozen gate required no aggregate execution regression, at least two
non-regressing seeds, at least 25% fewer unextractable answers, no infrastructure
regression, and token and p95-latency ratios no greater than 1.25. The candidate
passed every condition. The deterministic result is stored in
[`decision.json`](decision.json).

## What the screen established

For this model and task mix, a short explicit planning pass improved both final
code completeness and execution correctness while staying within the same 4096
aggregate output-token ceiling. The public stream still came directly from the
authoritative final-model call. The private outline was not emitted to the
client, experts were not consulted, and no hidden chain of thought was read or
stored by the runtime.

The two-call candidate is not compute-free: it repeated the input context. The
card therefore reports merged usage across both attempts, including the prompt
token increase, rather than comparing only final-call tokens.

## Limits and next gate

This is a development screen, not held-out evidence. Two tasks at seed 202 were
used in the preceding plan-budget diagnostic, so the candidate was partly chosen
using this task set. The run is also one model deployment, one benchmark family,
and 45 attempts per variant. It cannot establish cross-model benefit, community
RSI, production reliability, or safety.

The result only permits a larger genuinely held-out evaluation and a user-run
CC Switch to Claude Code deployment test. No production recipe or repository
version is promoted by this card.

## Frozen identity

- Benchmark: LiveCodeBench `code_generation_lite`, official execution grader
- Tasks: 15 records; 6 easy, 4 medium, 5 hard
- Seeds: `101`, `202`, `303`; temperature `0.2`
- Aggregate output ceiling: 4096 tokens for either variant
- Task digest: `2c0d398b9efbfb7c4fa60ac149b07d8061bbed2a32497c62459d327c0ea44816`
- Grader commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
- Concurrency: one; warm cache; seeded two-way balanced crossover

Raw prompts, model responses, private tests, model identity, endpoint details,
and deployment configuration are intentionally excluded from this public card.
