# Frozen output-budget ablation

Status: **completed — retain 4096 under the frozen sequential gate; no promotion**

This 15-task LiveCodeBench development ablation held the main model, explicit
256-token plan, tasks, seeds, temperature, grader, and runtime source constant.
It varied only the aggregate output-token ceiling: 4096, 8192, or 16384 tokens.
The authoritative final call remained a native stream, thinking was disabled,
and experts and recovery were disabled.

## Results

| Aggregate ceiling | Final-call ceiling | pass@1 by seed | Total pass@1 | Unextractable code | Truncated | Mean total tokens | p95 latency |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 3,840 | 10 / 10 / 10 | 30/45 (66.67%) | 11/45 | 11/45 | 3,102.022 | 282.421 s |
| 8,192 | 7,936 | 12 / 11 / 11 | 34/45 (75.56%) | 3/45 | 3/45 | 3,611.644 | 729.769 s |
| 16,384 | 16,128 | 12 / 11 / 12 | 35/45 (77.78%) | 0/45 | 0/45 | 3,831.044 | 1,026.326 s |

Larger ceilings improved execution correctness and answer completeness. Moving
from 4K to 8K gained four execution passes and reduced unextractable answers by
72.73%, while increasing mean total tokens by 16.43%. Moving from 8K to 16K
gained one further pass and removed the remaining three truncations, with a
further 6.07% mean-token increase.

## Frozen sequential decision

Selection began at the smallest ceiling and could move only to an adjacent
higher ceiling. Each step required no aggregate pass regression, at least two
non-regressing seeds, no increase in unextractable answers, a meaningful gain
(at least two passes or at least 25% fewer unextractable answers), no
infrastructure regression, and mean-token and p95-latency ratios no greater than
1.50.

The 4K-to-8K step met every quality, completeness, token, and infrastructure
condition. Its seed pass deltas were +2, +1, and +1. It failed only the latency
condition: p95 latency increased by 2.584x. The pre-frozen rule therefore stops
at 4K. The deterministic result is stored in
[`decision.json`](decision.json).

The 8K-to-16K figures remain useful descriptive evidence: pass@1 increased by
one, all remaining truncations disappeared, the mean-token ratio was 1.061, and
the p95-latency ratio was 1.406. They cannot be used to skip the failed first
step or retroactively select 16K.

## What this establishes

4096 is an experimental and policy budget, not a runtime requirement. A larger
ceiling can recover otherwise truncated code for this model and task mix. It
also permits very long generations that materially increase tail latency. There
is no single universally optimal ceiling in this evidence: 4K is the frozen
latency-constrained selection, while 8K and 16K expose a quality/latency Pareto
tradeoff.

The next candidate should test an explicit adaptive budget policy or documented
quality/latency profiles rather than silently raising every request's default.
That candidate needs its own frozen, matched-compute evaluation and must preserve
one authoritative public stream. No adaptive policy is validated by this card.

## Limits

This is a development ablation, not held-out evidence. The 4K cells are exact
reused evidence from the preceding reserve screen; the 8K and 16K cells were
interleaved in a new seeded crossover. Consequently, the 4K-to-8K latency ratio
includes cross-run service variation. The frozen gate deliberately applies it,
but the ratio should be reconfirmed in a single held-out crossover before any
product decision.

Two tasks in this set were previously used to choose the 256-token plan budget.
The evidence also covers only 15 tasks, three seeds, one model deployment, and
one benchmark family. It does not establish cross-model behavior, coding-agent
UX, actual dollar cost, production reliability, or production promotion.

## Frozen identity

- Benchmark: LiveCodeBench `code_generation_lite`, official execution grader
- Tasks: 15 records; 6 easy, 4 medium, 5 hard
- Seeds: `101`, `202`, `303`; temperature `0.2`
- Plan ceiling: 256 tokens for every variant
- Task digest: `2c0d398b9efbfb7c4fa60ac149b07d8061bbed2a32497c62459d327c0ea44816`
- Grader commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
- Concurrency: one; warm cache; no retries

Raw prompts, model responses, private tests, model identity, endpoint details,
and deployment configuration are intentionally excluded from this public card.
