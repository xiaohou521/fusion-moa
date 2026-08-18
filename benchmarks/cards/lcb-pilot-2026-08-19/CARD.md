# Frozen LiveCodeBench pilot: direct vs review-board

Status: **rejected — keep the direct route**

This is a 15-task engineering pilot, not a general performance claim. It compares
one direct main-model call with two parallel, read-only expert calls followed by
one authoritative call to the same main model. The task set, main-model decode
settings, seed, and balanced crossover order were frozen before the run.

## Results

| Route | pass@1 | p95 latency | Mean total tokens | Infra failures | Empty final code |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 7/15 (46.67%) | 291.295 s | 2,998.267 | 0/15 | 8/15 |
| review-board | 6/15 (40.00%) | 329.753 s | 5,613.067 | 0/15 | 9/15 |

The review board lost one paired task, won none, and tied on the other 14. Its
pass rate decreased by 6.67 percentage points, p95 latency was 1.132x the direct
route, and mean total tokens were 1.872x. Every direct task made one model call;
every review-board task made two expert calls and one main call. No route fell
back and every reported call contained non-zero usage.

The frozen promotion policy rejected the candidate because quality decreased
and no gated metric improved. See [`decision.json`](decision.json).

## Frozen identity

- Benchmark: LiveCodeBench code generation, official execution grader
- Tasks: 15 complete records (6 easy, 4 medium, 5 hard)
- Seed: `20260819`
- Task-set digest: `2c0d398b9efbfb7c4fa60ac149b07d8061bbed2a32497c62459d327c0ea44816`
- Environment digest: `130ce815b2979964e5ef200114e5689eebe069139cb5a5865ca0f19170731de0`
- Candidate order: seeded, balanced crossover
- Retries: none

## Limits and interpretation

The serving processes were already warm and caches were not reset per route.
This was a single seeded run, so repeat-run determinism was not measured. Actual
USD cost was unavailable; `mean_cost_usd: 0` is a schema placeholder, not a
claim that local GPU inference has no cost. Deployment endpoints, hardware
parameters, raw tasks, and generated answers are intentionally not published.

Eight direct answers and nine review-board answers contained no extractable
final code after the model consumed its output budget. The next experiment
should therefore change one variable first: enforce a final-answer completion
budget (or bounded/non-thinking decode), freeze a new direct baseline, and only
then retest selective expert routing. A larger card should run only after that
candidate wins repeated seeds on this pilot.
