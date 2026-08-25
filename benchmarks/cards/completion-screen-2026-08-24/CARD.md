# Frozen completion screen: provider default vs disabled thinking

Status: **passed screen — advance to a larger evaluation, not production promotion**

This 15-task LiveCodeBench screen tested whether a main model that often spent
its output budget on hidden reasoning could produce more usable public answers.
All three variants used the same tasks, stochastic decode settings, three fixed
seeds, warm serving process, and seeded three-way balanced crossover order.

## Results

| Variant | pass@1 by seed | Total pass@1 | Empty public output | Unextractable code | p95 latency | Mean total tokens | Infra failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Provider-default direct | 7 / 6 / 6 | 19/45 (42.22%) | 26/45 | 26/45 | 292.280 s | 3,071.333 | 0/45 |
| Thinking-disabled direct | 7 / 7 / 6 | 20/45 (44.44%) | 0/45 | 23/45 | 290.595 s | 2,863.911 | 0/45 |
| Thinking-disabled + recovery | 7 / 7 / 6 | 20/45 (44.44%) | 0/45 | 23/45 | 290.669 s | 2,863.911 | 0/45 |

The frozen gate compared the recovery variant with provider-default direct. All
three seeds were non-regressing, aggregate empty public output fell by 100%,
infrastructure failures stayed at zero, mean tokens fell to 0.9325x, and p95
latency was 0.9945x. All calls reported complete usage and the paired run had no
recorded reproducibility issue. The deterministic result is stored in
[`decision.json`](decision.json).

## What the screen actually established

The model exposed a supported `enable_thinking=false` chat-template control but
no verifiable thinking-token budget, so the experiment used the normalized
`disabled` capability and did not call it bounded thinking. The public generic
provider remains model-independent and rejects modes it cannot map.

No recovery was triggered in any of the 45 recovery-variant requests: disabling
thinking always produced some public text. The disabled-direct and recovery
variants were therefore byte-identical apart from normal wall-clock variation.
This card supports disabled thinking as the effective change; it provides no
evidence that recovery adds quality on this task set.

Public output is not the same as executable code. Twenty-three disabled-thinking
answers still lacked a complete extractable fenced program, usually after a
length-limited public response. The pass-rate gain was only one task out of 45.
The next completion candidate should target final-code reservation or safe
truncation handling, while keeping this screen as its parent evidence.

## Frozen identity and limits

- Benchmark: LiveCodeBench `code_generation_lite`, official execution grader
- Tasks: the same 15 records as the earlier pilot; 6 easy, 4 medium, 5 hard
- Seeds: `101`, `202`, `303`; temperature `0.2`; maximum output `4096`
- Task digest: `2c0d398b9efbfb7c4fa60ac149b07d8061bbed2a32497c62459d327c0ea44816`
- Grader commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
- Evaluator retries: none; concurrency: one; cache policy: declared warm

This screen used one model and one serving environment. It is not directly
comparable to the earlier greedy (`temperature=0`) pilot, and 15 tasks cannot
support a general performance or promotion claim. Actual USD cost was
unavailable. Deployment parameters, task contents, private tests, and raw model
answers are intentionally not published; their frozen environment identity is
represented only by a digest.
