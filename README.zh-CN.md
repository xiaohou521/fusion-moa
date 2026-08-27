# Fusion MoA

[English](README.md)

Fusion MoA 是一个面向 coding agent 的、模型/GPU/harness 无关的 MoA
运行时。用户自己接入主模型和可选的只读专家池，用一份 recipe 配置模型、角色、
预算与路由，最终对 Codex、Claude Code、OpenCode、DeepSeek Harness 等客户端只
暴露一个稳定模型 API。

## 当前可用能力

- 严格的 `fusion/v1` YAML 配置与引用、密钥检查；
- OpenAI-compatible、llama.cpp、Anthropic-compatible provider；
- `direct`、固定/自适应 `reasoning-reserve`、结构化自适应 `self-review`、单 critic、
  并行 `review-board` 策略；
- 每个模型独立配置上下文、输出、工具、推理、并发等能力；
- OpenAI Chat、Responses、Anthropic Messages 三套入口；
- function/tool call 回环、`/v1/models`、三套协议的最终模型原生 SSE；
- 同一原生连接内、公共输出前的一次有界空流恢复；
- provider/policy Python 插件入口；
- 评测门控的 RSI 晋升命令；
- 按官方 profile bundle 契约制作、版本锁定的 DeepSeek Harness 插件。

流式请求会先完成有界、只读的专家编排，然后把权威主模型产生的文本增量映射到客户端
协议，不等待完整答案。专家输出不会进入公共流。如果首轮主模型流完全没有可用公共输出，
显式开启的恢复策略可以在客户端收到终止事件前，用同一主模型的一次有界流替换它；不支持
原生流式的第三方 provider 会显式失败，不会静默退回伪流式。

第三方 provider 的规范化流应发送非空 `TextDelta` / `ToolCallDelta`，再发送且只
发送一个 `Finish`，其后可选发送 `Usage`。若已经输出增量后发生故障，应发送一个
终止型、不得含密钥的 `StreamError`；若首事件之前就失败，则抛出类型化的
`ProviderError`。插件作者可以把这项契约固化到自己的测试中：

```python
from fusion_runtime.conformance import assert_stream_conforms, collect_stream

events = await collect_stream(provider.stream(model, request))
assert_stream_conforms(events)
```

运行时会在发送 HTTP 流式响应头前预取一个规范化事件，因此连接、鉴权、限流以及
初始协议错误仍能返回普通 JSON 错误；已经开始的流则按 Chat、Responses 或
Anthropic Messages 各自的原生错误事件正常终止。预取不会缓冲完整回答。

`FusionRequest.seed` 会由 OpenAI-compatible provider 传给后端，并在内置策略的
主模型与专家调用中保留。Anthropic Messages 没有可移植的同等参数，因此内置
Anthropic provider 会显式拒绝带 seed 的请求；冻结卡必须把它记为可复现性问题，
或选择真正支持 seed 的 provider。

## 快速开始

```bash
git clone https://github.com/xiaohou521/fusion-moa.git
cd fusion-moa
python -m venv .venv
. .venv/bin/activate
pip install .

cp recipes/review-board.yaml my-recipe.yaml
# 修改 endpoint、模型 id，并 export YAML 中引用的密钥环境变量。
fusion-runtime --config my-recipe.yaml --port 18888
```

如果要使用“主模型先做有界计划、一个只读专家再审查”的路线，可从
[`recipes/adaptive-self-review.yaml`](recipes/adaptive-self-review.yaml) 开始修改。

客户端设置：

```text
Base URL: http://127.0.0.1:18888/v1
Model:    fusion-coding
```

配置分为六层：`providers` 定义传输，`models` 声明能力，`pools` 分配主模型和
专家角色，`policy` 定义编排与专家预算，`completion` 定义完成门与恢复预算，`serve`
定义公共模型名和协议。核心不会读取 GPU 型号，也不会从模型名字猜能力；本地模型、
云 API 和未来托管专家池可以混用。

thinking 必须作为生成能力显式声明，不能从模型 id 推断：

```yaml
models:
  main:
    # provider、model、context_window 等字段
    generation:
      thinking:
        modes: [provider-default, disabled, bounded]
      structured_output:
        modes: [json-schema]
      final_answer_reserve: true
```

`provider-default` 不发送规范化 thinking 覆盖。只有模型声明和 provider 插件同时支持
时，才可请求 `disabled` 或 `bounded`；bounded 请求还必须提供正整数 token 预算，且
不能超过有效输出上限。它与 OpenAI 特有的 `reasoning_effort` 透传是两套不同的能力。

第三方 provider 通过类似
`thinking_modes = frozenset({"provider-default", "bounded"})` 的属性声明自己真正完成
映射的模式，并负责把 `request.thinking` 翻译成上游协议。声明就是插件合同：不支持时
必须抛出 `CapabilityError`，不能静默丢弃。内置通用 provider 目前有意只声明
`provider-default`。

结构化输出也必须显式声明。`json-schema` 表示模型实际 endpoint 接受由 provider 强制的
JSON Schema，且 provider 插件会映射规范化的 `StructuredOutputConfig`，并不表示仅靠
prompt 就能稳定输出 JSON。内置 OpenAI-compatible provider 会把它映射为
`response_format.type=json_schema`；内置 Anthropic-compatible provider 暂未映射，
会在请求上游前显式失败。第三方 provider 应声明
`structured_output_modes = frozenset({"json-schema"})` 并完成真实翻译，不能静默丢弃。

上游不能限制隐藏思考预算时，可以用内置 `reasoning-reserve` 把预算拆成一次有界私有
规划和一次权威最终回答。若不同任务所需的最终答案长度差异很大，可使用自适应形式：

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

私有规划的第一个非空行必须严格等于 `OUTPUT_BUDGET: base` 或
`OUTPUT_BUDGET: extended`。marker 缺失、格式错误、重复、互相矛盾或规划调用失败时，
策略都会 fail closed 到 base。marker 不会进入最终上下文；规划正文会被限长、转义并
标记为非权威内容，策略本身不持久化规划文本。

`base_total_tokens` 和 `extended_total_tokens` 是“私有规划 + 最终回答”的总上限。
实际选择值还会被模型声明的 `max_output` 和客户端请求的 `max_tokens` 同时限制，策略
不会越过任一硬上限。客户端上限不允许扩容时仍走 base，并显式给出降级原因。最终路线
通过 `x-fusion-route` 暴露为 `adaptive-reasoning-reserve-base` 或
`adaptive-reasoning-reserve-extended`，fail-closed 原因通过 `x-fusion-fallback` 暴露。
规划调用不带工具；只有权威主模型最终调用保留工具，并作为原生流交付。两次调用的 usage
会合并，任一次缺失 usage 都会使 accounting 不完整。

固定专家上限对短任务浪费、对长任务又不够时，可以使用实验性的
`adaptive-self-review`：

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

专家只能返回 schema 强制的 `advise` 或 `abstain` envelope。只有 finish reason 明确表示
长度截断，或 JSON 无效且 usage 显示输出 token 已到当前档位时，才会进入下一档。
schema、语义、能力、传输和 provider 错误立即 fail closed，不会仅靠反复增加长度掩盖
问题。档位还受专家模型 `max_output` 的硬限制；例如配置 `[512, 1024, 2048]`、模型上限
为 768 时，实际档位为 `[512, 768]`。默认最多尝试三档，专家 completion token 的聚合
硬上限是 3584。

自计划和专家审查都属于不带工具的私有准备调用；审查内容经转义后作为不可信上下文交给
主模型，只有主模型保留原始工具并产生公共原生流。`x-fusion-route` 会显示类似
`adaptive-self-review-b512-advise` 的实际档位与动作，错误通过
`x-fusion-fallback` 显式暴露。计划和全部专家尝试的 usage 都计入总账；任一次 usage 缺失
或矛盾，accounting 就不完整。这个策略目前是候选机制，不代表通用性能提升，用户仍需在
自己的冻结任务上与 direct、matched-compute 做真实部署验证。

空完成恢复必须显式开启，并且有硬上限：

```yaml
completion:
  require_public_output: true
  require_tool_or_text: true
  max_recovery_attempts: 1
  recovery_max_tokens: 2048
```

`max_recovery_attempts` 默认是 `0`，所以旧 recipe 不会在升级后改变行为。开启后，只有既
没有可用文本、也没有有效工具调用的尝试才会被替换；运行时仍调用同一个权威主模型，最多
一次，不重新运行专家，不重放 provider 的隐藏 reasoning，而且客户端和模型的输出上限
仍是硬约束。

对于流式请求，前导控制事件、纯空白文本、终止事件和未完成工具调用会暂时停留在一个很小
的公共输出门内。首个非空白文本 delta 会立即提交；工具调用只有通过完整性校验后才提交。
一旦提交，运行时绝不重试，因此客户端不会收到重复文本或重复工具执行。首轮为空时，恢复
流继续使用原来的 Chat、Responses 或 Messages SSE 生命周期，不会缓冲完整回答。

两次调用中已知的 usage 会相加；任何一次缺少 usage，accounting 都不会被标记为完整。
恢复决定能在响应头之前确定时，会提供 `x-fusion-recovery-attempts` 和
`x-fusion-recovered`；有界恢复仍无公共输出时还会提供不含敏感信息的
`x-fusion-recovery-failure`。响应开始后的故障继续使用各客户端协议原生错误事件表达。

输出质量和 accounting 证据会分别分类。非流式调用结束后读取
`result.completion`；流式调用需要先消费到 `stream.events` 的终止事件，再读取
`stream.completion.outcome`。只有每次尝试都报告可识别、非负且总数不矛盾的 token
计数时，`accounting_complete` 才为 true。稳定的 `accounting_issues` 包括
`usage_missing`、`attempt_usage_missing`、`usage_tokens_missing`、
`usage_value_invalid` 和 `usage_total_mismatch`。缺失 usage 绝不会被解释为零成本；
兼容协议为满足 schema 输出的占位零值也不构成 accounting 证据。

## 安全边界

专家只提供建议：不拿 coding tools，输出会被限长并标记为不可信；主模型是唯一
可以给客户端返回文本或工具调用的模型。专家故障会显式降级并记录在响应头中。

RSI 指对 recipe、prompt、预算、停止规则和完成门做“离线候选→冻结评测→canary→
晋升/回滚”。生产模型不能在线修改代码、配置或权重。运行：

```bash
fusion-runtime-gate --baseline direct.json --candidate candidate.json
```

只有质量、延迟、成本、基础设施失败率和可复现性全部过门才返回成功。
摘要可记录主模型与全部专家调用的 `mean_total_tokens`，门策略可设置
`max_mean_token_ratio`；只要 `reproducibility_issues` 非空，晋升就会 fail closed。

第一张 [LiveCodeBench 冻结 pilot 卡](benchmarks/cards/lcb-pilot-2026-08-19/CARD.md)
据此拒绝了默认 review-board，继续保留 direct。公开卡只含聚合证据和限制，不含
私有部署配置或原始模型回答。

DeepSeek Harness 的安装见
[`integrations/deepseek-harness`](integrations/deepseek-harness)。官方目前处于 developer
preview，因此插件锁定了兼容版本，不应宣传为 DeepSeek 官方背书。

社区贡献和安全报告分别见 [`CONTRIBUTING.md`](CONTRIBUTING.md)、
[`SECURITY.md`](SECURITY.md)。
