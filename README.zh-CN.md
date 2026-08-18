# Fusion MoA

[English](README.md)

Fusion MoA 是一个面向 coding agent 的、模型/GPU/harness 无关的 MoA
运行时。用户自己接入主模型和可选的只读专家池，用一份 recipe 配置模型、角色、
预算与路由，最终对 Codex、Claude Code、OpenCode、DeepSeek Harness 等客户端只
暴露一个稳定模型 API。

## 当前可用能力

- 严格的 `fusion/v1` YAML 配置与引用、密钥检查；
- OpenAI-compatible、llama.cpp、Anthropic-compatible provider；
- `direct`、单 critic、并行 `review-board` 三种策略；
- 每个模型独立配置上下文、输出、工具、推理、并发等能力；
- OpenAI Chat、Responses、Anthropic Messages 三套入口；
- function/tool call 回环、`/v1/models`、三套协议的最终模型原生 SSE；
- provider/policy Python 插件入口；
- 评测门控的 RSI 晋升命令；
- 按官方 profile bundle 契约制作、版本锁定的 DeepSeek Harness 插件。

流式请求会先完成有界、只读的专家编排，然后把唯一一次权威主模型调用产生的文本、
工具参数、结束原因和 usage 增量映射到客户端协议，不等待完整答案。专家输出不会进入
公共流；不支持原生流式的第三方 provider 会显式失败，不会静默退回伪流式。

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

客户端设置：

```text
Base URL: http://127.0.0.1:18888/v1
Model:    fusion-coding
```

配置分为五层：`providers` 定义传输，`models` 声明能力，`pools` 分配主模型和
专家角色，`policy` 定义编排与预算，`serve` 定义公共模型名和协议。核心不会读取
GPU 型号，也不会从模型名字猜能力；本地模型、云 API 和未来托管专家池可以混用。

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
