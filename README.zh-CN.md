# Fusion MoA

[English](README.md)

给 Coding Agent 一个模型 API，后面可以接你的主模型和可选专家模型。

Fusion MoA 是一个开源、模型无关、GPU 无关的 Mixture-of-Agents 运行时。你可以接入已经在
使用的本地模型或云 API，用一份 recipe 选择编排方式，再把统一的 OpenAI/Anthropic 兼容模型
提供给 Claude Code、Codex、OpenCode、DeepSeek Harness 或其他 Coding Agent。

```text
Coding Agent
    │  OpenAI Chat / Responses / Anthropic Messages
    ▼
Fusion MoA  ── 策略、预算、降级、用量统计
    │
    ├── 权威主模型 ──► 最终答案原生流
    └── 可选只读专家 ─► 只提供私有建议
```

主模型始终是唯一对外输出者。专家不拿 coding tools，输出有硬上限并按不可信数据处理；专家
失败时会安全降级到主模型。项目不绑定模型家族、推理服务、云平台、GPU 或 Coding Agent。

## 现在能做什么

当前社区版本已经提供：

- 一份 `fusion/v1` YAML 统一配置 provider、模型、专家池、策略、完成门和服务入口；
- OpenAI-compatible 与 Anthropic-compatible 上游 provider；
- OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 三套客户端接口；
- 权威主模型最终答案的原生流式，包括工具参数的增量事件；
- function/tool call 回环和 `/v1/models`；
- `direct`、推理预算预留、critic、review-board、自适应 self-review 等策略；
- thinking、工具和 JSON Schema 结构化输出的显式能力检查；
- 有界降级，以及覆盖全部模型调用的 usage 汇总与完整性标记；
- 第三方 provider/policy 的 Python 插件入口；
- 版本锁定的 [DeepSeek Harness 集成](integrations/deepseek-harness/README.md)。

Fusion MoA 仍处于早期阶段。专家编排在某些任务上可能更贵，甚至不如 direct。建议先跑
`direct` 基线，再用自己的任务验证专家是否真的带来收益。

## 快速开始

需要 Python 3.11+ 和至少一个模型 endpoint。第一次运行只要有本地 vLLM、SGLang、llama.cpp，
或任意兼容 OpenAI Chat Completions 的服务即可。

### 1. 安装

```bash
git clone https://github.com/xiaohou521/fusion-moa.git
cd fusion-moa
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

### 2. 配置一个主模型

```bash
cp recipes/direct.yaml fusion.yaml
```

修改 `fusion.yaml` 中的 endpoint 和模型 ID：

```yaml
providers:
  main_api:
    base_url: http://127.0.0.1:8000/v1

models:
  main:
    model: your-coding-model-id
```

密钥只通过环境变量提供，不要写进 YAML：

```bash
export MAIN_MODEL_API_KEY='your-upstream-key'
export FUSION_RUNTIME_API_KEY='choose-a-key-for-coding-agents'
```

如果本地上游不需要鉴权，可以删除对应 provider 的 `api_key_env`。

### 3. 检查配置并启动

```bash
fusion-runtime --config fusion.yaml --check
fusion-runtime --config fusion.yaml --host 127.0.0.1 --port 18888
```

先检查网关：

```bash
curl http://127.0.0.1:18888/health
```

再发一个原生流式请求：

```bash
curl http://127.0.0.1:18888/v1/chat/completions \
  -H "Authorization: Bearer $FUSION_RUNTIME_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "fusion-coding",
    "stream": true,
    "messages": [{"role": "user", "content": "写一个 Python 二分查找。"}]
  }'
```

### 4. 接入 Coding Agent

OpenAI-compatible 客户端填写：

```text
Base URL: http://127.0.0.1:18888/v1
API key:  FUSION_RUNTIME_API_KEY 的值
Model:    fusion-coding
```

Claude Code 通过 CC Switch 接入时，新建一个 Anthropic-compatible provider：

```text
Base URL: http://127.0.0.1:18888
API key:  FUSION_RUNTIME_API_KEY 的值
Model:    fusion-coding
```

Anthropic 客户端会自行追加 `/v1/messages`；OpenAI 客户端使用带 `/v1` 的 Base URL。除非已经
配置 TLS、网络访问控制和足够强的 API key，否则请让 Fusion MoA 只监听本机回环地址。

## 加入一个专家 Reviewer

从自适应 self-review recipe 开始：

```bash
cp recipes/adaptive-self-review.yaml fusion.yaml
export MAIN_MODEL_KEY='your-main-model-key'
export EXPERT_MODEL_KEY='your-expert-model-key'
export FUSION_RUNTIME_API_KEY='choose-a-key-for-coding-agents'
```

修改主模型和专家模型的 endpoint、模型 ID，然后按前面的方式检查并启动。请求路径会变成：

```text
主模型生成有界私有计划
        ▼
一个只读专家做结构化审查
        ▼
主模型原生流式输出最终答案
```

专家 endpoint 必须支持 provider 强制的 JSON Schema，返回 `advise` 或 `abstain`。默认档位是
`512 → 1024 → 2048`，只有确实发生输出截断时才升级。Schema、语义、provider 或网络错误会
立即安全降级，不会靠不断增加 token 掩盖问题。只有主模型最终调用保留 Coding Agent 的工具。

测试时可以查看响应头：

- `x-fusion-route`：实际策略、专家档位和 advise/abstain；
- `x-fusion-fallback`：安全降级原因；
- `x-fusion-streaming-mode: native-final`：确认最终模型原生流式。

## 选择合适的 Recipe

| 起点 | 适合场景 |
| --- | --- |
| [`recipes/direct.yaml`](recipes/direct.yaml) | 最简单、成本最低的基线。 |
| [`recipes/local-main-critic.yaml`](recipes/local-main-critic.yaml) | 用一个有界 critic 审查主模型。 |
| [`recipes/review-board.yaml`](recipes/review-board.yaml) | 多个不同角色的专家并行给建议。 |
| [`recipes/adaptive-reasoning-reserve.yaml`](recipes/adaptive-reasoning-reserve.yaml) | 单模型预留最终答案空间，并按任务选择输出档位。 |
| [`recipes/adaptive-self-review.yaml`](recipes/adaptive-self-review.yaml) | 独立专家审查主模型的有界计划。 |

每份 recipe 都有六层：

1. `providers`：上游传输和密钥环境变量引用；
2. `models`：模型 ID、上限、工具和生成能力；
3. `pools`：一个权威主模型和按角色命名的专家；
4. `policy`：编排逻辑和专家硬预算；
5. `completion`：公共输出要求和可选的有界恢复；
6. `serve`：对外模型名和开启的客户端协议。

模型能力必须显式声明，不能从模型名字猜。只有 provider 插件真的完成了映射，才声明
`disabled` 或 `bounded` thinking；只有上游真正强制 Schema，才声明 `json-schema`。不支持的
组合会在推理前明确失败，不会被静默忽略。

## 插件和集成

第三方 Python 包可以在 `fusion_runtime.providers` 注册 provider，在
`fusion_runtime.policies` 注册策略。Provider 插件负责上游协议翻译，policy 插件负责有界
编排，gateway 负责客户端协议和最终流式输出。

仓库还包含社区维护的
[DeepSeek Harness profile bundle](integrations/deepseek-harness/README.md)。它通过 Harness 的通用
OpenAI-compatible 模型接口接入，不代表 DeepSeek 官方背书。

## 正在开发

当前路线集中在：

- 可断点恢复、模型无关的评测器和公开 Evidence Card；
- direct 与 matched-compute 对照，区分专家价值和单纯增加推理成本；
- 选择性路由，只在预期收益高于成本时调用专家；
- 隐私安全的 outcome 存储、失败聚类和可复用失败分类；
- 带谱系、拒绝、晋升和回滚的离线评测门控 recipe 演化；
- 更多 provider、Coding Agent 和社区专家池插件；
- 独立于核心运行时的可选训练与模型 adapter 插件。

这些目标不表示生产模型现在可以在线改代码、改权重或自行晋升。Fusion MoA 所说的 RSI 是一个
受控离线循环：观察失败 → 提出有界 recipe 候选 → 运行冻结评测 → 明确接受或拒绝。

## 证据、安全与贡献

Fusion 不会天然优于 direct。公开的聚合实验位于
[`benchmarks/cards`](benchmarks/cards)，其中既保留通过的假设，也保留被拒绝的假设；不会包含
私有 endpoint、凭据、原始 prompt 或模型回答。

- 凭据只放环境变量，不放 YAML 和 Git；
- recipe 与已安装插件属于可信部署代码；
- 用户输入、模型输出和专家建议都按不可信数据处理；
- 对外开放网关前先阅读 [SECURITY.md](SECURITY.md)；
- provider、policy、recipe 和评测贡献见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

Apache-2.0。
