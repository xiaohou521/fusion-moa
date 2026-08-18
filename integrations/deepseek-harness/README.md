# Fusion Runtime for DeepSeek Harness

This directory is a DeepSeek Harness profile bundle. It points the official
`@deepseek-ai/dsh-llm-pi-ai` OpenAI-compatible adapter at Fusion Runtime and
selects `fusion-coding` as the default model.

Compatibility is intentionally pinned to DeepSeek Harness
`@deepseek-ai/dsh-llm-pi-ai` `0.1.0-rc.7`. DeepSeek Harness is in developer
preview, so review this patch and update the pin after upstream breaking
changes.

Start Fusion Runtime first, then install the local bundle into the profile you
use:

```bash
export FUSION_RUNTIME_BASE_URL=http://127.0.0.1:18888/v1
# Must match serve.api_key_env. For an unauthenticated local runtime, the
# upstream adapter still needs a non-empty placeholder.
export FUSION_RUNTIME_API_KEY=local

dsh plugin --profile web add ./integrations/deepseek-harness
dsh web
```

The bundle contains no model implementation and no GPU assumptions. The
selected Fusion recipe decides which local or hosted main/expert models serve
the request.
