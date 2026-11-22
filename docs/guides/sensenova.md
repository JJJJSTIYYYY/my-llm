# SenseNova 接入

SenseNova 的三种接口使用现有 `ProviderProfile` 配置接入。业务调用仍使用
`MyLLM.generate()`，不需要自定义 Codec、Transport 或外层转换。
下列配置与仓库的 [SenseNova 示例](../../examples/sensenova_e2e.py) 一致；
测试复现见[端到端测试指南](../testing/sensenova.md)，历史结果见
[历史测试结果](../testing/results/2026-10-sensenova.md)。

| 上游 API | `protocol` | `base_url` | 接入差异 |
| --- | --- | --- | --- |
| Chat Completions | `openai_chat` | `https://token.sensenova.cn/v1` | 输出上限为 `$.max_tokens`；思维增量与回传值从 `$.choices[0].delta.reasoning` 读取 |
| Responses | `openai_responses` | `https://token.sensenova.cn/v1` | 输出上限为 `$.max_output_tokens`；原生思考参数放在 `reasoning` |
| Anthropic Messages | `anthropic` | `https://token.sensenova.cn` | 使用 `Authorization: Bearer ...`；输出上限为 `$.max_tokens` |

## Chat 上游

安装 `my-llm-py[openai]`，在运行环境中提供 `SENSENOVA_API_KEY` 和 `LLM_MODEL`。
模型 ID 填写该服务支持的值，密钥不写入源码、仓库配置或命令行参数。

```python
import asyncio
import os

from my_llm import (
    Attribute,
    Config,
    InputParameter,
    ModelProfile,
    MyLLM,
    OutputParameter,
    ProviderProfile,
)

provider = ProviderProfile(
    protocol=Config(default_value="openai_chat"),
    base_url=Config(default_value="https://token.sensenova.cn/v1"),
    api_key=InputParameter(path="$.api_key", default_value=os.environ["SENSENOVA_API_KEY"]),
    max_tokens=InputParameter(path="$.max_tokens", default_value=1024),
    reasoning_effort=InputParameter(path="$.reasoning_effort"),
    reasoning_content=OutputParameter(path="$.choices[0].delta.reasoning", default_value=None),
    feedback_reasoning_item=OutputParameter(path="$.choices[0].delta.reasoning"),
    extra_body=Config(default_value={"reasoning_effort": "none"}),
)


async def main():
    llm = MyLLM(
        provider=provider,
        model=ModelProfile(model=Attribute(default_value=os.environ["LLM_MODEL"])),
        stdio_protocol="openai_chat",
    )
    reply = await llm.generate(messages=[{"role": "user", "content": "你好"}])
    print(reply["content"])


asyncio.run(main())
```

## Responses 上游

同样使用 `[openai]`。将上例的 `provider` 替换为以下配置，业务仍可保持 Chat 输入输出：

```python
provider = ProviderProfile(
    protocol=Config(default_value="openai_responses"),
    base_url=Config(default_value="https://token.sensenova.cn/v1"),
    api_key=InputParameter(path="$.api_key", default_value=os.environ["SENSENOVA_API_KEY"]),
    max_tokens=InputParameter(path="$.max_output_tokens", default_value=1024),
    extra_body=Config(default_value={"reasoning": {"effort": "none"}}),
)
```

如果业务也使用 Responses 格式，将 `stdio_protocol` 设为 `openai_responses`，
消息按该协议传入。完整回复是 output items 列表，追加历史时使用 `history.extend(reply)`。

## Anthropic 上游

安装 `[anthropic]`；业务使用 Chat 格式时，还需 `[openai]`，可直接安装 `[all]`。
该服务使用 Bearer 认证，因此将密钥路由到认证头，并显式包含前缀：

```python
provider = ProviderProfile(
    protocol=Config(default_value="anthropic"),
    base_url=Config(default_value="https://token.sensenova.cn"),
    api_key=InputParameter(
        path="$.extra_headers.Authorization",
        default_value=f"Bearer {os.environ['SENSENOVA_API_KEY']}",
    ),
    max_tokens=InputParameter(path="$.max_tokens", default_value=1024),
    extra_body=Config(default_value={"thinking": {"type": "disabled"}}),
)
```

使用该配置时，`generate(api_key=...)` 也须传完整的 `Bearer ...` 字符串。
调用级密钥沿用配置的同一条路径，只覆盖本次调用。

## 工具、思考与流式结果

使用工具时，在 `ModelProfile` 中显式设置
`supports_function_calling=Config(default_value=True)`，工具由应用执行并回传结果。
流式调用使用 `async with stream`，完整历史保存 `stream.result`，不要仅保存展示文字。
调用和工具回合示例见 [API 使用指南](../reference/api.md)。

上面的原生参数请求关闭思考；已有实测中，上游仍可能返回思考块。
跨协议目标无法保存原生思考签名或加密项时，库默认抛出 `SemanticLossError`。
只有接受这些信息损失时，才选择 `semantic_loss_protocol="ignore"` 或显式 hook，
处理规则见[语义损失指南](semantic-loss.md)。启用思考时应根据服务与模型能力配置参数、
输出预算和回传策略，不能通过自定义转换伪造其他协议的签名。
