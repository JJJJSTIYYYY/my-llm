<div align="center">

# my-llm-py

[![PyPI](https://img.shields.io/pypi/v/my-llm-py.svg)](https://pypi.org/project/my-llm-py/)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/JJJJSTIYYYY/my-llm-py/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/tUCQthwK5)
[![QQ群](https://custom-icon-badges.demolab.com/badge/QQ群-639459172-00BFFF?style=flat&logo=tencent-qq)](https://qun.qq.com/universal-share/share?ac=1&authKey=ommoQrT2zhzHU%2FUxv8pfGCJbNifW%2BJyUAFBkNdzkHTPUxdxCnlgxm5aNgGslTmdE&busi_data=eyJncm91cENvZGUiOiI2Mzk0NTkxNzIiLCJ0b2tlbiI6Im9ZZkdNUWZnSVV1Y2REeUhKNnlTbWEwc05Bb093djRzUXdXNE55dklBVnlBQk9XbGNpS0ZXSDlzK3orSW1sQ3YiLCJ1aW4iOiIzMTI5NDI0NTcyIn0%3D&data=OGTchcr80RAQg8Z8_GZTdvBb7kZDeM9B3hHcNqLaAX2ZK_KYq260C4CubblEBT1bK5fP6zgtnCk2D8fIoph1ZQ&svctype=4&tempid=h5_group_info)

**通用的 LLM Provider 协议适配器**

</div>

异步调用 OpenAI Chat Completions、OpenAI Responses、Anthropic Messages、Gemini 和
Ollama。请求使用官方 SDK，输入、输出与流事件采用五种协议的标准字典，不另定义消息类型。
需要 Python 3.12+，不内置厂商或模型清单。

## 安装

```sh
pip install my-llm-py
# 或 uv add my-llm-py
```

## 调用

`ProviderProfile.protocol` 选择上游 API，`stdio_protocol` 选择调用者的输入输出协议。
两者独立；例如用 Chat 消息调用 Responses API，并继续收到 Chat 消息。

```python
import asyncio
import os
from my_llm import ModelProfile, MyLLM, ProviderProfile

async def main():
    llm = MyLLM(
        provider=ProviderProfile(
            protocol="openai_responses",
            api_key=os.environ["OPENAI_API_KEY"],
        ),
        model=ModelProfile(model=os.environ["LLM_MODEL"]),
        stdio_protocol="openai_chat",
    )
    message = await llm.generate(messages=[{"role": "user", "content": "你好"}])
    print(message["content"])

asyncio.run(main())
```

五个协议名为 `openai_chat`、`openai_responses`、`anthropic`、`gemini`、`ollama`。
省略 `stdio_protocol` 时，首次调用从整个 messages/tools 列表识别一次并绑定到实例。
纯文本消息等重叠输入默认使用 `openai_chat`；需要其他格式时显式指定。
同一实例的 messages 与 tools 使用同一种协议，不混用历史格式。

```python
async def print_answer(llm):
    # 此示例的 llm 使用 stdio_protocol="openai_chat"。
    stream = await llm.generate(
        messages=[{"role": "user", "content": "你好"}], stream=True,
    )
    async with stream:
        async for event in stream:
            for choice in event.get("choices", []):
                print(choice["delta"].get("content", ""), end="", flush=True)
    return stream.result  # 完成后的标准消息字典
```

流式工具调用完整后才返回；本库不执行工具。SDK Client 每次请求创建并关闭。
Provider、Model 和连接配置在构造时保存副本，生成时仅接受消息、工具和生成选项。
`replay_reasoning_content_when_tool_calls` 在兼容 Chat 服务中需要显式启用（默认 False）；Ollama 默认 True。仅思考已开启且 assistant 有工具调用时补空思维链。

## 独立格式转换

`MyCodec` 无状态，不绑定 profile，可复用于任意转换路径。

```python
from my_llm import MyCodec

codec = MyCodec()
converted = codec.convert(
    [{"role": "user", "content": "你好"}],
    input_format="openai_chat", target_format="gemini",
)
assert converted["messages"] == [{"role": "user", "parts": [{"text": "你好"}]}]
```

消息、工具和流事件各有对称的 encode/decode 接口；`profile=None` 使用标准协议。
同协议往返保留原生签名、未知字段和嵌套字典。

详见[文档目录](docs/README.md)、[API 使用指南](docs/reference/api.md)、
[配置指南](docs/reference/configuration.md)及[架构与测试边界](docs/design/architecture.md)。

## 本地开发与验证

```sh
uv sync --group dev --locked
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run python examples/local_demo.py
uv build
```

[构建与验证指南](docs/guides/building.md) 包含安装和发行包验证步骤。
[本地开发规范](docs/guides/development.md)说明目录、导入与测试辅助代码的约定。
项目采用 [Apache License 2.0](LICENSE)。
