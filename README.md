<div align="center">

# my-llm-py

[![PyPI](https://img.shields.io/pypi/v/my-llm-py.svg)](https://pypi.org/project/my-llm-py/)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/JJJJSTIYYYY/my-llm/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/tUCQthwK5)
[![QQ群](https://custom-icon-badges.demolab.com/badge/QQ群-639459172-00BFFF?style=flat&logo=tencent-qq)](https://qun.qq.com/universal-share/share?ac=1&authKey=ommoQrT2zhzHU%2FUxv8pfGCJbNifW%2BJyUAFBkNdzkHTPUxdxCnlgxm5aNgGslTmdE&busi_data=eyJncm91cENvZGUiOiI2Mzk0NTkxNzIiLCJ0b2tlbiI6Im9ZZkdNUWZnSVV1Y2REeUhKNnlTbWEwc05Bb093djRzUXdXNE55dklBVnlBQk9XbGNpS0ZXSDlzK3orSW1sQ3YiLCJ1aW4iOiIzMTI5NDI0NTcyIn0%3D&data=OGTchcr80RAQg8Z8_GZTdvBb7kZDeM9B3hHcNqLaAX2ZK_KYq260C4CubblEBT1bK5fP6zgtnCk2D8fIoph1ZQ&svctype=4&tempid=h5_group_info)

**通用的 LLM Provider 协议适配器**

</div>

my-llm-py 是一个 Python 异步 LLM 协议适配库，支持 OpenAI Chat、OpenAI Responses、Anthropic、Gemini 和 Ollama 五种协议自由组合。

**用一种协议请求另一种协议的 API，按输入协议返回结果。标准协议直接转换，非标准协议通过配置兼容。**

```text
OpenAI Chat 格式输入 → 任意协议 API → OpenAI Chat 格式输出
```

消息、工具和流事件一并转换。更换上游 API，业务层继续使用原有协议收发数据。

## 主要功能与亮点

| 功能 | 说明 |
| --- | --- |
| 跨协议调用 | 标准协议直接转换，输入输出协议与上游 API 独立选择。 |
| 标准消息翻译 | 只翻译已识别的协议字段；工具参数、结果和 JSON Schema 等开放字典保留内部键值。 |
| 统一工具调用 | 转换工具定义、调用和结果；流式参数自动组装，响应成功结束后交付完整调用。工具由应用执行。 |
| 流式与完整回复 | 支持文本、可见思考增量；流结束后通过 `stream.result` 获取完整消息。 |
| 多模态转换 | 支持用户图片、工具结果图片与文本；目标无法表达时按语义损失策略处理。 |
| 配置兼容 | 非标准协议通过配置兼容，适配服务与模型的参数、字段路径及思考内容等差异，配置可拓展。 |
| 独立格式转换 | 无需调用模型，即可转换已支持的消息、工具和完整流事件列表。 |
| 中间模型直连 | 显式 `stdio_protocol="middleware"`，用具体模型收发消息，跳过两端不需要的转换。 |
| 异步并发 | 复用官方 SDK 客户端，请求状态独立；支持取消、提前退出和资源清理。 |
| 协议扩展 | 实现四个抽象基类并注册，即可接入现有调用与转换流程。 |

输入接受标准字典或可转为字典的官方 SDK 对象，输出为所选输入协议的标准字典。
也可[直接使用中间模型](docs/guides/middleware-io.md)，输入 `Message` / `FunctionTool`，返回 `AssistantMessage` / `MessageEvent`。
文本、图片、思维链、工具与 usage 通过具体的 `ProtocolMiddleWare` 翻译；audio、file、用户 metadata 等未支持字段静默丢弃。
已支持字段无法表达时默认抛 `SemanticLossError`，可选择 ignore 或双向原位 hook，详见[迁移指南](docs/guides/semantic-loss.md)。
Profile 用 `Parameter` 声明字段路径、默认值与 source，`Config` 控制代码分支；继承不能扩大固定中间协议的支持集合。
字段 `None` 继承配置；`default_value=None` 是实际值，已有 `None` 不会被默认值覆盖；跨协议来回翻译不保证无损。
跨协议转换受目标协议的表达能力限制，详见[兼容边界](docs/reference/configuration.md#跨协议请求的处理边界)。
新增协议通过 `MyCodec.register(CodecFactory)` 在启动时注册，协议默认值随 Codec 声明；
无需修改公共入口或协议枚举，详见[接入指南](docs/guides/adding-protocol.md)。
参数路径统一使用 $. 前缀，例如 $.reasoning.effort；非标准请求参数通过 SDK 的 extra_body 等正式扩展入口配置。
完整生成输入和已解析配置统一经过 Codec，Decoder 区分客户端参数、请求参数与本地控制项；
连接配置按实例准备，请求配置按调用准备，Transport 负责官方 SDK 调用和响应关闭。
详细职责见[架构文档](docs/design/architecture.md)。

## 安装

需要 **Python 3.12+**。

```sh
pip install 'my-llm-py[all]'
# 或 uv add 'my-llm-py[all]'
```

推荐使用 `[all]` 安装全部 SDK。也可以按需选择，支持用逗号组合多个 extra：

```sh
pip install 'my-llm-py[openai]'
pip install 'my-llm-py[anthropic]'
pip install 'my-llm-py[google-genai]'
pip install 'my-llm-py[ollama]'
pip install 'my-llm-py[openai,anthropic]'
```

`pip install my-llm-py` 仅安装核心依赖。每种输入、输出及上游协议都需要对应 SDK；
例如 Chat 输入调用 Anthropic API 需要 `[openai,anthropic]`。
中间模型 stdio 只需要安装上游对应 SDK。
未安装的协议不会注册，显式选择时会提示安装命令。

项目开发环境执行 `uv sync` 默认安装开发工具和全部 SDK。
仅安装指定 SDK 可使用 `uv sync --no-default-groups --extra openai`。

## 快速开始

以下示例使用 **OpenAI Chat 格式**调用 **Anthropic Messages API**，返回 **Chat assistant 消息**。
设置 `ANTHROPIC_API_KEY` 和 `LLM_MODEL` 环境变量后运行，模型 ID 填写服务支持的值。

```python
import asyncio
import os

from my_llm import Attribute, Config, ModelProfile, MyLLM, Parameter, ProviderProfile


async def main():
    async with MyLLM(
        provider=ProviderProfile(
            protocol=Config(default_value="anthropic"),  # 实际请求的 API 协议
            api_key=Parameter(
                path="$.api_key",
                default_value=os.environ["ANTHROPIC_API_KEY"],
            ),
        ),
        model=ModelProfile(
            model=Attribute(default_value=os.environ["LLM_MODEL"]),
            max_tokens=Parameter(
                path="$.max_tokens",
                default_value=4096,
            ),
        ),
        stdio_protocol="openai_chat",  # 输入和输出均使用 Chat 协议
    ) as llm:
        message = await llm.generate(messages=[{"role": "user", "content": "你好"}])
        print(message["content"])


asyncio.run(main())
```

更换 `provider.protocol` 及对应连接、模型配置，即可切换上游 API；`stdio_protocol` 决定业务层使用的格式。
建议像示例一样显式指定 `stdio_protocol`。省略时仅在消息与工具整批唯一匹配一种协议时自动绑定；
零匹配或多种协议都匹配时抛出 `MessageValidationError`。普通 role/content 文本存在协议歧义，需要显式指定。

### 流式调用

在上例的 `async with MyLLM(stdio_protocol="openai_chat", ...)` 内调用下面的函数，流事件也按 Chat (openai_chat) 协议返回：

```python
async def print_answer(llm):
    stream = await llm.generate(
        messages=[{"role": "user", "content": "你好"}],
        stream=True,
    )
    async with stream:
        async for event in stream:
            for choice in event.get("choices", []):
                print(choice["delta"].get("content", ""), end="", flush=True)
    return stream.result
```

使用 `async with` 管理实例和流。传入工具前，在 `ModelProfile` 中设置
`supports_function_calling=Config(default_value=True)`。

### 独立格式转换

`MyCodec` 可离线转换协议格式，无需 API 密钥：

```python
from my_llm import MyCodec

converted = MyCodec().convert(
    [{"role": "user", "content": "你好"}],
    input_format="openai_chat",
    target_format="gemini",
)
print(converted["messages"])
# [{"role": "user", "parts": [{"text": "你好"}]}]
```

## 支持的协议

以下标识均可用于 `ProviderProfile.protocol` 和 `stdio_protocol`：

| 协议标识 | 对应 API |
| --- | --- |
| `openai_chat` | OpenAI Chat Completions 及兼容服务 |
| `openai_responses` | OpenAI Responses 及兼容服务 |
| `anthropic` | Anthropic Messages |
| `gemini` | Gemini GenerateContent |
| `ollama` | Ollama Chat |

同一实例的消息与工具使用同一种输入输出协议。

### 已通过端到端验证的协议组合

**3×3 九条真实 API 路径验证 (2026-10-20)**。

| 输入/输出协议 ↓ \ 上游 API → | Chat Completions | Responses | Anthropic Messages | Gemini Content | Ollama |
| --- | --- | --- | --- | --- | --- |
| `openai_chat` | ✅ 已验证 | ✅ 已验证 | ✅ 已验证 | ➖ 离线验证 | ➖ 离线验证 |
| `openai_responses` | ✅ 已验证 | ✅ 已验证 | ✅ 已验证 | ➖ 离线验证 | ➖ 离线验证 |
| `anthropic` | ✅ 已验证 | ✅ 已验证 | ✅ 已验证 | ➖ 离线验证 | ➖ 离线验证 |
| `gemini` | ➖ 离线验证 | ➖ 离线验证 | ➖ 离线验证 | ➖ 离线验证 | ➖ 离线验证 |
| `ollama` | ➖ 离线验证 | ➖ 离线验证 | ➖ 离线验证 | ➖ 离线验证 | ➖ 离线验证 |

每条路径覆盖文本汇总、流式增量与汇总一致性，以及工具调用、参数组装和工具结果回传。
汇总模式也通过上游 SSE 执行，本轮未单独验证供应商的 HTTP 非流式响应。
实测中 Chat、Anthropic 的关闭思考参数未完全阻止返回思考块，部分跨协议输出会将其转为可见文本；
上述路径分别校验最终答案与完整流式文本一致性。

接入配置、复跑方法及验证边界见 [SenseNova 端到端测试指南](docs/guides/sensenova.md)，
逐项结果见 [JSON 报告](docs/audits/data/sensenova-e2e-2026-10-20.json)。

## 安装最新的开发版本 (main 分支)

```sh
git clone https://github.com/JJJJSTIYYYY/my-llm.git
cd ./my-llm
uv sync --group dev --locked
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run python examples/local_demo.py
uv build
```

## 文档

- [API 使用指南](docs/reference/api.md)：生成、流式消费、工具回合与格式转换。
- [配置指南](docs/reference/configuration.md)：连接、模型、思考参数与兼容服务。
- [协议模板](docs/reference/schema/README.md)：五种协议的请求、返回和流事件。
- [扩展协议](docs/guides/adding-protocol.md) · [本地开发](docs/guides/development.md) · [构建验证](docs/guides/building.md)
- [全部文档](docs/README.md)

项目采用 [Apache License 2.0](LICENSE)。
