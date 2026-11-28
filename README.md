<div align="center">

# my-llm-py

[![PyPI](https://img.shields.io/pypi/v/my-llm-py.svg)](https://pypi.org/project/my-llm-py/)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/JJJJSTIYYYY/my-llm/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/tUCQthwK5)
[![QQ群](https://custom-icon-badges.demolab.com/badge/QQ群-639459172-00BFFF?style=flat&logo=tencent-qq)](https://qun.qq.com/universal-share/share?ac=1&authKey=ommoQrT2zhzHU%2FUxv8pfGCJbNifW%2BJyUAFBkNdzkHTPUxdxCnlgxm5aNgGslTmdE&busi_data=eyJncm91cENvZGUiOiI2Mzk0NTkxNzIiLCJ0b2tlbiI6Im9ZZkdNUWZnSVV1Y2REeUhKNnlTbWEwc05Bb093djRzUXdXNE55dklBVnlBQk9XbGNpS0ZXSDlzK3orSW1sQ3YiLCJ1aW4iOiIzMTI5NDI0NTcyIn0%3D&data=OGTchcr80RAQg8Z8_GZTdvBb7kZDeM9B3hHcNqLaAX2ZK_KYq260C4CubblEBT1bK5fP6zgtnCk2D8fIoph1ZQ&svctype=4&tempid=h5_group_info)

**通用的 LLM Provider 协议适配器**

</div>

my-llm-py 是一个 Python 异步 LLM 协议适配库，支持 OpenAI Chat、OpenAI Responses、Anthropic 三种协议自由组合。

**用一种协议请求另一种协议的 API，按输入协议返回结果。标准协议直接转换，非标准协议通过配置兼容。**

```text
OpenAI Chat 格式输入 → 任意协议 API → OpenAI Chat 格式输出
```

消息、工具和流事件一并转换。更换上游 API，业务层继续使用原有协议收发数据。
本项目面向调用方使用固定消息协议的场景；`MyLLM` 按 `stdio_protocol` 返回对应协议的消息，
便于直接续接同一协议的历史。不支持混合协议历史输入，详见[输入输出协议约定](docs/reference/api.md#固定协议与历史输入)。
业务调用使用 `MyLLM.generate` / `generate_json`；离线翻译使用 `MyCodec.encode` / `decode` / `convert`，各自可独立使用。
请求准备和原生协议组件属于内部实现，不作为独立业务入口。

## 主要功能

| 功能 | 说明 |
| --- | --- |
| 跨协议调用 | 输入输出协议与上游 API 独立选择，消息、工具和流事件一并转换。 |
| 标准输入输出 | 接受标准字典或官方 SDK 对象，完成消息随输入选择字典／SDK 对象表示；流事件保持原格式。 |
| 统一工具调用 | 转换工具定义、调用和结果；流式参数自动组装，成功结束后交付完整调用，由应用执行。 |
| 流式与完整回复 | 支持文本、可见思考增量，流结束后通过 `stream.result` 获取完整消息。 |
| 多模态转换 | 支持文本、用户图片、工具结果图片、思维链及 usage；转换受[目标协议能力](docs/reference/configuration.md#跨协议请求的处理边界)限制。 |
| 语义损失处理 | 已支持内容无法表达时默认抛 `SemanticLossError`，可选 `ignore` 或[双向 hook](docs/guides/semantic-loss.md) 补偿。 |
| Profile 配置兼容 | 用 `InputParameter` / `OutputParameter` 配置字段路径与默认值，`Config` 控制行为；[适配服务与模型差异](docs/reference/configuration.md)。 |
| 中间模型直连 | 显式 `stdio_protocol="middleware"`，直接用[中间模型](docs/guides/middleware-io.md)收发消息、工具与流事件。 |
| 增量消息缓存 | 缓存已翻译的增量历史，连续重发相同批次不重复追加；开启后同实例串行使用，见[缓存约定](docs/reference/api.md#增量消息缓存)。 |
| 异步与资源管理 | 关闭消息缓存时支持同实例并发；支持取消、提前退出和资源清理，见[架构职责](docs/design/architecture.md)。 |
| 独立格式转换 | 无需调用模型，即可转换已支持的消息、工具和完整流事件列表。 |
| 协议扩展 | 实现四个抽象基类，通过 `MyCodec.register(CodecFactory)` 注册；无需修改公共入口，见[接入指南](docs/guides/adding-protocol.md)。 |
| 字段保留边界 | 只翻译已识别字段；工具参数、结果和 JSON Schema 等开放字典保留内部键值；audio、file、用户 metadata 等未支持字段静默丢弃。 |

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

from my_llm import Attribute, Config, ModelProfile, MyLLM, InputParameter, ProviderProfile


async def main():
    llm = MyLLM(
        provider=ProviderProfile(
            protocol=Config(default_value="anthropic"),
            api_key=InputParameter(path="$.api_key", default_value=os.environ["ANTHROPIC_API_KEY"]),
        ),
        model=ModelProfile(
            model=Attribute(default_value=os.environ["LLM_MODEL"]),
            max_tokens=InputParameter(path="$.max_tokens", default_value=4096),
        ),
        stdio_protocol="openai_chat",
    )
    message = await llm.generate(messages=[{"role": "user", "content": "你好"}])
    print(message["content"])


asyncio.run(main())
```

通过 `provider.protocol` 配置默认上游 API，`model.protocol` 可覆盖它；`stdio_protocol` 决定业务层使用的格式。
建议像示例一样显式指定 `stdio_protocol`。省略时仅在消息与工具整批唯一匹配一种协议时自动绑定；
零匹配或多种协议都匹配时抛出 `MessageValidationError`。普通 role/content 文本存在协议歧义，需要显式指定。

### 流式调用

将上例的 MyLLM 实例设为 `stdio_protocol="openai_chat"` 后调用下面的函数，流事件也按 Chat 协议返回：

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

使用 `async with` 管理响应流，MyLLM 实例无需关闭。传入工具前，在 `ModelProfile` 中设置
`supports_function_calling=Config(default_value=True)`。

设置 `enable_message_cache=True` 可缓存已翻译的增量消息；连续重发相同批次不会重复追加历史。
启用后，同一 `MyLLM` 实例不支持并发：先完成或关闭前一条流，再调用下一次生成。
不同实例可以并发使用。用法和重试约定见[增量消息缓存](docs/reference/api.md#增量消息缓存)。

### 独立格式转换

`MyCodec` 可离线转换协议格式，无需 API 密钥：

```python
from my_llm import MyCodec

converted = MyCodec().convert(
    [{"role": "user", "content": "你好"}],
    input_format="openai_chat",
    target_format="anthropic",
)
print(converted["messages"])
# [{"role": "user", "content": [{"type": "text", "text": "你好"}]}]
```

完整响应流也可离线翻译：`encode(chunks=完整原生流, input_format=...)` 返回公开 `MessageEvent` 列表，
保存在 `ProtocolMiddleWare.stream`；`decode(chunks=完整中间事件, target_format=...)` 可单独调用，
也支持事件经 `to_wire()` 和 JSON 保存后重新翻译。实时生成使用 `MyLLM.generate(stream=True)`。

## 支持的协议

以下标识均可用于 `ProviderProfile.protocol`、`ModelProfile.protocol` 和 `stdio_protocol`：

| 协议标识 | 对应 API |
| --- | --- |
| `openai_chat` | OpenAI Chat Completions 及兼容服务 |
| `openai_responses` | OpenAI Responses 及兼容服务 |
| `anthropic` | Anthropic Messages |

同一实例的消息与工具使用同一种输入输出协议。

### 验证范围

常规测试使用官方 SDK 与本地模拟 HTTP/SSE，不请求真实模型。
真实服务 E2E 单独覆盖三种上游与四种输入/输出格式的文本、工具回合、能力处理和推理回传。
汇总调用也通过上游 SSE 执行，结果不能证明供应商的 HTTP 非流式接口行为。

运行方法见[测试文档](docs/testing/README.md)，按日期记录的通过项、失败项和验证限制见
[测试结果目录](docs/testing/results/README.md)。历史结果不作为当前模型或服务行为的保证。

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

文档分为四部分，完整导航见[项目文档](docs/README.md)。

- [使用文档](docs/README.md#使用文档)：公开接口、配置、中间模型、语义损失和服务接入。
- [开发文档](docs/README.md#开发文档)：设计取舍、职责边界、协议扩展和构建。
- [测试文档](docs/testing/README.md)：测试复现、结果汇总、性能与发行包验证。
- [审查文档](docs/reviews/README.md)：代码审查、根因、修复记录和不修复决定。

项目采用 [Apache License 2.0](LICENSE)。
