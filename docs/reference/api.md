# API 使用指南

my-llm 提供异步 `generate()` 和可复用的 `MyLLM`，统一调用 OpenAI
Responses / Chat Completions、Anthropic、Gemini 与 Ollama 原生 API。
需要 Python 3.12+。导入名为 `my_llm`。
配置字段的默认值、继承规则和兼容服务适配见[配置指南](configuration.md)。

## 一次调用

安装构建好的包后，设置 `OPENAI_API_KEY` 和供应商提供的 `LLM_MODEL`：

```python
import asyncio
import os

from my_llm import ProviderProfile, generate


async def main():
    message = await generate(
        provider=ProviderProfile(api_type="responses"),
        model=os.environ["LLM_MODEL"],
        messages=[{"role": "user", "content": "你好"}],
    )
    print(message["content"])


if __name__ == "__main__":
    asyncio.run(main())
```

`generate()` 的所有参数都是关键字参数。`provider`、`model`、`messages` 必填。
`model` 可以是模型 ID 字符串，也可以是 `ModelProfile`。字符串相当于
`ModelProfile(model=该字符串)`，不会查询 Provider 的模型列表。
API Key 未在配置中指定时，由对应官方 SDK 读取环境配置。

## 复用 Provider 和 Model

```python
from my_llm import ModelProfile, MyLLM, ProviderProfile


def make_llm(model_id: str):
    return MyLLM(
        provider=ProviderProfile(api_type="chat_completions"),
        model=ModelProfile(model=model_id, max_tokens=2048),
    )


async def ask(llm: MyLLM, question: str):
    return await llm.generate(messages=[{"role": "user", "content": question}])
```

`MyLLM(provider=..., model=..., client=None)` 的 `model` 必须是 `ModelProfile`。
构造时保存配置；需要更改配置时，用更新后的 Profile 创建新实例。
同一实例可并发调用，每次调用分别传入自己的历史消息。
Provider 模型列表用于配置管理，选择方式见[模型列表](configuration.md#模型列表)。

## generate 参数

以下参数适用于模块级 `generate()` 和 `MyLLM.generate()`；实例方法使用构造时的
Provider、Model 和 SDK Client，不再接受 `provider`、`model`、`client`。

| 参数 | 默认值 | 使用方式 |
| --- | --- | --- |
| `messages` | 必填 | 非空字典序列，支持列表、元组；格式见下文 |
| `protocol` | `"openai"` | 默认值跟随 Provider；其他值须与 `ProviderProfile.protocol` 一致 |
| `stream` | `None` | `True` 返回 `MessageStream`；`False`/`None` 返回完整 `Message` |
| `tools` | `omit` | Responses 或 Chat 的本地函数描述列表；`None`/`omit` 不传工具，`[]` 传空列表 |
| `max_tokens` | `None` | 正整数输出限制；`None` 继承 Model，`omit` 不设置限制 |
| `reasoning` | `None` | 可选布尔开关；`True` 且强度为 `none` 时使用 `medium` |
| `reasoning_effort` | `"none"` | `none`/`low`/`medium`/`high`/`xhigh`/`max`；受模型能力与映射约束 |
| `previous_response_id` | `None` | Responses 的服务端历史引用；不能用于 Chat |
| `extra_headers` | `None` | 按键覆盖 Profile 中的请求头；支持 `omit` |
| `extra_body` | `None` | 按键覆盖 Profile 中的额外请求字段；支持 `omit` |
| `base_url` | `None` | 覆盖服务地址；`None` 继承，`omit` 使用 SDK 默认配置 |
| `api_key` | `None` | 覆盖凭据；`None` 继承，`omit` 使用 SDK 环境配置 |
| `timeout` | `None` | 正数，单位秒；`None` 继承，`omit` 使用 SDK 默认值 |
| `max_retries` | `None` | 非负整数重试次数；`None` 继承，`omit` 使用 SDK 默认值 |

模块级函数还接受 `client=None`，用于注入调用者管理的官方 SDK Client：
OpenAI 为 `AsyncOpenAI`，Anthropic 为 `AsyncAnthropic`，Gemini 为 `google.genai.Client`
（内部使用 `.aio`），Ollama 为 `ollama.AsyncClient`。
注入 Client 后，四个连接覆盖参数必须保持 `None`，不能传 `omit` 或其他值。
`reasoning=False` 与非 `none` 强度冲突；推理字段、能力及 extras 的规则见
[配置指南](configuration.md#推理配置)。库不自动执行工具或管理应用的对话历史。

## 读取完整结果

返回的 `Message` 是普通字典，使用统一的 Chat 风格：

| 字段 | 含义 |
| --- | --- |
| `role` | 固定为 `"assistant"` |
| `content` | 回答文本；没有文本时为 `None` |
| `reasoning_content` | 可见推理文本，有内容时提供 |
| `tool_calls` | 完整工具调用列表，有调用时提供 |
| `finish_reason` | Chat 和原生协议保留完成原因；Responses 为 `stop` 或 `tool_calls` |
| `usage` | 启用用量后提供供应商原生用量字典，不自行计算 Token |
| `response_id` | 供应商返回的响应 ID（若提供） |
| `response_status` | Responses 的完成状态 |
| `response_output` | 原生历史元数据，包含 Responses items、Anthropic 签名块或 Gemini parts |

除 `role`、`content` 外，按需提供字段。工具调用形如
`{"id": "call_1", "type": "function", "function": {"name": "add", "arguments": "{\"a\":2,\"b\":3}"}}`，
其中 `arguments` 始终是完整的 JSON 字符串。

## 流式读取与提前停止

流式调用也需要先 `await`。返回流后，第一次迭代才开始请求。

```python
from my_llm import MessageChunk, MyLLM


async def print_stream(llm: MyLLM, question: str):
    stream = await llm.generate(
        messages=[{"role": "user", "content": question}], stream=True,
    )
    combined = MessageChunk()
    async with stream:
        async for chunk in stream:
            print(chunk.content, end="", flush=True)
            combined += chunk
    return combined.to_message(require_finished=True)
```

`MessageChunk.content` 和 `reasoning_content` 是增量文本；最后一个 Chunk 才携带
完整 `tool_calls`，同时提供完成原因和可用的用量、原生输出信息。
`chunk.has_delta` 判断是否有文本、推理或工具内容，`chunk.is_finished` 判断是否有
完成原因。`to_message(require_finished=True)` 在未完成时抛 `ValueError`。
Chunk 的嵌套数据按只读使用；需要修改结果时使用 `to_message()` 返回的独立字典。

使用 `async with stream`，可在 `break`、异常、取消时关闭单次请求。
也可显式 `await stream.aclose()`，重复关闭是安全的。单独使用 `async for` 后
`break` 不保证立即释放连接。完整结果模式会自动消费并关闭流。

## 输入消息格式

`messages` 接受 Chat Completions 消息、Responses input/output items，也可混合。
库根据结构自动识别，按 Provider 的 `api_type` 转换，无需选择输入模板。
公共输入类型 `MessageInput` 表示这些字典；允许角色为
`system`、`user`、`assistant`、`tool`。

```python
from my_llm import MyLLM


async def ask_with_response_input(llm: MyLLM):
    # llm 可以使用任意一种 api_type。
    return await llm.generate(messages=[
        {"role": "system", "content": "请简短回答"},
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "你好"},
        ]},
    ])
```

| 输入内容 | 自动转换行为 |
| --- | --- |
| 文本 `role/content` | 两套 API 均可使用 |
| `text`、`input_text`、`output_text` | 转为目标 API 的文本内容块 |
| `image_url`、`input_image` | 在可表示的情况下转换图片内容块 |
| `file`、`input_file` | 在可表示的情况下转换文件内容块 |
| Chat `tool_calls` / `role="tool"` | 转为 Responses `function_call` / `function_call_output` |
| Responses `function_call` / `function_call_output` | 转为 Chat 工具调用/结果，按 `call_id` 关联 |
| Responses `message` / `reasoning` | 转为 Chat assistant 内容及工具调用历史中的可见推理 |

同协议原生项保留 ID、status、phase 等字段。SDK 的 output 对象需先转为字典，
如 `[item.model_dump(exclude_none=True) for item in response.output]`。
加密推理仅在 Responses 中原样回放，不转为可见文本。

跨协议缺少对应表示时抛 `MessageValidationError`，例如 Chat 不支持
`item_reference`、仅含 `file_id` 的图片、仅含 `file_url` 的文件；Responses
不支持 Chat 音频块。Chat 工具结果只支持文本内容块。
历史还会检查重复调用 ID、缺失调用、未完成参数及结果关联，避免发送无效工具历史。

## 工具调用：描述、执行、回传

先设置 `ModelProfile(supports_function_calling=True)`。`tools` 支持两种 API
官方函数格式，列表中每项分别识别，同协议保持其字段，跨协议自动转换。
只支持调用者执行的函数工具，不提供独立模板配置。

```python
# Responses 扁平格式
responses_tool = {
    "type": "function", "name": "add",
    "parameters": {"type": "object", "properties": {
        "a": {"type": "integer"}, "b": {"type": "integer"},
    }, "required": ["a", "b"]},
}
# Chat Completions 嵌套格式，同样可用于任意一种 API
chat_tool = {
    "type": "function",
    "function": {key: value for key, value in responses_tool.items() if key != "type"},
}
```

下面完成一个工具回合。调用者按已知函数名执行本地工具，并为每个调用回传结果。
模型在后续回复中仍可能请求工具，应用可按相同方式继续循环。

```python
import json

from my_llm import ModelProfile, MyLLM, ProviderProfile


async def add_round_trip(provider: ProviderProfile, model_id: str, *, client=None):
    llm = MyLLM(
        provider=provider,
        model=ModelProfile(model=model_id, supports_function_calling=True),
        client=client,
    )
    tools = [{
        "type": "function", "name": "add",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "integer"},
        }, "required": ["a", "b"]},
    }]
    history = [{"role": "user", "content": "2 加 3 是多少？请调用 add。"}]
    first = await llm.generate(messages=history, tools=tools)
    if not first.get("tool_calls"):
        return first
    history.append(first)
    local_tools = {"add": lambda a, b: a + b}
    for call in first["tool_calls"]:
        function = call["function"]
        output = local_tools[function["name"]](**json.loads(function["arguments"]))
        history.append({
            "role": "tool", "tool_call_id": call["id"],
            "content": json.dumps(output, ensure_ascii=False),
        })
    return await llm.generate(messages=history, tools=tools)
```

工具返回统一的 `ToolCall` 字典，`FunctionCall` 描述其中 `name/arguments`。
用 `tool_call_id=call["id"]` 关联结果；Responses 原生函数结果使用 `call_id`，
不能使用 item 自身的 `id` 替代它。

## Responses 历史续接

有两种方式，由应用选择：

1. **本地保存历史**：将返回的完整 `Message` 原样追加到原历史，再追加工具结果或
   下一条用户消息。上面的工具示例适用于两套 API。保留 `response_output` 可完整
   回放 Responses 的 message、reasoning 和 function_call 项。
2. **使用服务端历史**：下一次调用传 `previous_response_id`，只提供新消息或工具
   结果。服务端必须支持并保留对应历史。此时不要重复发送整段旧历史。

```python
from my_llm import MyLLM


async def continue_response(llm: MyLLM):
    # llm 的 Provider 必须使用 api_type="responses"。
    first = await llm.generate(messages=[{"role": "user", "content": "介绍一下 Python"}])
    return await llm.generate(
        previous_response_id=first["response_id"],
        messages=[{"role": "user", "content": "再给一个简单例子"}],
    )
```

第一种方式中不要只保留 `content` 而丢失工具调用或原生输出。
若编辑返回消息中的统一工具调用，需同步原生 `response_output`；两者冲突会报错。
第二种方式中先前工具调用 ID 的存在性由服务端验证，库仍检查结果 ID 非空且不重复。

## 错误处理

```python
from my_llm import MyLLM, StreamError


async def ask_with_partial_output(llm: MyLLM, question: str):
    try:
        return await llm.generate(messages=[{"role": "user", "content": question}])
    except StreamError as exc:
        print("已收到的部分消息：", exc.partial_message)
        print("供应商错误详情：", exc.details)
        raise
```

| 异常 | 含义 |
| --- | --- |
| `ConfigurationError` | Profile、参数类型、组合或配置值无效；也继承 `ValueError` |
| `UnsupportedProviderError` | 协议未实现，或供应商不支持必需的上游流式请求 |
| `UnsupportedCapabilityError` | 请求的推理强度、工具等能力未启用或不受支持 |
| `MessageValidationError` | 消息、工具或跨协议转换无效 |
| `StreamError` | 响应失败，提供 `details` 和 `partial_message` |
| `IncompleteStreamError` | 流缺少完成状态，或工具调用不完整 |
| `ResponseFailedError` | 响应出现失败/错误事件 |
| `ResponseIncompleteError` | Responses 明确报告 incomplete |

所有上述异常都继承 `MyLLMError`。前三种配置细分类及消息校验异常继承
`ConfigurationError`；后三种流异常继承 `StreamError`，
`ResponseIncompleteError` 也继承 `IncompleteStreamError`。
流式模式下，网络或响应错误可能在迭代过程中才发生。
四种官方 SDK 的 HTTP、认证、超时等异常保留原类型，按 SDK 异常处理。

## 共享 SDK Client

```python
from openai import AsyncOpenAI

from my_llm import ModelProfile, MyLLM, ProviderProfile


async def with_shared_client(model_id: str):
    async with AsyncOpenAI() as client:
        llm = MyLLM(
            provider=ProviderProfile(api_type="responses"),
            model=ModelProfile(model=model_id),
            client=client,
        )
        return await llm.generate(messages=[{"role": "user", "content": "你好"}])
```

注入 Client 时，在该 Client 上配置 URL、密钥、超时与重试。库关闭每次请求的流，
Client 本身由调用者关闭。不注入时，库在每次请求完成、失败、取消或显式关闭后
释放自行创建的 Client。

## 接口迁移

旧接口 `create()` 更名为 `generate()`，`Invoke` 更名为 `MyLLM`，实例方法
相应更名为 `MyLLM.generate()`。旧公开名称不再导出。
离线可运行的完整演示见 [local_demo.py](../../examples/local_demo.py)；
开发构建步骤见[构建指南](../guides/building.md)。


## 原生协议调用

```python
from my_llm import ModelProfile, ProviderProfile, generate


async def ask_anthropic(model_id: str):
    return await generate(
        provider=ProviderProfile(protocol="anthropic", usage_mode="native"),
        model=ModelProfile(model=model_id, max_tokens=4096),
        messages=[{"role": "user", "content": "你好"}],
    )
```

Gemini 和 Ollama 使用相同的调用参数，只需更改 Provider 的 `protocol` 与模型 ID。
`api_type` 只选择 OpenAI 的两种 API；原生协议忽略它在请求中的作用。
非流式结果仍由对应 SDK 的异步流拼接，未实现额外 HTTP、SSE 或重试层。
`usage_mode="none"` 不返回用量；原生协议的另外两种模式均读取 SDK 原生用量。
Anthropic 的 `finish_reason` 例如 `end_turn` / `tool_use`，Gemini 例如 `STOP` / `MAX_TOKENS`，
Ollama 例如 `stop` / `length`。截断流与无效工具 JSON 会抛出 `StreamError` 子类。

工具输入仍是本地 Function Tool，参数仍返回完整 JSON 字符串；工具执行由调用者负责。
回传整个 assistant Message，能保留 Anthropic thinking 签名及 Gemini thought signature。
Gemini / Ollama 缺少原生调用 ID 时生成本地唯一 ID，结果仍用 `tool_call_id` 关联。
`previous_response_id` 只适用于 OpenAI Responses，原生协议须传完整本地历史。
