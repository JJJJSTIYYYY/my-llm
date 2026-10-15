# API 与配置

## 公共入口

`await create(provider=ProviderProfile(...), model=ModelProfile(...), messages=[...])`
返回 OpenAI Chat Completions 风格的 `Message` 字典。
`model` 也可传模型 ID 字符串，使用默认模型配置。

`stream=True` 返回 `MessageStream` 异步迭代器；`stream=None` 或 `False` 将同一条流
拼接成 `Message`。无论哪种模式，底层 SDK 调用都使用 `stream=True`。
要复用已解析的配置，可以创建 `Invoke(provider=..., model=..., client=...)`，
再调用 `await invoke.create(...)`。每次请求使用独立的解析状态。

`create` 还接受设计中的 `base_url`、`api_key`、`timeout`、`max_retries`、
`extra_headers`、`extra_body`、`max_tokens`、`reasoning_effort` 与 `tools`。
`reasoning=True` 等价于未指定强度时使用 `medium`；`reasoning=False` 与非 `none`
强度同时传入属于配置冲突。通常直接使用 `reasoning_effort` 即可。

设计的接口草图未指定模型、消息和 Profile 的传入方式，本文明确采用上述关键字参数。
只有异步接口；`protocol` 默认 `openai`，`anthropic`、`gemini`、`ollama` 为预留接入点，
当前会抛出 `UnsupportedProviderError`。

## Profile 配置

不提供预制 Provider 或 Model。`ProviderProfile.api_type` 必须是 `responses`
或 `chat_completions`；模型 ID 由 `ModelProfile.model` 指定。

| 字段 | 可配置层 | 协议默认值 |
| --- | --- | --- |
| `base_url`、`api_key`、`timeout`、`max_retries` | Provider，调用参数优先 | 不发送，使用 SDK 默认值/环境变量 |
| `extra_headers`、`extra_body` | Provider、Model、调用参数 | `{}` |
| `max_tokens` | Model、调用参数 | 不显式限制 |
| `supports_stream` | Provider | `True`，禁用时拒绝调用 |
| `supports_reasoning` | Model | `True` |
| `supports_function_calling` | Model | `False` |
| `supports_message_name` | Model | `True`，仅影响 Chat 消息中的 `name` |
| `supports_reasoning_effort` | Provider、Model | Responses 为 True，Chat 为 False |
| `reasoning_enable_parameter` | Provider、Model | Responses: `reasoning`；Chat: `extra_body.thinking` |
| `reasoning_effort_parameter` | Provider、Model | Responses: `reasoning.effort`；Chat: `reasoning_effort` |
| `reasoning_effort_mapping` | Provider、Model | 六个强度均原样映射 |
| `content_path` | Provider、Model | Responses: `response.output_text.delta`；Chat: `$.choices[0].delta.content` |
| `reasoning_content_path` | Provider、Model | Responses: `response.reasoning_summary_text.delta`；Chat: `$.choices[0].delta.reasoning_content` |
| `replay_reasoning_content_when_tool_calls` | Provider、Model | `True`，只控制 Chat 的 assistant 工具调用历史 |
| `usage_mode` | Provider | `none` |

`supported_roles` 固定为 system/user/assistant/tool，不能在构造 Profile 时配置。

配置按协议 → Provider → Model → 调用参数覆盖。`None` 表示继承，`omit` 表示禁用
整个字段或删除字典中的单个键；字典按键浅覆盖，同一键的嵌套字典整体替换。
`reasoning_effort_mapping={"xhigh": None, "max": None}` 明确禁止相应强度。

例如，使用配置创建支持自定义推理参数的兼容服务：

```python
from my_llm import ProviderProfile, ModelProfile, omit

provider = ProviderProfile(
    api_type="chat_completions",
    base_url="https://your-provider.example/v1",
    supports_reasoning_effort=True,
    reasoning_enable_parameter={"extra_body": {"enable_thinking": True}},
    reasoning_effort_parameter={
        "reasoning_effort": omit,
        "extra_body": {"thinking_budget": "{effort}"},
    },
    reasoning_effort_mapping={"low": "small", "medium": "normal", "high": "large"},
)
model = ModelProfile(model="your-model-id", reasoning_effort_mapping={"xhigh": None, "max": None})
```

设计中的 `reasoning=Reasoning` / `extra_body=dict` 通过上述 SDK 参数字典表达。
`"{effort}"` 是动态强度占位符，Codec 将其替换为映射后的字符串。
Responses 默认请求 `summary="detailed"`，可通过推理参数配置覆盖或省略。
自动生成的推理字段优先于调用者的 `extra_body`；也确保生成的 Token 参数与
Usage 开关不会被同名 body 字段覆盖。`extra_body` 不能绕过底层流式调用、模型、
输入或工具格式的约束。

`max_tokens` 在 Responses 转换为 `max_output_tokens`，在 Chat 转换为
`max_completion_tokens`。强度 `none` 不自动生成开启字段；模型不支持推理时
请求非 `none` 强度会抛异常。模型支持推理但不支持强度时，只生成开启字段。

## 消息与工具

`Message` 是 TypedDict，普通字典即可。角色、文本/内容块、`tool_calls` 和
`tool_call_id` 遵循 Chat Completions 格式；工具参数保持完整 JSON 字符串。
Responses 会转换普通文本、文本内容块和用户图片内容块；不支持的内容块抛出
`MessageValidationError`。Chat 的内容块按其协议透传。

`tools` 接受 OpenAI 官方 Responses 或 Chat Completions 的函数工具格式。
Codec 对列表中的每个工具自动识别：Chat API 收到 Responses 的扁平格式时，
将函数字段放入 `function`；Responses API 收到 Chat 的嵌套格式时，展开函数字段；
与目标 API 一致的格式保持透传，不提供独立的模板配置。字段值及扩展字段保留，
转换不修改调用者传入的字典；混合列表中的每个工具分别处理。

```python
# 两种格式都能用于任意一种 API，Codec 只在需要时转换。
responses_tool = {"type": "function", "name": "get_weather", "parameters": {"type": "object"}}
chat_tool = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
```

工具调用返回仍转换为统一的 Chat 风格字典。传入非空工具列表前必须
配置 `supports_function_calling=True`。库不执行工具：调用者按函数名找到本地函数，
执行后将结果作为 `role="tool"` 的消息回传，`tool_call_id` 必须匹配原调用 ID。

工具参数增量只在 Codec 内累积。最后一个 Chunk 才包含完整 `tool_calls` 列表，
文本与推理增量仍实时返回。无完成状态、截断 JSON、缺少 ID 或错误结束的工具调用
不会作为有效结果返回。

`MessageChunk` 采用设计附带的 `docs/message.py` 实现作为基础。
使用 `combined += chunk` 累加，`combined.to_message(require_finished=True)` 得到字典。
输入 Chunk 和其工具字典不会被累加操作修改。

## Responses 历史回放

最终 Chunk/Message 附带 `response_id`、`response_status` 与原生 `response_output`。
将返回的 Message 原样加入历史：下一次请求优先回放原生 reasoning、function_call
及 message items，不重复构造相同工具调用，不发送 Chat 风格推理文本。
这也保留了 reasoning item 的 ID、summary、encrypted_content 等原生字段。
原生与统一工具调用同时存在时必须一致。

也可显式传入 `previous_response_id`，只提供新的消息或工具结果。这时先前调用 ID
由服务端上下文验证；库仍验证结果 ID 非空且不重复。两种历史管理方式由调用者
选择，避免在使用 Response ID 时再次发送整个旧历史。

## 完成、用量与错误

Chat 保留原生 `finish_reason`；Responses 完成转换为 `stop` 或 `tool_calls`，并附带
`response_status="completed"`。只有读取到必要的完成状态才返回最终结果。

`usage_mode="none"` 不返回 usage；Responses 用 `native` 读取最终 response.usage；
Chat 用 `stream_options` 自动发送 `{"include_usage": True}`，并读取可能没有 choices
的用量事件。Usage 保留原生字典字段，不重新计算 Token 数。
与 API 不兼容的 Usage 模式属于配置错误。

配置和能力问题抛出 `ConfigurationError`、`UnsupportedCapabilityError` 或
`UnsupportedProviderError`。`StreamError` 附带 `details` 与 `partial_message`；
Responses failed/error/incomplete 事件分别抛出 `ResponseFailedError` 或
`ResponseIncompleteError`。流缺少完成状态或工具不完整时抛 `IncompleteStreamError`。
SDK HTTP、认证、超时以及 SDK 已识别的流错误保留其 SDK 异常类型。

## 生命周期

库自行创建的 SDK 客户端与流会在完成、失败、取消或显式关闭后释放。
通过 `client=AsyncOpenAI(...)` 注入的 SDK 客户端由调用者管理，库只关闭单次流；
此时配置中的客户端初始化字段不再参与构造，也不能在 create 中传入初始化覆盖。

部分消费时使用 `async with stream:` 或主动 `await stream.aclose()`，确保提前退出
后立即释放连接。异步迭代本身不会保证 bare `break` 时立即关闭生成器。

协议实现核对依据：
[OpenAI Function Calling](https://developers.openai.com/api/docs/guides/function-calling)、
[Responses 流式事件](https://developers.openai.com/api/reference/resources/responses/streaming-events)、
[Responses reasoning item 回放](https://developers.openai.com/cookbook/examples/responses_api/reasoning_items)。
