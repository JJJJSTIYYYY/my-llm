# API 使用指南

`MyLLM` 构造时接收 `provider: ProviderProfile`、`model: ModelProfile` 和可选的
`stdio_protocol: Protocol`。实例保留 `provider` / `model` 的副本、`resolved_profile`、
无状态的 `codec: MyCodec` 和输入输出协议 `stdio_protocol`。
`generate(provider=..., model=..., messages=..., stdio_protocol=..., **parameters)`
是创建临时实例的一次性入口，其中 model 也可传模型 ID 字符串。

## 协议与返回值

| stdio_protocol | messages 标准格式 | 非流式返回 | 流式事件 |
| --- | --- | --- | --- |
| `openai_chat` | Chat 消息列表 | assistant 消息字典 | ChatCompletionChunk 字典 |
| `openai_responses` | Responses input/output items | output items 列表 | Responses 事件字典 |
| `anthropic` | role/content Messages 列表 | assistant role/content 字典 | Messages 事件字典 |
| `gemini` | role/parts Content 列表 | model role/parts 字典 | GenerateContentResponse 字典 |
| `ollama` | 原生 role/content 消息列表 | assistant 消息字典 | ChatResponse 字典 |

返回的是消息或 items，可直接追加到相同协议的历史；不额外包一层统一结果对象。
Responses 使用 `history.extend(result)`，其他协议使用 `history.append(result)`。
完成原因和用量出现在各协议标准流事件的位置，不增加自定义消息字段。
所有输入可使用普通字典或可转为字典的官方 SDK 模型。

`stdio_protocol` 与 `ProviderProfile.protocol` 独立。前者未指定时，
`guess_stdio_protocol(messages_or_tools)` 从整个列表识别并绑定一次；空列表或无法识别的
输入报 `MessageValidationError`。纯文本 Chat / Responses / Anthropic / Ollama 输入有重叠，
默认按 `openai_chat`；显式指定可消除歧义。绑定后不会重新猜测。
messages 和 tools 都遵循这个协议，历史不混用不同协议格式。

```python
from my_llm import ModelProfile, MyLLM, ProviderProfile

llm = MyLLM(
    provider=ProviderProfile(protocol="openai_responses"),
    model=ModelProfile(model="your-model", max_tokens=2048),
    stdio_protocol="openai_chat",
)

async def ask(llm, prompt):
    return await llm.generate(messages=[{"role": "user", "content": prompt}])
```

## 生成选项

`MyLLM.generate()` 只接受以下 keyword 参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `messages` | 必填 | 非空、同协议的消息或 items 列表 |
| `tools` | `omit` | 同协议的工具声明列表；None/omit 不发送，[] 发送空列表 |
| `stream` | False | True 返回需异步消费的 MessageStream |
| `max_tokens` | None | 继承模型；omit 取消输出 token 限制 |
| `reasoning` | None | True 在 effort=none 时使用 medium；False 与非 none effort 冲突 |
| `reasoning_effort` | `none` | none/low/medium/high/xhigh/max，经 profile 映射 |
| `extra_headers` | None | 按键覆盖 profile headers |
| `extra_body` | None | 按键覆盖 profile body，不能覆盖协议核心字段 |
| `previous_response_id` | None | 仅上游 openai_responses 支持的服务端历史引用 |

provider、model、protocol、连接参数和 SDK client 不作为实例 generate 的动态参数。
改变连接或模型时构造新实例。配置默认值和协议参数映射见 [配置指南](configuration.md)。

## 流与资源

调用 `await llm.generate(..., stream=True)` 后使用 `async with stream`。
未消费的流不启动请求；提前退出、取消、SDK 失败和正常结束都会关闭上游流和请求 client。
`aclose()` 可重复调用。并发请求有独立的累积状态和 client；
同一实例的并发请求使用同一个 stdio 协议，一路取消或失败不会关闭其他请求，
验证范围见[并发复核](concurrency.md)。

```python
async def print_stream(llm, prompt):
    # llm 的 stdio_protocol 是 openai_chat。
    stream = await llm.generate(messages=[{"role": "user", "content": prompt}], stream=True)
    async with stream:
        async for event in stream:
            for choice in event.get("choices", []):
                delta = choice["delta"]
                print(delta.get("content", ""), end="")
                if delta.get("tool_calls"):
                    print(delta["tool_calls"])  # 参数已经完整
    return stream.result
```

`stream.result` 在完整消费后提供与非流式模式相同的标准消息/items，读取时返回独立副本。
文本和可见思考增量即时输出；工具参数保存在请求状态中，完整后再生成目标协议的工具事件。
Anthropic/Gemini 的签名内容保留在完成消息和对应的原生流字段中。
启用用量时，Chat 先发送结束 choice，再发送 `choices=[]` 的独立 usage 尾块。
跨协议输出到 Responses 时，输出限制与内容过滤分别生成相应的 `response.incomplete`，
成功结束才生成 `response.completed`；协议字段依据与校验范围见[协议复核](protocol-conformance.md)。

## 工具回合

调用者执行工具；本库只转换声明、调用和结果，并校验消息间的 ID 对应关系。
重复调用 ID、重复结果、未知结果 ID 和缺少结果均报错。
Gemini/Ollama 中没有 ID 的调用按历史顺序建立局部 ID；无 ID 的结果按名称关联待完成调用。

```python
import json

async def add_round_trip(provider, model_id):
    from my_llm import ModelProfile, MyLLM
    llm = MyLLM(
        provider=provider,
        model=ModelProfile(model=model_id, supports_function_calling=True),
        stdio_protocol="openai_chat",
    )
    tools = [{"type": "function", "function": {
        "name": "add", "parameters": {
            "type": "object", "properties": {
                "a": {"type": "integer"}, "b": {"type": "integer"},
            }, "required": ["a", "b"],
        },
    }}]
    history = [{"role": "user", "content": "What is 2 + 3?"}]
    response = await llm.generate(messages=history, tools=tools)
    history.append(response)
    for call in response.get("tool_calls", []):
        if call["function"]["name"] != "add":
            raise ValueError("Unknown local tool")
        arguments = json.loads(call["function"]["arguments"])
        output = arguments["a"] + arguments["b"]
        history.append({"role": "tool", "tool_call_id": call["id"], "content": str(output)})
    return await llm.generate(messages=history, tools=tools)
```

`previous_response_id` 允许发送仅包含新消息/结果的历史，并允许结果引用服务端已保存的调用 ID。
仅上游 API 为 `openai_responses` 时可用，不作为 input item 发送。
原生 Responses 的 reasoning / encrypted_content / item_reference 等字段同协议回放保留。

## 独立 Codec

`BaseCodec` 只声明协议和对应的 `BaseEncoder` / `BaseDecoder`；它们不保存 profile、
流片段或识别结果。每种协议独立一对 encoder/decoder。

- `encode_messages` / `decode_messages`
- `encode_tools` / `decode_tools`
- `encode_chunks` / `decode_chunks`

以上方法签名均为 `(values: list, profile: ResolvedProfile | None = None, **parameters)`。
encode 从标准协议到内部开放字典；decode 从内部字典到标准协议。
`profile=None` 表示标准协议，有 profile 时应用厂商配置。
BaseCodec 的 `match_protocol(values)` 返回协议名或 None。

```python
from my_llm import MyCodec

codec = MyCodec()
intermediate = codec.encode(
    messages=[{"role": "user", "content": "Hello"}],
    input_format="openai_chat",
)
native = codec.decode(**intermediate, target_format="gemini")
assert native["messages"] == [{"role": "user", "parts": [{"text": "Hello"}]}]

# 同样可用 codec.convert(..., input_format=..., target_format=...) 一步转换。
```

MyCodec 同时持有五个 Codec，可在不同调用中复用。中间态是内部实现，不是应用消息协议；
源编码将未知字段保留在中间态的原层级，字段冲突时转换值优先；标准格式（`profile=None`）下，同协议与未经修改的中间态之间无损。跨协议输出可能丢弃目标协议不支持的字段，不保证跨协议往返无损。详见[保留规则与验证](lossless-audit.md)。目标 SDK 对不支持的扩展仍可能报错。
增量 encode/decode 不累积片段；需要完整消息或完整工具时由 MyLLM 的请求状态负责累积。

## 异常边界

`ConfigurationError` 表示配置错误，`MessageValidationError` 表示标准输入/消息关联错误；
`UnsupportedCapabilityError` 表示模型禁用了所需能力。SDK HTTP/传输错误保持原有类型。
`StreamError` 及其子类包含原生 `details` 和相同 stdio 协议的 `partial_message`。
缺少消息级终止事件报 `IncompleteStreamError`；Responses failed/incomplete 使用对应子类。
本库不提供消息内部中断恢复、工具执行或部分 JSON 工具调用接口。

`MyCodec.convert(chunks=完整事件列表, input_format=..., target_format=...)` 也可离线转换
整个流；跨协议的完整 batch 在本次调用内组装，不保存到转换器。
实时消费使用 MyLLM，由每次请求持有累积状态；BaseEncoder/BaseDecoder 仍只转换单次传入的块。
