# API 使用指南

`MyLLM` 构造时接收 `provider: ProviderProfile`、`model: ModelProfile` 和可选的
`stdio_protocol: Protocol`。实例保留 `provider` / `model` 的副本、`resolved_profile`、
无状态的 `codec: MyCodec` 和输入输出协议 `stdio_protocol`。
`generate(provider=..., model=..., messages=..., stdio_protocol=..., **parameters)`
是创建临时实例的一次性入口，其中 model 也可传模型 ID 字符串。
Protocol 是 str 别名；内置协议之外，也可使用启动时通过 MyCodec.register(codec) 注册的协议名。

## 协议与返回值

| stdio_protocol | messages 标准格式 | 非流式返回 | 流式事件 |
| --- | --- | --- | --- |
| `openai_chat` | Chat 消息列表 | assistant 消息字典 | ChatCompletionChunk 字典 |
| `openai_responses` | Responses input/output items | output items 列表 | Responses 事件字典 |
| `anthropic` | role/content Messages 列表 | assistant role/content 字典 | Messages 事件字典 |
| `gemini` | role/parts Content 列表 | model role/parts 字典 | GenerateContentResponse 字典 |
| `ollama` | 原生 role/content 消息列表 | assistant 消息字典 | ChatResponse 字典 |

返回的是消息或 items，可直接追加到相同协议的历史；不额外包一层统一结果对象。
结果只包含目标协议已识别的字段，不附加未知字段或跨协议恢复记录。
Responses 使用 `history.extend(result)`，其他协议使用 `history.append(result)`。
完成原因和用量出现在各协议标准流事件的位置，不增加自定义消息字段。
所有输入可使用普通字典或官方 SDK 模型；只按已支持的字段读取模型属性。

`stdio_protocol` 与 `ProviderProfile.protocol` 独立。前者未指定时，
`guess_stdio_protocol(messages_or_tools)` 从整个列表识别，仅在唯一匹配一种协议时绑定；
空列表、零匹配或多匹配均报 `MessageValidationError`，错误不会绑定协议。
纯文本 Chat / Responses / Anthropic / Ollama 输入有重叠，应显式指定 `stdio_protocol`。
显式指定会跳过推测，输入仍按所选协议验证；绑定后不会重新猜测。
messages 和 tools 都遵循这个协议，历史不混用不同协议格式。

```python
from my_llm import ModelProfile, MyLLM, Parameter, ProviderProfile


async def ask(llm, prompt):
    return await llm.generate(messages=[{"role": "user", "content": prompt}])


async def session():
    async with MyLLM(
        provider=ProviderProfile(protocol="openai_responses"),
        model=ModelProfile(
            model="your-model",
            max_tokens=Parameter(
                name="输出上限",
                path="$.max_output_tokens",
                default_value=2048,
                phase="decode",
            ),
        ),
        stdio_protocol="openai_chat",
    ) as llm:
        return await ask(llm, "Hello")
```

## 生成选项

`MyLLM.generate()` 只接受以下 keyword 参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `messages` | 必填 | 非空、同协议的消息或 items 列表 |
| `tools` | `omit` | 同协议的工具声明列表；None/omit 不发送，[] 发送空列表 |
| `stream` | False | True 返回需异步消费的 MessageStream |
| `max_tokens` | 未设置 | 未传时继承 Profile 默认值；显式 None 保留，omit 删除目标参数；按 `max_tokens` 的 decode 参数路径写入 SDK 请求 |
| `reasoning` | None | True 在 effort=none 时使用 medium；False 与非 none effort 冲突 |
| `reasoning_effort` | `none` | none/low/medium/high/xhigh/max，经 profile 映射 |
| `extra_headers` | None | 按键覆盖 profile headers |
| `extra_body` | None | 按键覆盖 profile body，不能覆盖协议核心字段 |
| `previous_response_id` | None | 仅上游 openai_responses 支持的服务端历史引用 |

provider、model、protocol、连接参数和 SDK client 不作为实例 generate 的动态参数。
改变连接或模型时构造新实例。配置默认值和协议参数映射见 [配置指南](configuration.md)。

## 流与资源

调用 `await llm.generate(..., stream=True)` 后使用 `async with stream`。
未消费的流不启动请求或创建客户端；提前退出、取消、SDK 失败和正常结束都会关闭本次响应流。
同一 MyLLM 实例的串行、并发、流式与非流式调用复用官方 SDK 客户端，累积状态仍按请求独立。
一路取消或失败不会关闭共享客户端或其他请求。

应用结束时使用 `await llm.aclose()`，或者将实例放入 `async with MyLLM(...) as llm`；
先完成或关闭所有响应流，再关闭实例。关闭操作可重复调用，关闭后新请求及尚未开始消费的流
均报 RuntimeError，不重新创建客户端。实例应在同一异步事件循环内使用。
独立函数 `generate(...)` 的临时实例自动清理；返回流时，其所有权延续至 MessageStream 关闭。
高级集成可通过 `stream.add_close_callback(async_callback)` 登记异步清理；回调在流关闭时执行一次，包含尚未开始消费的情况。

Ollama 的公开 chat 接口不接受逐请求 headers，因此它按合并后的请求头分别缓存 SDK 客户端；
相同请求头复用，不修改共享客户端的 headers。其余协议将请求头作为本次 SDK 调用的参数。
不检测直接修改 MyLLM 内部属性后的配置失效。详见[客户端生命周期](../design/client-lifecycle.md)
和[并发复核](../audits/concurrency.md)。

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
工具调用之前的文本和可见思考增量即时输出；工具参数保存在请求状态中，
整条响应成功结束后再生成目标协议的工具事件。为保留内容顺序，工具调用之后出现的块
一并暂存到完成时发布；失败或中断不会发布这些待完成调用。
Anthropic/Gemini 的签名内容保留在完成消息和对应的原生流字段中。
启用用量时，Chat 先发送结束 choice，再发送 `choices=[]` 的独立 usage 尾块。
跨协议输出到 Responses 时，输出限制与内容过滤分别生成相应的 `response.incomplete`，
成功结束才生成 `response.completed`；协议字段依据与校验范围见[协议复核](../audits/protocol-conformance.md)。
块引用、快照协调及发布规则见[流状态设计](../design/stream-state.md)。

## 工具回合

调用者执行工具；本库只转换声明、调用和结果，并校验消息间的 ID 对应关系。
重复调用 ID、重复结果、未知结果 ID 和缺少结果均报错。
Gemini/Ollama 中没有 ID 的调用按历史顺序建立局部 ID；无 ID 的结果按名称关联待完成调用。

```python
import json


async def add_round_trip(provider, model_id):
    from my_llm import Config, ModelProfile, MyLLM

    async with MyLLM(
        provider=provider,
        model=ModelProfile(
            model=model_id,
            supports_function_calling=Config(name="工具调用", default_value=True),
        ),
        stdio_protocol="openai_chat",
    ) as llm:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]
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

`BaseCodec` 声明 ProtocolDefaults 并组合协议对应的 `BaseEncoder` / `BaseDecoder` / `BaseTransport`；它们不保存 profile、
流片段或识别结果。每种协议独立一对 encoder/decoder。

- `encode_messages` / `decode_messages`
- `encode_tools` / `decode_tools`
- `encode_chunks` / `decode_chunks`

以上方法签名均为 `(values: list, profile: ResolvedProfile | None = None, **parameters)`。
encode 从标准协议到内部开放字典；decode 从内部字典到标准协议。
`profile=None` 表示标准协议，有 profile 时应用厂商配置。
BaseCodec 的 `match_protocol(values)` 返回协议名或 None。
MyCodec 的 `match_protocol(values)` 只有在整批输入唯一匹配时返回协议名；零匹配或多匹配返回 None。

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

MyCodec 持有创建时的注册快照，默认包含五个内置 Codec，可复用于不同转换路径。
`MyCodec.register(codec: BaseCodec) -> None` 是静态注册入口，须早于 MyCodec / MyLLM 实例创建；
既有实例保留原快照，重名注册报 ConfigurationError。实现与注册示例见[接入指南](../guides/adding-protocol.md)。
中间态是内部实现，不是应用消息协议。
Encoder 和 Decoder 只翻译明确识别的协议字段。未知字段不进入中间态，也不出现在翻译输出中；
非标准差异通过 Profile 明确支持的字段路径、参数绑定或思考回放配置处理。
工具参数、结果对象和 JSON Schema 等已识别开放字典完整保留内部应用键值。

原生 Anthropic 签名、redacted thinking、Responses encrypted_content 等标准字段由协议实现
显式处理；跨协议转换不附加隐藏快照或恢复记录，也不保证转换回原协议后恢复原始对象。
流式多轮对话应保存 stream.result，完整文本不能替代工具调用及原生内容块。

实际请求由 Transport 再过滤并校验目标 SDK 支持的结构，不修改业务历史。
详见[翻译字段边界](../design/translation-boundaries.md)与[兼容边界](configuration.md#跨协议请求的处理边界)。

## 异常边界

`ConfigurationError` 表示配置错误，`MessageValidationError` 表示标准输入/消息关联错误；
`UnsupportedCapabilityError` 表示模型禁用了所需能力，或目标协议 / SDK 无法安全表达必要内容或约束。SDK HTTP/传输错误保持原有类型。
`StreamError` 及其子类包含原生 `details` 和相同 stdio 协议的 `partial_message`。
缺少消息级终止事件报 `IncompleteStreamError`；Responses failed/incomplete 使用对应子类。
本库不提供消息内部中断恢复、工具执行或部分 JSON 工具调用接口。

`MyCodec.convert(chunks=完整事件列表, input_format=..., target_format=...)` 也可离线转换
整个流；跨协议的完整 batch 在本次调用内组装，不保存到转换器。
实时消费使用 MyLLM，由每次请求持有累积状态；BaseEncoder/BaseDecoder 仍只转换单次传入的块。

Encoder / Decoder 负责已识别字段的翻译；Decoder 还翻译生成选项并组织原生请求结构。
BaseTransport 负责输入与工具关联验证、模型能力、生成参数和原生结构验证、字段过滤，
同时管理官方 SDK 创建、关闭、复用键及请求执行。MyLLM 只编排每次请求并缓存客户端。

自定义实现需提供以下接口；公开 messages / tools / chunks 入口由基类固定实现：

| 实现 | 必须提供 |
| --- | --- |
| BaseCodec | protocol_or_api_style、defaults: ProtocolDefaults、encoder / decoder / transport。 |
| BaseEncoder | protocol、_messages / _tools / _chunks、match_item、usage。 |
| BaseDecoder | protocol、_messages / _tools / _chunks、usage、assemble_request；result 可覆盖收集结果形状。 |
| BaseTransport | request_schema、message_type / tool_type、message_path / tool_path、validate_request、create_client、upstream。 |

BaseCodec.request 接收中间态，由 Transport.validate_parameters 一次校验、解析生成控制参数，
再把返回的 PreparedOptions 显式传给 Decoder.request 翻译消息、工具和请求选项，
最后进入 Transport.request，验证工具关联、模型能力和原生结构，并过滤不支持字段。
BaseDecoder.request 的 final 流程统一调用 request_parameters(profile, options) 应用 decode 参数，
再将生成的参数字典传给 assemble_request(messages, tools, profile, *, request: dict, **parameters)。
自定义 assemble_request 只向已有 request 写入消息和工具，不接收 options，也不重复应用参数。
协议专属控制可通过 _request_parameters(request, profile, options) 钩子追加，布局由 body_container、
headers_field、body_overlay、omit_empty_body 类属性声明。参数流程由基类保证执行一次，不重新解析原始生成参数。
BaseEncoder.parameter_value() 固定读取 encode 参数；协议钩子只负责事件判别与中间态语义映射。
原始参数继续传给消息翻译，以保留显式思考开关和兼容服务的回放语义。
PreparedOptions 和 ProtocolDefaults 均从 my_llm 公开导出。
Decoder._messages / _tools 与公开 messages / tools 均返回普通 list。
ProtocolDefaults 使用 `Parameter` 声明路径、默认值和 encode / decode 阶段，使用 `Config` 声明行为开关；
不用的参数通过 omit 禁用。`name` / `comment` 只用于展示，功能由 Profile 字段标识决定。
字段 None 继承，wrapper 的 default_value=None 保留为实际值；声明及嵌套内容按只读借用。
Profile 默认值来自已注册 Codec.defaults 的副本，解析负责分层合并、通用字段校验和快照，
不创建 Codec 或 SDK 客户端；协议特有限制与 SDK 约束由 Transport 补充验证。
直接使用 resolve_profiles / MyCodec 的集成应调用 codec.validate_profile(resolved)；MyLLM 已在构造时完成此步骤。
完整实现约定见[接入协议指南](../guides/adding-protocol.md)。

Transport.create_client 创建官方 SDK 客户端，由 MyLLM 保存并复用；close_client 默认调用异步 close。
client_key 默认 None，SDK 将请求选项绑定到客户端时可覆盖。
Transport.upstream 借用客户端，只关闭本次响应流与响应体。
validate_profile 与 allows_external_calls 可按协议覆盖；Decoder 不再提供这些方法。
