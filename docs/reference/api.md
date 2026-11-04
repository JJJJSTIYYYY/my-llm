# API 参考

本文介绍 `my_llm` 的公开接口、参数、返回值和使用约束。
除另行注明外，下列类型、函数和异常均可从 `my_llm` 导入。
完整配置示例见[配置指南](configuration.md)，协议字段见[标准协议参考](schema/README.md)。

## 生成入口

### `MyLLM`

```python
from my_llm import Attribute, Config, ModelProfile, MyLLM, InputParameter, ProviderProfile


async def ask(llm, prompt):
    return await llm.generate(messages=[{"role": "user", "content": prompt}])


async def session():
    async with MyLLM(
        provider=ProviderProfile(protocol=Config(default_value="openai_responses")),
        model=ModelProfile(
            model=Attribute(default_value="your-model"),
            max_tokens=InputParameter(path="$.max_output_tokens", default_value=2048),
        ),
        stdio_protocol="openai_chat",
    ) as llm:
        return await ask(llm, "Hello")
```

认证通过 Provider 配置或相应 SDK 支持的环境变量提供，见[配置指南](configuration.md#provider-与-model-字段)。

`MyLLM(*, provider, model, stdio_protocol=None, enable_message_cache=False, ...)` 创建可重复调用的实例；所有构造参数仅限关键字：

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `provider` | `ProviderProfile` | 必填，连接配置和默认上游协议 |
| `model` | `ModelProfile` | 必填，模型 ID 和模型配置 |
| `resolve_profile_with_hash` | `tuple[ResolvedProfile, int] \| None` | 默认 `None`；传入同一配置的预解析结果，跳过解析和哈希，复制快照后仍检查协议约束 |
| `stdio_protocol` | `Protocol \| None` | 输入消息、工具和输出共用的协议；默认尝试唯一推断 |
| `enable_message_cache` | `bool` | 默认 `False`；开启后缓存已翻译的增量输入，同一实例必须串行使用 |

`ModelProfile.protocol` 优先于 `ProviderProfile.protocol` 选择上游 API，并决定使用哪份 Codec 默认值。
Model 的 `protocol=None` 继承 Provider；Model 明确配置时，Provider 可省略协议。
`stdio_protocol` 选择应用使用的消息格式，可以与上游协议不同。
`stdio_protocol` 未指定时，第一次生成从整个消息与工具列表中推断；只有唯一匹配时才绑定。
纯文本输入常同时符合多种协议，建议显式指定。零匹配、多匹配或空输入无法推断时抛出
`MessageValidationError`；推断失败不会绑定协议。绑定后后续调用沿用该协议。

实例提供以下可读属性；配置和嵌套数据应视为只读，更换配置时创建新实例：

| 属性 | 含义 |
| --- | --- |
| `provider` / `model` | 构造时传入配置的副本 |
| `resolved_profile` | 已解析的 `ResolvedProfile` |
| `profile_hash` | 构造时配置快照的整数哈希，只读 |
| `stdio_protocol` | 绑定的输入输出协议，推断前可为 `None` |
| `codec` | 当前实例的 `MyCodec` |
| `client_parameters` | 已准备的连接参数，供高级集成使用 |

### `MyLLM.compare(provider, model)`

返回 `True` 表示解析后的配置哈希与构造时一致；不一致时返回
`tuple[ResolvedProfile, int]`，供创建新实例复用。比较不更新当前实例配置或客户端，
每次使用当前实例的协议默认值重新解析、计算哈希；无效配置仍抛出解析异常。
元组也是真值，判断结果时应使用 `result is True` 或 `result is not True`。

```python
async def replace_if_changed(llm, provider, model):
    result = llm.compare(provider, model)
    if result is not True:
        replacement = MyLLM(
            provider=provider,
            model=model,
            resolve_profile_with_hash=result,
        )
        await llm.aclose()
        return replacement
    return llm
```

复用结果时必须传入对应的 provider/model，并保留原快照与哈希的配对。
构造函数信任该配对，不再次计算哈希或核对原始配置；为隔离实例而复制快照，仍进行协议校验。
快照按只读数据使用，同一结果可用于创建多个实例。
比较范围仅为解析后的 provider/model 配置，不包含 stdio_protocol、语义损失策略或消息缓存开关。

### `await MyLLM.generate(...)`

所有参数仅限关键字。`messages` 必填，其余参数可省略：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `messages` | 必填 | 非空、同协议的历史序列；元素可为字典或支持的官方 SDK 模型；middleware stdio 使用具体 `Message` 模型 |
| `stream` | `False` | `False` 等待完整结果；`True` 返回 `MessageStream` |
| `tools` | `omit` | 同协议工具声明；`None` / `omit` 不发送，`[]` 发送空列表 |
| `max_tokens` | 继承配置 | 正整数输出上限；显式 `None` 保留为空值，`omit` 删除本次请求的目标参数；目标 API 仍可要求有效数值 |
| `reasoning` | `None` | `True` 且 effort 为 `none` 时使用 `medium`；`False` 与非 `none` effort 冲突 |
| `reasoning_effort` | `"none"` | `none` / `low` / `medium` / `high` / `xhigh` / `max`，受模型支持状态和配置映射约束 |
| `extra_headers` | `None` | 请求头字典，按键覆盖配置默认值；字典中的 `omit` 删除对应键 |
| `extra_body` | `None` | 协议原生生成参数字典，按键覆盖配置默认值；不能覆盖模型、消息、工具等受管理字段 |
| `previous_response_id` | `None` | 非空服务端历史 ID，用于 `openai_responses` 上游；其他协议记录不支持的语义损失并省略；`None` / `omit` 不发送 |

实例的生成方法不接收 provider、model 或连接选项。输入历史和可变选项在协程被 `await` 时隔离，
调用不会自动更新业务历史。工具由调用者执行。

### 增量消息缓存

设置 `MyLLM(enable_message_cache=True, ...)` 后，首批仍传普通消息列表，后续只传新增消息：

```python
async def cached_conversation(provider, model):
    async with MyLLM(
        provider=provider,
        model=model,
        stdio_protocol="openai_chat",
        enable_message_cache=True,
    ) as llm:
        reply = await llm.generate(messages=[{"role": "user", "content": "Hello"}])
        return await llm.generate(messages=[reply, {"role": "user", "content": "Please continue"}])
```

第二次实际发送首批 user 消息、`reply` 和新增 user 消息。返回的 assistant 回复由调用者放入
下一批列表；Responses 的返回 items 使用 `[*reply, ...]` 展开。增量也可以包含 system、developer、
assistant 或一整批工具调用及结果；工具回合应放在同一批。工具关联只在当前批次修复，
已缓存的对应关系不会重复解析或修复，具体规则见[工具回合](../guides/middleware-io.md#模型与工具回合)。
Anthropic、Gemini 已有普通消息缓存时，可以单独追加 system 消息。

缓存保存消息翻译得到的列表，不保存 SDK 请求字段。当前批次先与历史合并，再由 Decoder 组装、
校验完整请求；请求准备完成后，在调用语义损失 adaptor 之前保存这份翻译结果。
格式或请求校验失败不会写入缓存；此后的 raise 策略、hook 或网络调用抛错不会撤销已保存的输入。
hook 修改只影响本次请求，不会自动回写消息缓存。使用者负责 hook 的全部行为，
包括修改翻译结果后需要进行的缓存维护。显式导入的工具图片预设已同步处理缓存与重试，
见[预设工具图片 hook](../guides/semantic-loss.md#预设工具图片-hook)。
流尚未消费或调用失败后，连续重发相同消息批次会复用已有历史，不再追加一次。
每次重发仍执行翻译和语义损失处理，发送时使用本次请求选项、消息历史和本次 hook 的修改。
例如，丢弃尚未消费的流后，可以修改输出上限并重试：

```python
async def retry_unconsumed(llm):
    # llm 已开启消息缓存，stdio_protocol 为 openai_chat。
    batch = [{"role": "user", "content": "Hello"}]
    abandoned = await llm.generate(messages=batch, stream=True)
    await abandoned.aclose()
    return await llm.generate(messages=batch, max_tokens=1024)
```

去重比较当前原始消息与最后一次成功追加批次的稳定哈希，字典键顺序不影响比较；
工具声明、请求头、stream 和生成参数不参与消息哈希。只去重连续相同批次，`A → B → A` 的最后一个
`A` 仍会追加。连续相同批次统一视为重试；继续对话时，应把上次 assistant 回复加入下一批增量。
消息含无法稳定序列化的非标准扩展对象时，沿用原有转换规则，该批不做哈希去重。
完整历史不会被自动识别为增量。
缓存按实例隔离，`aclose()` 时释放；
启用缓存时不能同时设置非空的 `previous_response_id`，`None` / `omit` 仍可使用。

**启用消息缓存后，同一 `MyLLM` 实例不支持并发生成。** 调用方必须先完成或关闭前一条流，再发起
下一次调用，也不能交错消费多个流。多个独立实例可以并发使用；关闭消息缓存时，同一实例仍支持
在一个异步事件循环内并发生成。

### 协议和返回值

`Protocol` 是字符串类型别名，内置协议如下；自定义协议须提前注册。

| `stdio_protocol` | 消息及工具所用格式 | 非流式返回 / `stream.result` | 流事件 |
| --- | --- | --- | --- |
| `openai_chat` | Chat messages / tools | assistant 消息字典 | ChatCompletionChunk 字典 |
| `openai_responses` | Responses input/output items / tools | output items 列表 | Responses 事件字典 |
| `anthropic` | Messages role/content / tools | assistant role/content 字典 | Messages 事件字典 |
| `gemini` | Content role/parts / tools | model role/parts 字典 | GenerateContentResponse 字典 |
| `middleware` | `Message` / `FunctionTool` 具体模型 | `AssistantMessage` | `MessageEvent` 具体模型 |

`middleware` 是必须显式指定的 stdio 名称，不注册新协议组件，也不能作为上游协议。
其模型列表、工具回合、事件字段和错误契约见[直接使用中间协议](../guides/middleware-io.md)。

非流式结果直接是消息或 items。流式完成原因通过原生终态事件或中间 `finish` 事件提供，
用量只通过 `get_last_usage()` 获取。
保存历史时，Responses 使用 `history.extend(result)`，其他协议使用 `history.append(result)`。
同一历史中的消息与工具必须遵循所选协议。

只转换已支持的字段；未知协议字段不保证保留，跨协议往返转换不保证无损。
工具参数、结果对象和 JSON Schema 等已识别开放字段中的应用数据会保留。
需要保留工具调用、思考签名或其他原生内容时，应保存完整返回值或 `stream.result`。

### `await generate(...)`

模块函数提供一次性调用：

```python
from my_llm import generate


async def ask_once(provider, model_id):
    return await generate(
        provider=provider,
        model=model_id,
        messages=[{"role": "user", "content": "Hello"}],
        stdio_protocol="openai_chat",
        max_tokens=512,
    )
```

签名为 `generate(*, provider, model, messages, stdio_protocol=None, semantic_loss_protocol="raise", input_semantic_loss_hook=None, output_semantic_loss_hook=None, **parameters)`。
`provider` 与实例入口相同；`model` 可为 `ModelProfile` 或非空模型 ID 字符串。
字符串直接作为模型 ID，不查找 Provider 的模型目录。
`semantic_loss_protocol` 与两个 hook 的含义同 `MyLLM` 构造参数；`parameters` 接收
`MyLLM.generate` 的其他生成选项。返回值和异常约定相同。

非流式调用结束后自动释放资源。返回流时，必须消费或显式关闭该流，以释放本次调用的资源。

### 协议推断与关闭

| 方法 | 返回值及约束 |
| --- | --- |
| `llm.guess_stdio_protocol(messages_or_tools)` | 返回已绑定协议；未绑定时从整个列表推断并绑定，非唯一匹配抛出 `MessageValidationError` |
| `await llm.aclose()` | 关闭实例；返回 `None`，可重复调用 |
| `async with MyLLM(...) as llm` | 退出上下文时自动关闭实例 |

先完成或关闭所有响应流，再关闭实例。关闭后调用 `generate` 或开始消费先前未启动的流会抛出
`RuntimeError`。同一实例须在一个异步事件循环内使用；关闭消息缓存时支持并发，启用时必须串行调用。

## `MessageStream`

通过 `await llm.generate(..., stream=True)` 或模块函数获取，使用 `async with` 管理生命周期：

```python
async def print_stream(llm, prompt):
    # llm 的 stdio_protocol 为 openai_chat。
    stream = await llm.generate(messages=[{"role": "user", "content": prompt}], stream=True)
    async with stream:
        async for event in stream:
            for choice in event.get("choices", []):
                delta = choice["delta"]
                print(delta.get("content", ""), end="")
                if delta.get("tool_calls"):
                    print(delta["tool_calls"])
    return stream.result
```

| 接口 | 含义 |
| --- | --- |
| `async for event in stream` | 按所选协议产生原生事件字典或中间 `MessageEvent`；流只能消费一次，同一流不能并发消费 |
| `stream.result` | 成功完整消费后的标准消息、items 或中间 `AssistantMessage`；此前、提前关闭或失败时为 `None`；重复读取返回本次最终 draft 本身，与其他调用隔离 |
| `await stream.aclose()` | 关闭响应并执行清理回调，返回 `None`；可在开始消费前调用，也可重复调用 |
| `stream.add_close_callback(callback)` | 登记无参数异步清理函数，关闭时按登记逆序各执行一次；关闭后登记抛出 `RuntimeError` |
| `async with stream` | 离开上下文时关闭流，适用于提前退出、异常和取消 |

文本和可见思考逐步输出；工具调用以完整参数提供，成功完成后才发布。
提前退出或失败时，不能将部分文本视为完成的工具调用或成功结果。
对可复用实例，关闭一个响应流不会关闭该实例。

## 中间协议模型

显式选择 `stdio_protocol="middleware"` 后，生成接口接收具体消息和工具模型，
完整返回 `AssistantMessage`，流式迭代返回 `MessageEvent`。
模型支持按字段名构造和直接读取属性；完整字段及协议归属见[中间协议契约](../design/protocol-middleware.md#支持集合)。

| 类型 | 公开契约 |
| --- | --- |
| `Message`、`UserMessage`、`AssistantMessage`、`InstructionMessage`、`ToolMessage` | `role` 与有序 `content`；可选 `name`、来源 `protocol`、`item`、`coalesce`、`origin`；具体类型不增加字段 |
| `TextContent` | `text`；固定 `type="text"`，可选 `item`、`origin` |
| `ImageContent`、`ImageSource` | 图片内容的 `source`、固定 `type="image"`、可选 `item/origin`；来源的 `type="url" / "base64"` 及 `url/data/media_type`；用户消息和工具结果可使用 |
| `ReasoningContent` | 可选 `reasoning: str \| None`、`feedback_item: Any \| None`、`item`、`origin`；固定 `type="thinking"`。可见文本实时输出，回传值保留原生对象，跨协议丢弃并记录损失 |
| `ToolCall` | `call_id`、`name`、完整 `arguments: JsonObject`；可选 `signature`、`opaque_protocol`、`item`、`origin`；固定 `type="tool_use"` |
| `ToolResult` | `call_id`、有序 `content: list[ResultContent]`；可选 `name`、`item`、`origin`；固定 `type="tool_result"` |
| `FunctionTool` | `name` 及可选 `description`、`parameters: JsonObject`、`strict`、`origin` |
| `ItemIdentity` | 可选 `id`、`status`；保存源输出项身份，与工具的 `call_id` 和流块的 `source_ref` 分开 |
| `Origin` | `path`、`value`；仅供错误定位，不作为 payload 字段输出 |
| `Usage` | 可选整数 `input`、`output`、`total`、`cached`、`reasoning`；缺失为 `None` |
| `ProtocolMiddleWare` | 可选 `messages`、`tools`、`stream`、`usage`；生成时传模型列表属性；字典投影中 `stream` 对应 `chunks` |
| `StreamBatch` | `protocol` 和内部操作 `events: list[dict]`，供 encoder 与状态机交互；应用流使用 `MessageEvent` |
| `MessageEvent` | `type`、可选 `index`、`source_ref`、`block`、`tool_index`、`field`、`value`、`reason`、`messages`；各事件有效字段见[流式输出](../guides/middleware-io.md#流式输出) |
| `MessageContent`、`ResultContent` | 分别为消息支持内容的联合类型、`TextContent \| ImageContent` |
| `JsonValue`、`JsonObject` | JSON 值和 `dict[str, JsonValue]` 类型别名，使用普通 Python 值与字典，无额外包装 |

消息及所有内容块最多各有一个可选 `item: ItemIdentity`。源项合并后内容仍保留自身项身份；
工具结果通过 `call_id` 对应调用，`ToolMessage.call_id` 是从结果内容读取的只读属性。
`reasoning` 统一保存可见思维文本，`feedback_item` 保存原生回传项；跨协议时仅后者丢弃并记录语义损失。

`to_wire()` 提供显式投影并借用部分嵌套值；它不提供深拷贝或自动校验。
生成请求在准备阶段隔离输入，公开结果和事件拥有自己的可变数据。
未知原生字段由 encoder 过滤。公共中间输入入口遇到未知内容类型、操作或增量字段，
以及缺少必需字段时，抛出带路径的 `MessageValidationError`；`ignore` / `hook` 不会跳过格式校验。
`decode(chunks=...)` 对 `StreamBatch` 和原始批次字典使用相同校验，再进入状态机。

校验函数也可从 `my_llm.middleware` 导入：`validate_content`、`validate_messages`、`validate_tools`、
`validate_operations`（接收批次列表）、`validate_events`；这些函数只校验，不复制或转换输入，
返回 `None`，可用 `path` 指定错误路径前缀。

## 配置接口

### `Attribute`、`Parameter`、`Config` 与 `omit`

包装类型均为可变、仅限关键字参数的 dataclass，通过 `.default_value` 读取值：

| 类型 / 构造参数 | 用途 |
| --- | --- |
| `Attribute(default_value=..., comment=None)` | 模型 ID、上下文窗口或模型目录等数据；不参与配置继承 |
| `Parameter(path=..., default_value=omit, supports=True, comment=None)` | 参数公共基类；具体 Profile 字段使用 InputParameter / OutputParameter |
| `InputParameter(path=..., default_value=omit, supports=True, comment=None)` | 请求或连接字段；source 固定为 input |
| `OutputParameter(path=..., event_type="", default_value=omit, supports=True, comment=None)` | 响应字段；source 固定为 output；event_type 非空时按事件类型精确匹配，空字符串沿用原生分类 |
| `Config(default_value=..., comment=None)` | 行为设置、连接选项或一组原生生成参数；字典按键合并 |
| `omit` / `Omit` | 显式省略标记及其类型；`Omit()` 返回同一标记，布尔值为假 |

`Parameter.name` 和 `Config.name` 由所属 Profile 的声明固定，不接受构造参数。
`Parameter.source` 由具体参数类型固定为 `input` 或 `output`，不接受构造参数；`comment` 仅用于说明。
`event_type` 匹配 SDK 输出对象的 `type`，Chat 无 `type` 时匹配 `object`；非匹配事件不应用默认值。
历史消息和原生最终快照沿用协议结构解析。
路径使用 `$.` 前缀、点分字段及非负列表索引，例如 `$.max_tokens`、`$.choices[0].delta.content`。
`supports=False` 表示不支持自动路由，并将参数默认值设为 `omit`。

配置的省略规则：

| 值 | 含义 |
| --- | --- |
| Profile 字段为 `None` | 继承上一层声明 |
| Profile 字段为 `omit` | 禁用该声明的自动应用 |
| 包装对象的 `default_value=None` | 显式的空值 |
| `Parameter(default_value=omit, ...)` | 不补默认值，已有原生值保留 |

Parameter 覆盖时替换完整声明。Config 字典按键合并，嵌套值整体替换，键值 `omit` 删除该键。
显式 token 控制和启用 reasoning 时生成的控制值优先于同位置的普通原生值。
禁用一个参数声明不会删除手动提供的原生字段；`generate(max_tokens=omit)` 则删除本次请求的 token 目标键。

### `ProviderProfile`

`ProviderProfile(*, ...)` 描述连接和上游协议，所有字段默认 `None`，并继承下文的共享覆盖项：

| 字段 | 包装类型 | 含义 |
| --- | --- | --- |
| `base_url` | `Config[str]` | 服务地址；未设置时遵循 SDK 默认值 |
| `api_key` | `InputParameter[str]` | 认证值及其位置；使用所选协议支持的密钥或认证头路径 |
| `timeout` | `Config[float]` | 正数，单位为秒 |
| `max_retries` | `Config[int]` | 非负重试次数 |
| `supports_stream` | `Config[bool]` | 生成要求支持流，默认继承协议的 `True` |
| `usage_mode` | `Config[str]` | 内置协议默认 `"none"`；协议不支持的非空模式在 Decoder 记录语义损失 |
| `models` | `Attribute[list[ModelProfile]]` | 可选模型目录，不自动选择生成模型 |

模型目录方法直接修改当前 Provider 的目录，不影响已经创建的 MyLLM：

| 方法 | 行为 |
| --- | --- |
| `add_models(model_profile)` | 接收单个 `ModelProfile` 或列表；按 ID 替换，新 ID 追加，已有 ID 保留原位置；重复 ID 后者优先 |
| `list_models()` | 返回目录列表副本，未设置时为 `[]`；其中的模型对象复用原对象 |
| `delete_model(model_id)` | 接收 ID 字符串列表，忽略不存在的 ID；空列表合法 |

修改方法返回 `None`；输入无效时抛出 `ConfigurationError`，目录保持原状。

### `ModelProfile`

`ModelProfile(*, model, ...)` 描述模型及其覆盖项：

| 字段 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `model` | 必填 `Attribute[str]` | 非空模型 ID |
| `max_context_window` | `Attribute[int]`，默认 `256000` | 描述性上下文窗口，不自动截断输入或限制输出 |
| `supports_function_calling` | `Config[bool] \| Omit \| None`，默认继承 | 协议默认 `False`；使用非空 tools 前启用 |
| `supports_vision` | `Config[bool] \| Omit \| None`，默认继承 | 协议默认 `False`；不支持时翻译会移除图片并标记语义损失，默认抛出 `SemanticLossError` |
| `supports_message_name` | `Config[bool] \| Omit \| None`，默认继承 | 协议默认 `True`；控制消息名称支持 |

`ModelProfile.supported_roles` 为固定类属性 `("system", "user", "assistant", "tool")`，不是构造参数。
Provider 与 Model 均为可变的、仅限关键字参数的 dataclass；用 `dataclasses.replace()` 创建配置变体。

### `OverridableProfile`

Provider 与 Model 共享以下覆盖项，均默认 `None`，表示继承：

| 字段 | 包装类型 | 含义 |
| --- | --- | --- |
| `protocol` | `Config[Protocol]` | Model 优先于 Provider；最终必须是已注册协议名，不接受 `omit` |
| `extra_headers` / `extra_body` | `Config[dict]` | 默认请求头与协议原生生成选项 |
| `max_tokens` | `InputParameter[int]` | 输出 token 上限 |
| `content` / `reasoning_content` | `OutputParameter` | 可见文本与思考字段 |
| `reasoning_enable` | `InputParameter` | 开启思考的参数；`supports=False` 表示不支持 reasoning，请求开启思考及已有思考内容均记录语义损失 |
| `reasoning_effort` | `InputParameter` | 思考强度参数及支持状态 |
| `reasoning_effort_mapping` | `Config[dict]` | 公共强度到原生值的映射；值为 `None` / `omit` 时该级别不可用 |
| `feedback_reasoning_item` | `OutputParameter[Any] \| Omit \| None` | 输出声明；从协议默认或自定义路径提取原生回传项；`None` 继承，`omit` 禁用 |
| `should_feedback_reasoning_item` | `Config[Literal["when_tool_calls", "always", "none"]]` | 内置协议默认 `always`；仅回传工具调用消息中的思考、全部历史思考或不回传；`omit` 等同 `none`，策略排除的思考不记录损失 |

`ReasoningEffort` 是 `Literal["none", "low", "medium", "high", "xhigh", "max"]` 类型别名。
可接受的强度还取决于具体模型及其配置。完整字段配置示例见[配置指南](configuration.md)。

### `Profile` 成员查询

`Profile` 是配置基类；Provider、Model、OverridableProfile 和 ProtocolDefaults 提供以下方法：

| 方法 | 返回值 |
| --- | --- |
| `get_attributes()` | 所有声明为 Attribute 的字段字典 |
| `get_configs()` | 所有声明为 Config 的字段字典 |
| `get_parameters()` | 所有声明为 Parameter 的字段字典，包括不支持的参数 |

结果按字段声明顺序排列，包含继承字段和未设置字段，排除类属性。
键为字段名，值是当前字段的包装对象、`None` 或 `omit`；不会展开 `.default_value`。
返回字典可独立增删，其中的对象与嵌套数据仍与原配置共享。

### `resolve_profiles` 与 `ResolvedProfile`

`resolve_profiles(provider, model, *, defaults_for=None) -> ResolvedProfile` 按协议默认值、Provider、Model 顺序解析，
返回独立配置快照。无效声明或值抛出 `ConfigurationError`，不可用协议抛出 `UnsupportedProviderError`。
直接集成时再调用 `MyCodec.validate_profile(resolved)` 检查协议约束；MyLLM 自动完成这些检查。
`defaults_for(protocol)` 可提供已有实例的 ProtocolDefaults；MyLLM 使用自身 MyCodec.defaults，
与后续验证和请求共用同一个协议组件实例。省略时独立创建所选注册 codec 读取默认声明。

`resolve_profiles_with_hash(provider, model, *, defaults_for=None) -> tuple[ResolvedProfile, int]`
先调用上述解析函数，再对完整快照计算 SHA-256 并转为整数。包含连接配置和 Parameter 的所有字段；
不额外包含已被覆盖的原始值或未进入快照的 Attribute。字典键、绑定元组和列表的顺序均保留，
顺序改变可能使哈希改变，即使两个字典用 `==` 比较相等。
快照值支持 JSON 标量、字典、列表、元组、bytes、集合、Enum 和 dataclass 实例；
不支持的对象或循环引用抛出 `ConfigurationError`。哈希仅用于配置指纹，不作为配置版本或认证凭据。

| 属性 / 方法 | 含义 |
| --- | --- |
| `protocol` / `model` | 上游协议名和模型 ID |
| `values` | 字段名到已解析默认值或行为配置的字典 |
| `parameters` | 字段名到 Parameter 声明或 `omit` 的字典 |
| `request_parameters` | 受支持的请求输入参数绑定元组 |
| `connection` | 已验证的连接选项字典 |
| `get(key, default=None)` | 读取值；缺失或 `omit` 返回 default，显式 `None` 保留 |
| `parameter(name, source=None)` | 返回受支持的 Parameter；缺失、禁用、不支持或 source 不符时为 `None` |

`source` 可选 `"input"` 或 `"output"`。默认值为 `omit` 的 Parameter 仍可受支持。
快照及查询返回的嵌套对象应视为只读，通常通过解析函数获取此类型。

## 工具回合

库返回工具调用，由应用执行工具、追加结果并再次生成。
下面示例使用 Chat 输入输出协议，并明确开启工具能力：

```python
import json


async def add_round_trip(provider, model_id):
    from my_llm import Attribute, Config, ModelProfile, MyLLM

    async with MyLLM(
        provider=provider,
        model=ModelProfile(
            model=Attribute(default_value=model_id),
            supports_function_calling=Config(default_value=True),
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

重复调用 ID 去重，相同 ID 的结果按内容出现顺序合并并保留重复内容，缺少结果补空；
包含孤立或无 ID 结果的消息整条丢弃。
结果 ID 不按名称或历史顺序推断。全部输入丢弃且没有缓存时抛出 `MessageValidationError`，不补造 user 提示。
Chat 上游将同一 assistant 的工具结果按到达顺序集中到其 `tool_calls` 后方，保持连续，
普通消息之间的顺序不变。同条源消息里的工具结果先于普通用户文本输出。
此排序在消息缓存与语义损失 hook 之前完成，缓存重试沿用该序列。
`previous_response_id` 用于 Responses 上游时，可仅发送新增消息或结果，并引用服务端已有的调用。
其他协议收到合法的非空 ID 时记录语义损失并省略；空字符串或错误类型仍属于无效参数。

## 语义损失与 usage

`MyLLM` 构造接收 `semantic_loss_protocol="raise" | "ignore" | "hook"` 及
`input_semantic_loss_hook` / `output_semantic_loss_hook`，实例上可用同名参数调用
`set_semantic_loss_policy()` 二次设置。hook 接收 `(draft, losses, context)` 并原位修改，
请求 hook 可以异步执行，准备阶段会等待其完成；离线转换与输出 hook 仍为同步函数，
具体值和示例见[迁移指南](../guides/semantic-loss.md)。

Chat、Gemini 分别提供 `prebuilt.move_tool_images_to_user_message`，由调用者显式导入并在
初始化时传入；不会自动改变请求行为或默认损失策略。预设读取实例解析后的
`supports_vision`，不能发送的图片替换为 `[Unsupported Image]`，其他损失视为已处理。
请求阶段的 `SemanticLossContext.profile` 提供解析后的 Profile；启用缓存时，
`context.message_cache.extend(messages)` 可显式补写本批新增原生消息并保留输入 fingerprint。
`message_cache.identity` 提供连续重试共享的弱引用标识；预设自己记录已处理批次并决定是否补写，
缓存不保存预设状态或执行补写去重。

流事件过滤接口返回的 usage；成功结束后调用 `get_last_usage()` 获取独立的 `Usage`。
未报告字段为 `None`，非流式结果不含 usage。关闭消息缓存后的并发调用中，该接口代表最后完成的流式调用。
无法解析的 usage 样本和辅助元数据静默忽略，不影响正文生成；没有可用用量时返回空。

## `MyCodec` 协议转换

`MyCodec(semantic_loss_protocol="raise", input_semantic_loss_hook=None, output_semantic_loss_hook=None)` 创建转换器，不发送请求。协议名必须已注册；官方协议需要安装对应 SDK 可选依赖。
转换调用不修改原始业务输入。

```python
from my_llm import MyCodec

codec = MyCodec()
converted = codec.convert(
    messages=[{"role": "user", "content": "Hello"}],
    input_format="openai_chat",
    target_format="gemini",
)
assert converted["messages"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
```

### 转换方法

| 方法 | 输入与返回 |
| --- | --- |
| `convert(messages=None, tools=None, *, input_format, target_format, profile=None, chunks=None, **parameters)` | 标准协议直接转换为目标标准协议 |
| `encode(messages=None, tools=None, *, input_format, profile=None, chunks=None, **parameters)` | 将标准值转换为具体的 ProtocolMiddleWare 及其类型化模型 |
| `decode(messages=None, tools=None, *, target_format, profile=None, chunks=None, **parameters)` | 将中间值转换为目标标准值 |
| `result(messages, *, target_format)` | 将中间消息列表转换为完整结果：单个消息字典，或 Responses items 列表 |

`convert`、`decode` 返回只包含本次转换集合的字典；`encode` 返回 `ProtocolMiddleWare`，
其 `.messages`、`.tools`、`.stream` 为具体模型列表。
字典投影只包含传入的 `messages`、`tools`、`chunks`；参数为 `None` 时省略，空列表仍保留。
`profile=None` 表示标准字段行为；`convert` 的 profile 和额外选项作用于目标转换。
额外选项由目标协议支持情况决定，例如事件中的 `model`。

`convert(chunks=完整事件列表, ...)` 支持离线转换完整响应流。
实时响应请使用 `MyLLM.generate(stream=True)`；转换器不记住前一次调用的流片段。
中间模型也可作为应用对话历史：显式设置 `stdio_protocol="middleware"`，然后将
`ProtocolMiddleWare.messages` / `.tools` 属性直接传给生成方法。`decode()` / `result()`
输出到 `target_format="middleware"` 时还须提供原生来源 `input_format`；无需目标 decoder。

### 验证、注册和高级调用

| 方法 / 属性 | 契约 |
| --- | --- |
| `codecs` | 构造时可用的协议名到 BaseCodec 的映射，按只读使用 |
| `MyCodec.register(CodecFactory)` | 注册返回独立 BaseCodec 及组件的零参数工厂；须早于使用它的 MyCodec / MyLLM 创建；重名或无效实现抛出 `ConfigurationError` |
| `match_protocol(messages_or_tools)` | 整批输入唯一匹配时返回协议名，否则为 `None` |
| `validate_input(protocol, messages, tools=None)` | 验证指定协议的输入，失败抛出 `MessageValidationError` |
| `validate_profile(profile)` | 验证已解析配置的协议约束，失败抛出配置或能力异常 |
| `decoder(protocol)` / `transport(protocol)` | 获取指定协议的 BaseDecoder / BaseTransport |
| `defaults(protocol)` | 只读借用本实例所选协议的 ProtocolDefaults，供配置解析使用 |
| `bind_input(inputs, protocol)` | 返回显式协议或从 messages/tools 唯一推断；推断时更新传入字典的这两个值；不记住协议 |
| `client_parameters(profile)` | 返回连接参数字典 |
| `await prepare(inputs, profile, *, input_format, client)` | 异步准备完整调用并等待请求 hook，返回 `client`、`request`、`local` 三个字典；inputs 需包含 messages、tools、stream 和使用的生成选项 |
| `request(intermediate, profile, **parameters)` | 从 encode 返回的消息与可选工具构造原生请求参数字典 |
| `upstream(client, request, profile)` | 返回异步上下文管理器，迭代原生事件；退出时关闭响应，由调用者管理 client |

`prepare` 的 `client` 参数来自 `client_parameters`。返回值中 `client` 是 SDK 构造参数，
`request` 是 SDK 调用参数，`local` 包含 `stream`。
参数准备接口不创建模型客户端或发出上游模型请求；显式注册的异步请求 hook 可以执行图片下载等 I/O。
新增注册仅对后续创建的实例可见；实现要求和完整接入示例见[协议接入指南](../guides/adding-protocol.md)。

四个内置 Codec 类从 `my_llm.codec` 导入，均无需构造参数，提供 BaseCodec 接口：

| 类 | 协议名 | 所需可选依赖 |
| --- | --- | --- |
| `ChatCompletionsCodec` | `openai_chat` | `my-llm-py[openai]` |
| `ResponsesCodec` | `openai_responses` | `my-llm-py[openai]` |
| `AnthropicCodec` | `anthropic` | `my-llm-py[anthropic]` |
| `GeminiCodec` | `gemini` | `my-llm-py[google-genai]` |

内置协议在对应依赖可用时自动注册，无需再次手动注册。

## 协议扩展接口

以下接口供自定义协议集成使用。具体字段定义与实现示例见[协议接入指南](../guides/adding-protocol.md)，
各方法的完整签名可通过 docstring 或 IDE 查看。

| 类型 | 公开契约 |
| --- | --- |
| `BaseCodec` | 抽象协议接口；提供 protocol_or_api_style、defaults、encoder、decoder、transport 属性 |
| `BaseEncoder` | 将原生消息、工具和事件转换为中间值；协议名、消息及工具 schema 由子类声明 |
| `BaseDecoder` | 将中间值转换为原生值，验证配置并构造原生请求参数 |
| `BaseTransport` | 创建、关闭 SDK 客户端并提供原生异步响应事件 |
| `ProtocolDefaults` | 协议默认配置，可变且仅限关键字参数；必填 should_feedback_reasoning_item，其余为标准参数与能力默认值 |
| `PreparedOptions` | 已验证的生成控制值，通常由 BaseDecoder.validate_parameters 返回 |

`BaseCodec` 提供 `encode_messages` / `decode_messages`、`encode_tools` / `decode_tools`、
`encode_chunks` / `decode_chunks`，签名均为 `(values, profile=None, **parameters)`。
编码分别返回 `Message`、`FunctionTool`、`StreamBatch` 模型列表，解码返回原生字典列表。
`match_protocol(values)` 返回本协议名或 `None`；`validate_input`、`validate_profile` 和 `request`
分别验证输入、验证配置、返回原生请求参数。

BaseEncoder / BaseDecoder 的 `messages`、`tools`、`chunks` 采用相同参数约定，分别执行对应方向的转换；
`usage(value)` 转换用量数据，缺失时返回 `None`。
BaseEncoder 还提供 `matches` / `match_item`、`validate_input`、`validate_values`、
`parameter_value`、`parameter_matches_event` 和 `request`。BaseDecoder 提供请求准备、过滤、验证及连接选项的扩展方法；
覆写要求见接入指南与对应方法 docstring。

BaseTransport 的 `create_client(parameters)` 返回调用者拥有的 SDK 客户端；
`await close_client(client)` 释放客户端；`upstream(client, request)` 提供异步事件上下文。
退出响应上下文后再关闭客户端。

`PreparedOptions` 的字段为 `tokens`、`headers`、`body`、`effort`、`explicit_tokens=False`、`losses=()`，
分别表示已解析 token 控制、请求头、原生选项、映射后的强度或 `None`、是否显式传入 token 控制，
以及待 Decoder 完成请求草稿后统一处理的不支持控制项。
其构造参数可按字段顺序或名称传入；嵌套字典按只读使用。

## 异常

| 异常 | 继承关系 | 含义 |
| --- | --- | --- |
| `MyLLMError` | `Exception` | 库异常基类 |
| `ConfigurationError` | `MyLLMError`、`ValueError` | 无效配置、路径或生成选项 |
| `UnsupportedProviderError` | `ConfigurationError` | 协议不可用或不能用于生成 |
| `SemanticLossError` | `MyLLMError` | 已支持字段无法表达或配置声明不支持；包含完整 draft、损失集合与方向上下文，由语义损失策略处理 |
| `MessageValidationError` | `ConfigurationError` | 输入格式、协议推断或工具调用与结果关联错误 |
| `StreamError` | `MyLLMError` | 流式或非流式响应失败 |
| `IncompleteStreamError` | `StreamError` | 响应未完整结束 |
| `ResponseFailedError` | `StreamError` | Responses 返回 failed / error 事件 |
| `ResponseIncompleteError` | `IncompleteStreamError` | Responses 返回 incomplete 状态 |

`StreamError(message, *, details=None, partial_message=None)` 及其子类提供：

- `details`：可用时为原生错误或状态信息。
- `partial_message`：可用时为与 stdio 协议一致的部分消息或 items，不能作为成功结果使用。

官方 SDK 的 HTTP、认证与传输异常保留原始类型；对已关闭对象的不合法操作使用 `RuntimeError`。
导入内置 Codec 或选择缺少对应 SDK 的上游协议时抛出 `ModuleNotFoundError`，错误信息包含所需安装项。

## 版本

`my_llm.__version__` 是当前库版本字符串。
