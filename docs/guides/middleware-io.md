# 直接使用中间协议

应用可以直接使用从 `my_llm` 导出的具体模型收发消息。构造实例时必须显式设置
`stdio_protocol="middleware"`；中间模型不参与协议推断，也不能混入原生协议的消息和工具列表。
`provider.protocol` 仍然选择真实上游，例如 `openai_chat`、`anthropic` 或 `gemini`。

```python
from my_llm import (
    Attribute, Config, ModelProfile, MyLLM, ProviderProfile,
    TextContent, UserMessage,
)

async def ask():
    # 连接和认证仍按通常方式配置；例如 OpenAI SDK 使用 OPENAI_API_KEY。
    provider = ProviderProfile(protocol=Config(default_value="openai_chat"))
    model = ModelProfile(model=Attribute(default_value="your-model"))
    history = [UserMessage(role="user", content=[TextContent("你好")])]
    async with MyLLM(
        provider=provider, model=model, stdio_protocol="middleware",
    ) as llm:
        reply = await llm.generate(messages=history)
        # reply 是 AssistantMessage，仍使用有序的具体内容块。
        for part in reply.content:
            if isinstance(part, TextContent):
                print(part.text)
        history.append(reply)
        return history
```

`generate()` 的 `messages` 接受非空的模型列表或元组，`tools` 接受 `FunctionTool` 列表或元组。
`tools=None` / `omit` 不发送工具，`tools=[]` 发送空工具列表。原生字典不能用于此入口；
来自 `MyCodec.encode()` 的值可以直接传入：

```python
middle = llm.codec.encode(
    messages=[{"role": "user", "content": "你好"}],
    input_format="openai_chat",
)
reply = await llm.generate(messages=middle.messages, tools=middle.tools)
```

传入 `ProtocolMiddleWare.messages` 和 `.tools` 属性，保留具体模型；字典索引提供的是供底层转换查看的投影。
`ProtocolMiddleWare` 容器本身不是 `messages` 参数。顶层一次性 `generate()` 同样支持此入口，
并继续管理临时实例和返回流的关闭。

## 模型与工具回合

所有具体模型及其字段类型均从包根导出，完整字段见[中间协议契约](../design/protocol-middleware.md)。

| 用途 | 公开类型 |
| --- | --- |
| 历史 | `Message`、`UserMessage`、`AssistantMessage`、`InstructionMessage`、`ToolMessage` |
| 内容 | `TextContent`、`ImageContent`、`ImageSource`、`ReasoningContent`、`ToolCall`、`ToolResult` |
| 工具与类型别名 | `FunctionTool`、`MessageContent`、`ResultContent`、`JsonObject`、`JsonValue` |
| 公开生成结果 | `AssistantMessage`、`MessageEvent` |
| 转换容器 | `ProtocolMiddleWare`、`StreamBatch` |
| 源项身份 | `ItemIdentity` |
| 用量和来源 | `Usage`、`Origin` |

历史角色为 `user`、`assistant`、`system`、`developer`。工具结果使用 `user` 角色的
`ToolResult` 内容块；`ToolMessage` 是这种消息的辅助类型，其 `call_id` 从内容块读取。
不要将原生 Chat 的 `role="tool"` 搬到中间消息中。

```python
from my_llm import AssistantMessage, FunctionTool, ToolCall, ToolMessage, ToolResult

tools = [FunctionTool(
    name="lookup",
    description="查询名称",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
)]
history = [
    AssistantMessage("assistant", [ToolCall("call_1", "lookup", {"name": "示例"})]),
    ToolMessage("user", [ToolResult("call_1", [TextContent("查询结果")])]),
    UserMessage("user", [TextContent("请继续")]),
]
```

发送非空工具声明时仍须在 Profile 中启用 `supports_function_calling`。
生成前统一修复工具关联，工具仍由应用执行：

- 调用缺少结果时补空结果；相同 ID 的调用保留首次调用，同名重复调用可补齐首次为空的参数。
- 相同 ID 的结果合并为一条结果，内容块按出现顺序追加，保留重复内容。
- 找不到此前调用的工具结果会导致所在消息整条丢弃，包括同条消息中的普通文本和其他结果。
  本条消息对工具关联的修改也会撤销。Responses 指定 `previous_response_id` 时允许引用服务端调用。
- 缺少调用 ID 时仅为已有调用分配技术标识；无 ID 的结果一律整条丢弃，不按名称或顺序推断关联。
  先丢弃无效结果，再为仍无有效结果的已有调用补空。Gemini / Ollama 的源 Encoder 也不推断结果 ID。
- 全部消息被丢弃且没有可用缓存时抛出 `MessageValidationError`，不补造 user 消息；已有缓存时复用历史。

修复不修改传入模型。启用消息缓存时仅修复当前增量批次，完整工具回合应放在同一批，
不会重新解析缓存前缀。独立的严格工具关联验证仍可用于检查历史。
工具结果必须携带明确 ID。Ollama 已安装 SDK 的历史字段不声明关联 ID，需可靠保存工具回合时，
可使用 middleware stdio 保存 `ToolCall.call_id`，再原样填入 `ToolResult.call_id`；不会仅凭名称关联。
修复通过原有消息、内容及结果部件校验循环收集，最终检查不扫描工具表或重建消息列表；
翻译在既有循环中惰性读取有效消息及必要的空结果。回传思维链补空规则共用同一实现。
`ToolCall.arguments` 是完整 JSON 对象，结果对象/数组应先序列化为 `TextContent`，图片使用 `ImageContent`。
图片支持用户消息及工具结果；指令为文本，思维和工具调用属于 assistant。
opaque 思维的 `opaque_protocol`、签名和加密内容仍需符合上游的原生回放条件。

`Message` 和所有内容块都可携带 `item=ItemIdentity(id=..., status=...)`，用于保留源输出项身份，
当前主要用于 Responses 回放。应用通常直接保存 encoder 或生成接口返回的模型，无须手动设置。
`ToolCall.call_id` / `ToolResult.call_id` 关联工具调用与结果，和源输出项 id 的用途不同。
流事件的 `source_ref` 是源块地址，`index` 是本次返回内容的顺序。

思维模型始终使用 `ReasoningContent`：正文为 `reasoning`，标记为
`mark="raw" / "summary" / "encrypted"`，可选摘要为 `summary: list[str]`，
隐去思维通过 `redacted=True` 表示。原生摘要对象、隐去块和签名字段由相应 codec 转换，
应用直接使用这些模型字段即可。

## 流式输出

```python
from my_llm import ToolCall

stream = await llm.generate(messages=history, stream=True)
async with stream:
    async for event in stream:
        if event.type == "delta":
            # field 为 text 或 thinking；不会出现工具参数碎片。
            print(event.field, event.value)
        elif event.type == "tool":
            call = event.block
            assert isinstance(call, ToolCall)
            print(call.name, call.arguments)

reply = stream.result  # 成功耗尽后为 AssistantMessage；此前或提前关闭后为 None。
usage = llm.get_last_usage()
```

`MessageEvent` 是累积和校验后的公开事件。`StreamBatch.events` 是供协议扩展和状态机使用的
内部操作字典列表；应用消费生成流时使用下表事件。

| `type` | 有效内容 |
| --- | --- |
| `start` | 响应开始 |
| `block_start` | `index`、`source_ref`、初始 `block` |
| `delta` | `index`、`source_ref`、`field=text / thinking`、`value` |
| `block_stop` | `index`、`source_ref`、完整文本/思维 `block` |
| `block` | `index`、`source_ref`、完整非文本块 |
| `tool` | `index`、`tool_index`、`source_ref`、完整 `ToolCall` |
| `finish` | `reason`、完整 `messages` |

工具调用及其后续块等到上游成功 EOF 后才集中交付；失败或截断不会泄露未完成调用。
用量不进入事件或回复，只通过成功流的 `get_last_usage()` 获取。事件、结果和不同调用之间
不共享可变应用内容；`stream.result` 重复读取返回同一结果对象。

## 错误、所有权与转换边界

错误类型、配置覆盖、开放的 `extra_body`、自定义 `Parameter`、服务端历史与资源生命周期均沿用原有契约。
无效模型或工具关联抛出 `MessageValidationError`。支持内容无法被目标表达时仍按
raise / ignore / hook 处理；请求 hook 收到完整原生 SDK 请求，返回 hook 收到中间模型结果或事件列表，
原位修改生效。请求来源位置引用实际传入的中间内容模型，不修改调用者的 `Origin`。

输入在 `await generate()` 的准备阶段隔离；调用者可在流消费前修改自己的历史。
应用仍负责追加完成回复。必须关闭未耗尽的流及实例。
`enable_message_cache=True` 对模型列表使用相同的增量与重试规则：连续相同消息批次不重复追加，
同一实例不支持并发生成，先完成或关闭前一条流再调用下一次生成。
不同实例可以并发；详细约定见[增量消息缓存](../reference/api.md#增量消息缓存)。

`to_wire()` 和字典索引提供转换投影，会借用嵌套应用值及内部操作；它们不是深拷贝接口。
需要直接调用生成方法时传 `.messages` / `.tools` 模型属性。
`Origin` 只记录错误定位来源，不作为 payload 字段输出。
未知原生字段由 encoder 过滤。公共中间输入入口遇到未知内容类型、操作或增量字段，
以及缺少必需字段时，抛出带路径的 `MessageValidationError`；`ignore` / `hook` 不会跳过格式校验。
`decode(chunks=...)` 对 `StreamBatch` 和原始批次字典使用相同校验，再进入状态机。

请求直接进入上游现有 `Decoder.prepare()`，跳过源 encoder。返回由上游现有 encoder 和
同一个请求状态机生成具体模型，跳过目标 decoder。没有注册名为 `middleware` 的 codec，
也没有增加 encoder、decoder 或 transport；该名称保留给 stdio。
SDK 客户端仍属于实例；Schema 缓存仅保存有界类型规则，中间输入、结果和来源不进入全局缓存。

离线 `MyCodec.convert(..., input_format=<原生协议>, target_format="middleware")`
也可得到具体消息、工具和完整公开事件。直接调用 `decode()` / `result()` 输出中间模型时，
须额外传入 `input_format=<原生来源>`，用于语义损失上下文。

性能方法与结果见[性能指南](performance.md#中间协议直接输入输出)。
