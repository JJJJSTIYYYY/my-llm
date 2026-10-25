# 中间协议与语义损失契约

本文定义中间层的固定支持集合与转换边界。`strict` 按用户确认：
Chat、Responses、Anthropic 保留，Gemini、Ollama 静默丢弃。

## 支持集合

`ProtocolMiddleWare` 及其消息、工具、结果模型独立于 SDK。模型表达固定支持语义，
同时明确保留少量同协议回放所需的可选字段；这些字段的归属如下。
encoder 负责原生投影，decoder 负责目标表达，Transport 只执行 SDK 调用。
所有模型按调用创建，不进入全局缓存。

| 模型 | 完整字段 | 语义与归属 |
| --- | --- | --- |
| `ProtocolMiddleWare` | `messages`、`tools`、`stream`、`usage` | 通用转换容器；未提供为 `None`，显式空集合保留；`stream` 字典投影键为 `chunks` |
| `Message` | `role`、`content`、`name`、`protocol`、`item`、`coalesce`、`origin` | `role/content` 通用；`name` 由 Chat 表达；`protocol` 标识来源；`item` 为可选源项身份；`coalesce` 用于 Responses 项合并；`origin` 属运行时定位 |
| `UserMessage` | 继承 `Message` 全部字段，无新增字段 | `role=user`；有序文本和图片 |
| `InstructionMessage` | 继承 `Message` 全部字段，无新增字段 | `role=system / developer`；有序文本 |
| `AssistantMessage` | 继承 `Message` 全部字段，无新增字段 | `role=assistant`；有序文本、思维和工具调用 |
| `ToolMessage` | 继承 `Message` 全部字段，无新增字段 | `role=user`；用 `ToolResult` 保存工具结果；只读属性 `call_id` 从内容中读取 |
| `ItemIdentity` | `id`、`status` | 通用的可选源项身份，当前用于 Responses 回放；两个字段均为可选字符串 |
| `TextContent` | `text`、`origin`、`type`、`item` | 通用文本，`type=text`；`item` 当前保留 Responses 项身份 |
| `ImageContent` | `source`、`origin`、`type`、`item` | 通用图片，`type=image`；`item` 当前保留 Responses 项身份 |
| `ImageSource` | `type`、`url`、`data`、`media_type` | 通用；`type=url` 使用 URL，`type=base64` 使用数据与媒体类型；不下载 URL |
| `ReasoningContent` | `reasoning`、`mark`、`signature`、`encrypted_content`、`opaque_protocol`、`item`、`origin`、`type`、`summary`、`redacted` | `reasoning/mark` 通用，`type=thinking`；`summary: list[str]` 用于 Responses 摘要；`signature` 用于 Anthropic / Gemini 回放；`encrypted_content` 用于 Responses / Anthropic 密文；`redacted` 用于 Anthropic 隐去思维；`opaque_protocol` 标识签名或密文所属协议 |
| `ToolCall` | `call_id`、`name`、`arguments`、`signature`、`opaque_protocol`、`origin`、`type`、`item` | 通用完整调用，`type=tool_use`、`arguments: JsonObject`；`signature/opaque_protocol` 用于 Gemini 回放；`item` 当前保留 Responses 项身份 |
| `ToolResult` | `call_id`、`content`、`name`、`origin`、`type`、`item` | 通用结果，`type=tool_result`；有序文本 / 图片；`item` 当前保留 Responses 结果项身份 |
| `FunctionTool` | `name`、`description`、`parameters`、`strict`、`origin` | 通用函数声明；`parameters: JsonObject` 保留应用 Schema；`strict` 按上文协议集合处理 |
| `StreamBatch` | `protocol`、`events` | encoder 到状态机的内部载体；`events: list[dict]` 是有序操作，不作为应用流事件使用 |
| `MessageEvent` | `type`、`index`、`source_ref`、`block`、`tool_index`、`field`、`value`、`reason`、`messages` | 通用公开事件；工具调用完整，终态携带具体消息模型；各类型的有效字段见[使用指南](../guides/middleware-io.md#流式输出) |
| `Usage` | `input`、`output`、`total`、`cached`、`reasoning` | 通用可选整数；缺失不填零；附加计数由 encoder 归入对应总量 |
| `Origin` | `path`、`value` | 运行时错误定位；路径与来源引用按只读使用，不枚举未知字段 |

消息和所有内容块至多各有一个 `item: ItemIdentity | None`。消息级身份描述整个源消息项，
内容级身份在合并多项为同一 assistant 回合后仍区分源项；二者不应与工具的 `call_id` 混用。
`call_id` 关联调用和结果，`source_ref=(scope, index, part)` 定位流中的块，
元数据中的 `response_id` 标识整个响应。它们具有不同作用域。
Responses decoder 按内容项边界分组，内容未提供身份时使用消息身份；空消息仍可保留消息身份。

图片仅支持用户消息和工具结果。工具结果中的对象 / 数组可以序列化为文本；图片不能
序列化成工具文本。audio、file、refusal、citations、用户 metadata、cache_control 等
不在支持集合的结构在 encoder 投影时静默删除，随后仅验证支持结构。

`ReasoningContent.mark=raw / summary / encrypted` 在中间投影中显式保留。
原生 encoder 根据源语义设置标记，模型构造时补齐默认值；middleware 公共读取入口先检查结构和必需字段，
例如缺少 `mark` 时抛出带字段路径的 `MessageValidationError`，不推断原生格式或补齐缺失字段。
通过检查后的内部投影直接读取已确定字段。
隐去思维也使用同一 `type=thinking` 结构和 `redacted=True`，摘要为字符串列表；
对应的原生块类型、摘要对象和回放格式均由各自 codec 构造。标记不写入原生 SDK 字段。

## 模型投影与内部流操作

具体 dataclass 是公共模型；`to_wire()` 提供转换管线使用的显式字典投影。
保留两种表示是为了让应用使用属性和明确类型，同时让 codec 组装原生结构；
不使用反射遍历任意字段，也不序列化 `Origin.value` 指向的原始对象。
`origin` 不成为 wire 键，仅附在内部来源包装上用于定位错误。
字段级往返测试覆盖全部模型字段；增加字段而未补充映射及用例时会失败。

模型属性与少量 wire 键有明确映射：`ReasoningContent.reasoning → thinking`、
`opaque_protocol → protocol`、`ToolCall.call_id → id`、`arguments → input`、
`ToolResult.call_id → tool_use_id`；工具声明使用 `functions` 分组。
这些键属于中间契约，不接受旧版字段别名。`JsonObject` 是 `dict[str, JsonValue]` 的类型别名，
运行时仍使用普通字典。

`StreamBatch.events` 直接承载 encoder 已构造并拥有的操作字典，不再经额外 DTO 拆解和还原。
操作种类为 `start`、`metadata`、`usage`、`snapshot`、`reason`、`finish`、`block_set`、
`block_append`、`block_update`、`block_delta`、`block_stop`。工具参数碎片仅在状态机内累积。
`block_update` 保留实际补丁字段，不用完整内容模型的默认值填充。
元数据操作只搬运 `path` 与 `values`，middleware 不识别其协议专有键，
也不把用户提供的未知原生 metadata 纳入支持集合；可搬运哪些字段由源 encoder 决定。
响应时间统一为通用 `created_at`，Chat decoder 在输出时映射回原生 `created`。

源协议不支持的原生字段由 encoder 静默丢弃。已进入中间契约后，未知内容类型、未知操作种类、
不支持的增量字段属于中间协议错误，抛出 `MessageValidationError`，不静默删除。
`encode()` 和 `decode(chunks=...)` 共用操作校验；后者对 `StreamBatch` 模型及原始字典一视同仁，
即使模型在编码后被修改也会重新检查，再交给状态机。不允许混合内部批次与公开事件。
公共消息、工具、事件读取入口同样拒绝缺少必需字段的结构；这些格式错误不受语义损失策略影响。
`block_update` 仍按部分补丁检查，`finish` 操作允许沿用此前的结束原因。
状态机及内部投影不重复兜底，也不通过捕获任意 `KeyError` / `TypeError` 掩盖应用或内部错误。

校验统一归入 `middleware/validation.py`，公开接口为 `validate_content`、`validate_messages`、
`validate_tools`、`validate_operations`（批次及其内部操作）、`validate_events`（公开事件）。
codec 只依赖公开接口；需要先校验再做能力过滤时，使用 `messages_from_validated_wire` /
`MessageEvent.from_validated_wire` 投影已检查的结果，不重复校验。
成功路径不构造错误路径字符串、路径节点或闭包；校验失败后才逐层补充字段名和数组索引。
工具结果允许的内容种类、reasoning 标记集合由 wire 与模型输入校验共享。
事件必需字段和可选字段分别列于规则表中，`finish` 的 `reason`、`messages`、`output_blocks`
均可缺省；结构字段只能用于对应事件种类。

显式投影会借用嵌套应用字典、摘要列表及内部操作，不能把 `to_wire()` 视作独立深拷贝。
输入在准备 SDK 请求时完成隔离；公开生成结果和各公开事件拥有自己的可变数据，
调用者修改历史或已收到的事件不会影响别的结果、事件或请求。

## 翻译与错误边界

encoder 只读取源协议，不接收目标协议。decoder 只实现自己的目标表达规则；
opaque 签名所属协议是值的类型约束，不能跨协议套用或伪造。
中间签名统一为 `signature`；Gemini 原生 `thought_signature` 的入出映射仅由其 encoder / decoder 负责。

decoder 把无法表达的支持值记为 `SemanticLossItem(origin_value, origin_path, reason)`，
继续处理全部消息、工具或本批输出。完成原生 draft 后统一抛出
`SemanticLossError(draft, SemanticLossGroup(items, origin_source), context)`。
请求的 draft 是完整 SDK 请求；响应的 draft 是本次要交付的完整消息或事件列表。
实时流只能处理当前已收到的批次，不预读未来块；签名完整性在完整块 / 终态再校验。

`SemanticLossContext` 包含源协议、目标协议、方向和阶段。`BaseCodec` 捕获错误并交给
实例所有的 `BaseSemanticLossAdaptor`。默认 `raise`；`ignore` 无日志、无损失历史；
`hook` 调用相应同步函数 `(draft, losses, context)`，函数返回值忽略。
hook 可任意原地编辑 draft，进入 transport 或返回用户的对象身份保持不变。
`MessageStream.result` 返回该调用最终 draft 本身；重复读取不另做复制，也不与其他调用共享。
hook 未传时为空函数，但不会改变默认的 `raise` 策略。

Chat reasoning 进入 `reasoning_content`，Responses 进入 reasoning item，Ollama 进入
`thinking`。Anthropic / Gemini 对历史思维所需签名不能伪造。工具图片只有 Responses /
Anthropic 原生表达；其余目标报告损失。兼容开关和能力声明阻止已识别支持字段表达时，
同样报告损失。目标 Schema 无法保持工具参数约束时报告对应工具定义损失。

消息 `name` 属于支持字段。Responses、Anthropic、Gemini、Ollama 无法表达消息名称时报告
损失；Chat 保留名称，除非 Profile 禁用 `supports_message_name`，此时也报告损失。

## 生命周期与缓存

应用可[直接使用中间模型](../guides/middleware-io.md)，但必须显式设置
`stdio_protocol="middleware"`。请求跳过源 encoder，使用上游既有 decoder；返回经过
上游 encoder 和同一个状态机后交付 `AssistantMessage` / `MessageEvent`，跳过目标 decoder。
该名称仅用于 stdio，不在注册表中创建 codec 或其他协议组件，也不参与格式推断。

注册表仅保存工厂及固定协议定义。每个 `MyLLM` 创建自己的 codec、encoder、decoder、
transport、adaptor；注册快照不受后续注册影响。stdio / upstream 一经绑定不再动态切换。
损失收集器、draft、流状态和 usage 均属于调用；并发调用不共享这些可变数据。

允许共享的缓存仅包括有界的 SDK 类型分析、过滤 / 校验计划、声明路径与不可变规则。
不缓存消息、图片、工具参数、原始响应、损失、hook、Profile 或 SDK 客户端。
SDK 客户端仍由 `MyLLM` 实例持有并复用，到实例关闭时释放；不能按 LRU 淘汰活跃客户端。
`get_last_usage()` 返回最后完成的流式调用的独立 usage 副本；非流式结果不携带 usage。

## 验收与性能比较

验收覆盖五协议请求 / 响应转换、三种损失策略、完整 draft 与原位 hook、签名回放、
工具图片、未知字段删除、完整工具调用、usage、工厂隔离、并发与资源关闭。
比较使用同一解释器、依赖、基准代码和双方均支持的负载；旧版源码单独快照。
协议行为变化造成的事件数变化单独报告。时间测量不启用 tracemalloc；内存、缓存和
真实重叠并发独立探测，保留原始样本、源码摘要与可复现命令。
