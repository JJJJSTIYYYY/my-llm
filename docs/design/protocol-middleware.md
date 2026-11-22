# 中间协议与语义损失契约

本文说明中间层支持集合、显式投影和损失处理的设计契约。
应用如何构造模型见[中间协议使用指南](../guides/middleware-io.md)，接口用法见[API 指南](../reference/api.md)。

## 支持集合

`ProtocolMiddleWare` 及其消息、工具、结果模型独立于 SDK。模型表达固定支持语义，
同时明确保留少量同协议回放所需的可选字段；这些字段的归属如下。
encoder 负责原生投影，decoder 负责目标表达，Transport 只执行 SDK 调用。
请求与响应模型不进入进程级类型缓存；启用的消息缓存仅保存实例会话的已翻译输入历史。

| 模型 | 完整字段 | 语义与归属 |
| --- | --- | --- |
| `ProtocolMiddleWare` | `messages`、`tools`、`stream`、`usage` | 通用转换容器；未提供为 `None`，显式空集合保留；`stream: list[MessageEvent]`，字典投影键为 `chunks` |
| `Message` | `role`、`content`、`name`、`protocol`、`item`、`origin` | `role/content` 通用；`name` 由 Chat 表达；`protocol` 标识来源；`item` 为可选源项身份；`origin` 属运行时定位 |
| `UserMessage` | 继承 `Message` 全部字段，无新增字段 | `role=user`；有序文本和图片 |
| `InstructionMessage` | 继承 `Message` 全部字段，无新增字段 | `role=system / developer`；有序文本 |
| `AssistantMessage` | 继承 `Message` 全部字段，无新增字段 | `role=assistant`；有序文本、思维和工具调用 |
| `ToolMessage` | 继承 `Message` 字段，`role: Literal["tool"]`，新增必填的 `call_id: str`，并重写 `to_wire()` | `role=tool`；一条消息对应一次调用，内容直接为有序文本 / 图片；`name` 和 `item` 保存结果名称及源结果项身份 |
| `ItemIdentity` | `id`、`status` | 通用的可选源项身份，当前用于 Responses 回放；两个字段均为可选字符串 |
| `TextContent` | `text`、`origin`、`type`、`item` | 通用文本，`type=text`；`item` 当前保留 Responses 项身份 |
| `ImageContent` | `source`、`origin`、`type`、`item` | 通用图片，`type=image`；`item` 当前保留 Responses 项身份 |
| `ImageSource` | `type`、`url`、`data`、`media_type` | 通用；`type=url` 使用 URL，`type=base64` 使用数据与媒体类型；不下载 URL |
| `ReasoningContent` | `reasoning`、`feedback_item`、`item`、`origin`、`type` | `reasoning: str \| None` 保存可见思维；`feedback_item: Any \| None` 保存原生回传值；固定 `type=thinking`，来源协议由消息或公开流事件标识 |
| `ToolCall` | `call_id`、`name`、`arguments`、`origin`、`type`、`item` | 通用完整调用，`type=tool_use`、`arguments: JsonObject`；`item` 当前保留 Responses 项身份 |
| `FunctionTool` | `name`、`description`、`parameters`、`strict`、`origin` | 通用函数声明；`parameters: JsonObject` 保留应用 Schema；三种内置协议均保留 `strict` |
| `StreamBlock` | `index`、`block`、`source_ref` | 终态内容块及源引用，由公开完成事件携带 |
| `MessageEvent` | `type`、`index`、`source_ref`、`block`、`tool_index`、`field`、`value`、`reason`、`messages`、`protocol`、`metadata`、`output_blocks` | 通用公开事件；工具调用完整，终态携带具体消息模型；各类型的有效字段见[使用指南](../guides/middleware-io.md#流式输出) |
| `Usage` | `input`、`output`、`total`、`cached`、`reasoning` | 通用可选整数；缺失不填零；附加计数由 encoder 归入对应总量 |
| `Origin` | `path`、`value` | 运行时错误定位；路径与来源引用按只读使用，不枚举未知字段 |

消息和所有内容块至多各有一个 `item: ItemIdentity | None`。消息级身份描述整个源消息项，
内容级身份在合并多项为同一 assistant 回合后仍区分源项；二者不应与工具的 `call_id` 混用。
`call_id` 关联调用和结果，`source_ref=(scope, index, part)` 定位流中的块，
元数据中的 `response_id` 标识整个响应。它们具有不同作用域。
Responses decoder 按内容项边界分组，内容未提供身份时使用消息身份；空消息仍可保留消息身份。
共享分组函数内部识别连续的 Responses 来源 assistant 项，无需 decoder 传入分组协议。
Chat、Anthropic 合并组内内容；Responses 保留独立项与身份，但按组判断工具调用所需的推理回传；其他来源消息逐条处理。
公共消息模型不携带合并指令；角色或来源协议变化即结束分组。
该内部来源分组是当前接受的设计取舍，不要求各目标 Decoder 解析其他协议的原生结构，
也不为其在公共模型新增协议专用的兼容字段，详见[设计取舍](architecture.md#设计取舍)。

图片仅支持用户消息和工具结果。工具结果中的对象 / 数组可以序列化为文本；图片不能
序列化成工具文本。audio、file、refusal、citations、用户 metadata、cache_control 等
不在支持集合的结构在 encoder 投影时静默删除，随后仅验证支持结构。

`ReasoningContent.reasoning` 统一保存接口返回的可见思维文本，包括完整思维或摘要，流式文本实时累积。
`feedback_item` 保存下一轮接口需要的原始回传值：Responses 的完整 reasoning item、
Anthropic 的完整 `thinking` / `redacted_thinking` 块，
或 Chat 的原始思维文本。完整回传值保留到块和响应终态，不产生公开字段增量。
同协议按回传策略使用该值；跨协议由 decoder 丢弃并记录语义损失，可见文本单独处理。
源协议编码时不产生损失记录。

## 模型投影与内部流操作

具体 dataclass 是公共模型；`to_wire()` 提供转换管线使用的显式字典投影。
保留两种表示是为了让应用使用属性和明确类型，同时让 codec 组装原生结构；
不使用反射遍历任意字段，也不序列化 `Origin.value` 指向的原始对象。
`origin` 不成为 wire 键，仅附在内部来源包装上用于定位错误。
字段级往返测试覆盖全部模型字段；增加字段而未补充映射及用例时会失败。

模型属性与少量 wire 键有明确映射：`ReasoningContent` 使用同名的 `reasoning` 和 `feedback_item`；
`ToolCall.call_id → id`、`arguments → input`、
`ToolMessage.call_id → tool_use_id`；工具消息投影时生成内部 `tool_result` 包装，公共 `content` 直接为文本 / 图片列表。
原生消息含有多个结果或混合普通内容时，公共模型按原顺序拆为独立消息；原生请求准备保留源消息的整体修复边界。工具声明使用 `functions` 分组。
这些键属于中间契约，不接受旧版字段别名。`JsonObject` 是 `dict[str, JsonValue]` 的类型别名，
运行时仍使用普通字典。

`StreamBatch` 是 encoder 到状态机的内部载体，不从公开包导出，不作为公开 encode / decode 输入或输出。
其 `events` 直接承载 encoder 已构造并拥有的操作字典，不再经额外 DTO 拆解和还原。
操作种类为 `start`、`metadata`、`usage`、`snapshot`、`reason`、`finish`、`block_set`、
`block_append`、`block_update`、`block_delta`、`block_stop`。工具参数碎片仅在状态机内累积。
`block_update` 保留实际补丁字段，不用完整内容模型的默认值填充。
元数据操作只搬运 `path` 与 `values`，middleware 不识别其协议专有键，
也不把用户提供的未知原生 metadata 纳入支持集合；可搬运哪些字段由源 encoder 决定。
响应时间统一为通用 `created_at`，Chat decoder 在输出时映射回原生 `created`。

中间层支持集合之外的原生字段由 encoder 静默丢弃。已进入中间契约后，未知内容类型、未知操作种类、
不支持的增量字段属于中间协议错误，抛出 `MessageValidationError`，不静默删除。
公开 `encode(chunks=...)` 接收完整原生流，内部校验操作、累积并检查 EOF 后，返回 `MessageEvent`。
公开 `decode(chunks=...)` 只接收完整公开中间事件，并重新检查其结构，不接受内部批次。
公开事件通过 `protocol` 保留源协议，`finish.metadata` 保存 encoder 已提取上下文，
`finish.output_blocks` 保存 `StreamBlock` 终态块。经 `to_wire()` 和 JSON 保存后仍可独立翻译。
公开方法对非空流要求恰好一个末尾 `finish`，不跨调用保存片段；实时应用使用 `MyLLM.generate(stream=True)`。
公共消息、工具、事件读取入口同样拒绝缺少必需字段的结构；这些格式错误不受语义损失策略影响。
`block_update` 仍按部分补丁检查，`finish` 操作允许沿用此前的结束原因。
状态机及内部投影不重复兜底，也不通过捕获任意 `KeyError` / `TypeError` 掩盖应用或内部错误。

校验统一归入内部 `middleware/validation.py`，包含 `validate_content`、`validate_messages`、
`validate_tools`、`validate_operations`（批次及其内部操作）、`validate_events`（公开事件）。
codec 依赖这些内部校验函数；需要先校验再做能力过滤时，使用 `messages_from_validated_wire` /
`MessageEvent.from_validated_wire` 投影已检查的结果，不重复校验。
成功路径不构造错误路径字符串、路径节点或闭包；校验失败后才逐层补充字段名和数组索引。
工具结果允许的内容种类由 wire 与模型输入校验共享；可见 reasoning 只允许字符串或 None。
内部事件片段按各类型规则检查，结构字段只能用于对应事件种类。公开 `MyCodec.decode`
还会验证完整流：非空事件列表必须以唯一的 `finish` 结束，且终态必须包含
`reason`、`messages` 和 `output_blocks`。
这些检查不承诺检测应用主动破坏有效数据的所有方式，也不保证删除生命周期事件或单独改写一份终态快照后的行为。

显式投影会借用嵌套应用字典、原生回传值及内部操作，不能把 `to_wire()` 视作独立深拷贝。
输入在准备 SDK 请求时完成隔离；公开生成结果和各公开事件拥有自己的可变数据，
调用者修改历史或已收到的事件不会影响别的结果、事件或请求。

## 翻译与错误边界

encoder 只读取源协议，不接收目标协议。decoder 只实现自己的目标表达规则；
`feedback_item` 的归属使用消息或流的来源协议，跨协议翻译时丢弃并记录损失。

decoder 把无法表达的支持值记为 `SemanticLossItem(origin_value, origin_path, reason)`，
继续处理全部消息、工具或本批输出。完成原生 draft 后统一抛出
`SemanticLossError(draft, SemanticLossGroup(items, origin_source), context)`。
请求的 draft 是完整 SDK 请求；响应的 draft 是本次要交付的完整消息或事件列表。
实时流只能处理当前已收到的批次，不预读未来块；签名完整性在完整块 / 终态再校验。
Chat 请求翻译可额外关联 `SemanticLossTarget`，记录目标消息下标、字段和对应的中间层值。
工具结果排序完成后确定下标，缓存前缀追加后调整偏移，供显式预设精确修复；
这些关联只附在损失项上，不进入中间层模型字段、消息缓存或原生请求。

`SemanticLossContext` 包含源协议、目标协议、方向和阶段。请求阶段还提供解析后的 `profile`，
以及启用缓存时的 `message_cache` 批次句柄，供显式注册的 hook 追加或替换原生历史。
上下文不选择或调用预设，也不会自动缓存 hook 修改。`BaseCodec` 捕获错误并交给
实例所有的 `BaseSemanticLossAdaptor`。默认 `raise`；`ignore` 无日志、无损失历史；
`hook` 调用相应函数 `(draft, losses, context)`，函数返回值忽略；异步请求准备会直接等待
异步请求 hook，离线转换和输出 hook 使用同步函数。
hook 可任意原地编辑 draft，进入 transport 或返回用户的对象身份保持不变。
图片、思考、工具与消息名称能力由 `codec.shared.capabilities` 统一判断并记录原因，
各协议在现有消息和内容循环内调用，损失保留原始来源信息；
工具结果中的图片在子内容翻译时处理。`reasoning_enable.supports=False` 时所有思考内容记录为不支持的损失。
`should_feedback_reasoning_item` 只选择 `feedback_item` 的回传范围，策略排除的历史思考不记录损失。
输出 Parameter `feedback_reasoning_item` 选择原生回传值的提取路径，`None` 继承，`omit` 禁用提取。
历史消息按轮合并和回传判断共用一次分组；输出过滤只在内容实际改变时创建容器，公开出口继续保证结果独立。
`MessageStream.result` 返回该调用最终 draft 本身；重复读取不另做复制，也不与其他调用共享。
hook 未传时为空函数，但不会改变默认的 `raise` 策略。

Chat reasoning 进入 `reasoning_content`，Responses 进入 reasoning item。Anthropic 对历史思维所需签名不能伪造。工具图片只有 Responses /
Anthropic 原生表达；其余目标报告损失。能力声明阻止已识别支持字段表达时，同样报告损失，
包括不支持的生成控制和原生请求字段；仅 `supports_stream` 声明项目调用能力，禁用时直接拒绝生成。
目标 Schema 无法保持工具参数约束时报告对应工具定义损失。

消息 `name` 属于支持字段。Responses、Anthropic 无法表达消息名称时报告
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
进程级缓存不保存消息、图片、工具参数、原始响应、损失、hook、Profile 或 SDK 客户端。
可选消息缓存单独按实例持有已翻译历史，不改变这一进程级规则。
SDK 客户端由每次生成调用独立持有，响应正常结束、提前关闭、异常或取消后释放。
`get_last_usage()` 返回最后完成的流式调用的独立 usage 副本；非流式结果不携带 usage。

## 验证依据

支持集合、损失策略、签名回传和生命周期的复现入口见[测试文档](../testing/README.md)。
性能方法见[性能测试指南](../guides/performance.md)；历史样本只证明其记录的版本与环境，
不作为当前实现的性能保证。审查与不修复范围见[审查文档](../reviews/README.md)。
