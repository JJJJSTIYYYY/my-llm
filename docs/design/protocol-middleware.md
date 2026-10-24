# 中间协议与语义损失契约

本契约对应 `my-llm-refactor-requirements.md` 的 C1–C10。`strict` 按用户确认：
Chat、Responses、Anthropic 保留，Gemini、Ollama 静默丢弃。

## 支持集合

`ProtocolMiddleWare` 是独立于 SDK 的具体类；字段是以下具体模型组成的列表，
不是原生协议字段的并集。原生解析和输出组装可以使用临时字典，但跨 encoder / decoder
边界传递此类及其模型。所有模型按调用创建，不进入全局缓存。

| 模型 | 字段与含义 |
| --- | --- |
| `ProtocolMiddleWare` | `messages`、`tools`、`stream`、`usage`；未提供的集合为 `None`，显式空集合保留；字典索引时流集合的键为 `chunks` |
| `UserMessage` | `role=user`、有序 `content: TextContent / ImageContent`、可选 `name` |
| `InstructionMessage` | `role=system / developer`、有序文本；保留已有系统指令行为 |
| `AssistantMessage` | `role=assistant`、有序文本 / 思维 / 工具调用；工具调用不混入正文 |
| `ToolMessage` | `call_id`、可选 `name`；以有序 `ToolResult` 块保存文本 / 图片并与调用关联 |
| `TextContent` | `type=text`、`text: str`；可选 `item_id`、`item_status`（Responses 输出项身份） |
| `ImageContent` | `type=image`、`source: ImageSource` |
| `ImageSource` | `type=url / base64`，对应 `url` 或 `data` 与 `media_type`；不下载 URL |
| `ReasoningContent` | `type=thinking`、`reasoning: str`、`mark=raw / summary / encrypted`；可选 signature、encrypted_content、summary、同协议回放所需 id、`status`；opaque 字段带所属协议 |
| `ToolCall` | `type=tool_use`、`call_id`、`name`、`arguments: JsonObject`；可选同协议签名、`item_id`、`status` |
| `ToolResult` | `call_id`、可选 name、有序 `TextContent / ImageContent` |
| `FunctionTool` | `name`、可选 `description`、完整 `parameters: JsonObject`、可选 `strict`；JSON Schema 内应用字段保持原样 |
| `AssistantMessageChunk` | 明确的流操作种类、块引用 / 顺序、文本或思维增量、完整块 / 终态快照；原生工具参数碎片只在内部累积 |
| `MessageEvent` | 状态机累积后的公开事件：start、block_start、delta、block_stop、block、tool、finish；工具参数始终完整，终态包含具体消息模型 |
| `StreamBatch` | `protocol` 与有序操作；保留响应 id、模型、终止原因等流控制数据，不支持用户 metadata |
| `Usage` | `input / output / total / cached / reasoning: int \| None`；缺失不填零；各上游附加计数由 encoder 归入对应总量 |
| `Origin` | 支持值的原始位置与只读来源引用，只用于错误定位；不枚举未知字段，不写入 SDK / 用户 payload |

消息模型还带可选 `protocol`、`id`、`status` 与 `coalesce`（Responses 合并已识别内容）；
`ReasoningContent.redacted` 为真时投影为原生 redacted_thinking。内容、消息与工具模型另带
只用于错误定位的 `origin`，它不写入 wire；`Usage`、`StreamBatch`、`MessageEvent` 等容器没有该字段。

图片仅支持用户消息和工具结果。工具结果中的对象 / 数组可以序列化为文本；图片不能
序列化成工具文本。audio、file、refusal、citations、用户 metadata、cache_control 等
不在支持集合的结构在 encoder 投影时静默删除，随后仅验证支持结构。

`ReasoningContent.mark` 在中间投影中显式保留。原生解析未提供标记时，根据摘要、密文及
思维内容推导；显式标记不被重新推导覆盖。该标记仅属于中间协议，不写入原生 SDK 字段。

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
