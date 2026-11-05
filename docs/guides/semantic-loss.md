# 中间协议与语义损失迁移

`MyCodec.encode()` 返回 SDK 无关的 `ProtocolMiddleWare`。通过 `.messages`、`.tools`、
`.stream`、`.usage` 访问具体模型；`.to_wire()` 用于检查内部投影，不应作为应用会话历史。
完整字段、方向与缓存边界见[协议定义](../design/protocol-middleware.md)。
也可显式设置 `stdio_protocol="middleware"`，直接使用 `.messages` / `.tools` 中的具体模型
作为应用历史，详见[直接输入输出指南](middleware-io.md)。

中间层仅支持文本、用户图片、工具结果图片、思维链、工具调用、函数定义和 usage。
audio、file、用户 metadata、refusal、citations、cache_control 等在源编码时静默丢弃。
`strict` 在 Chat / Responses / Anthropic 均保留。
JSON Schema 内应用属性不按这些名称删除。

如果目标无法表达已支持的值，默认抛出公开的 `SemanticLossError`。例如，Chat 无法表达
工具结果图片；Anthropic 无法把外来无有效签名的历史思维直接回放。思维不会拼入
最终回答，图片不会自动变成工具文本。错误提供 `.draft`、`.losses.items` 和 `.context`，
单项的 `.origin_value` 引用原始来源对象，不复制或遍历其中未支持的字段，并提供`$.messages[...].content[...]` 形式的 `.origin_path` 与原因。
decoder 完整翻译后统一报出本批损失，请求在发送前处理，响应在交付前处理。

配置声明不支持的字段与生成控制也由 decoder 记录损失，原因标记为不支持，沿用相同策略。
`supports_stream` 声明项目调用能力，禁用时仍抛出 `UnsupportedProviderError`；非法输入结构使用
`MessageValidationError`，无效配置或参数值使用 `ConfigurationError`。

`reasoning_enable.supports=False` 时，所有思考内容都记录为不支持的损失。
`ReasoningContent.feedback_item` 保存原生回传值，跨协议由 decoder 丢弃并记录语义损失。
`feedback_reasoning_item` 是输出 Parameter，选择回传值提取路径，`omit` 禁用提取。
`should_feedback_reasoning_item` 只决定回传全部历史的回传值、工具调用消息的回传值或不回传，
因回传策略排除的内容不记录损失；例如 `when_tool_calls` 会静默跳过没有工具调用的历史思考。
Chat 在回传策略启用且 assistant 含工具调用时，如果缺少思考正文，补
`reasoning_content=""`，不依赖本次生成是否开启 reasoning。
`none`、`omit` 或模型声明不支持 reasoning 时不补。Anthropic 仍要求真实原生签名，不伪造空签名。

创建 `MyLLM` 时选择策略，也可以在实例上二次设置：

```python
llm.set_semantic_loss_policy(semantic_loss_protocol="ignore")
llm.set_semantic_loss_policy(
    semantic_loss_protocol="hook",
    input_semantic_loss_hook=repair_input,
    output_semantic_loss_hook=repair_output,
)
```

`ignore` 不保留损失历史或输出日志；`hook` 未提供函数时使用空函数。
hook 接收 `(draft, losses, context)`，返回值忽略。请求 hook 可以是同步或异步函数，
`MyLLM.generate()` 与 `await MyCodec.prepare(...)` 会直接等待它完成；离线转换及输出 hook 使用同步函数。
hook 可以原地修改、抛错、
重排内容或不做修改。生成请求的 hook 收到完整 SDK 请求；离线 convert 的消息 / 工具 hook 收到相应目标集合。
启用消息缓存时，消息翻译结果在调用 adaptor 前已经保存。hook 对请求的修改不会自动回写缓存，
即使 hook 抛错也不撤销这份翻译结果；使用者自行负责 hook 行为及所需的缓存维护。
响应 `context.stage == "result"`
时收到完整原生消息 / 列表，实时流的其他阶段收到本批原生事件列表。
中间 stdio 的结果 draft 为 `AssistantMessage`，事件 draft 为 `MessageEvent` 列表。
实时流只能基于已经收到的块作决定，不能预读未来内容。

## 预设工具图片 hook

Chat Completions 的 `prebuilt` 目录提供
`move_tool_images_to_user_message`。它将本批工具结果中的图片按原顺序汇总到一条新的
user 消息，追加到本次历史末尾；原工具结果保留文字和调用关联。

| 上游目标 | 导入位置 | 图片支持 |
| --- | --- | --- |
| Chat Completions | `my_llm.codec.openai.chat.prebuilt` | URL、Base64 |

选择与上游目标匹配的 hook，最外层仍可使用 Anthropic 或 Responses 等协议：

```python
from my_llm import MyLLM
from my_llm.codec.openai.chat.prebuilt import move_tool_images_to_user_message

llm = MyLLM(
    provider=provider,  # 已配置为 openai_chat 上游。
    model=model,
    stdio_protocol="openai_responses",
    semantic_loss_protocol="hook",
    input_semantic_loss_hook=move_tool_images_to_user_message,
    enable_message_cache=True,
)
```

外部使用者必须显式导入处理器，在构造 `MyLLM` 时传给 `input_semantic_loss_hook`，
并显式选择 `semantic_loss_protocol="hook"`。预设不会自动注册或被请求链路直接调用，
默认 `raise` 策略不变。请求阶段通过 `SemanticLossContext.profile` 提供解析后的 Profile，
启用缓存时还提供 `SemanticLossContext.message_cache`；处理器不需要绑定 `MyLLM` 实例。

处理器依据解析后的 `supports_vision` 决定是否发送图片，包含模型声明、协议默认值与
`omit` 的结果。支持时搬移可表示的图片；模型不支持图片或协议无法表示该图片时，
逐张插入 `[Unsupported Image]`。工具调用仍需启用 `supports_function_calling`。
其余语义损失视为已处理，不再抛错。
图片来源支持 Anthropic、Responses 原生字典 / SDK 对象和直接中间模型。

启用 `enable_message_cache=True` 时，预设会显式补写新增 user 消息，并保留此前缓存的
系统指令、历史及当前输入 fingerprint。下一轮增量请求继续携带图片；连续重发相同批次
（包括尚未消费的流）不会重复插图；不同工具回合即使图片数据相同也分别保留。

占位消息也遵循相同的缓存与重试规则。各协议的 `prebuilt` 自己记录已处理的批次，
通过上下文中缓存批次的 `message_cache.extend(messages)` 补写新增消息。
`message_cache.identity` 是连续重试共享的弱引用标识，新输入批次使用新标识；
预设使用弱集合记录处理状态，缓存过期、清空或进入新批次后，旧状态可随之释放。
缓存不保存 hook 状态，也不对补写消息去重；何时补写、是否重试均由 hook 负责。
只有预设新增的原生消息会写入缓存，普通 draft 修改仍仅影响当前请求。
预设仅补写新增消息，用于请求阶段，
不补偿离线 `convert()` 或输出阶段的损失。

这些预设是显式启用的补偿行为，不会再次翻译 hook 已修改的原生 draft。
交给 transport、交付事件以及最终结果使用该 draft；`MessageStream.result`
重复读取返回本次最终结果本身。修改结果不影响其他调用或输入历史。

工具参数仍可从上游逐片到达，但仅在成功读取完整响应后以完整调用输出。
Chat 的并行工具调用出现在一个最终 `tool_calls` 列表；Anthropic / Responses
使用原生工具生命周期事件，每个调用携带完整参数，不发送工具参数 delta。

工具结果对象 / 数组在中间层序列化为文本。即使文本外观是 JSON，也按文本传递，避免再次猜测应用数据类型。

usage 不出现在公开事件或非流式结果中。流成功结束后，`llm.get_last_usage()` 返回独立的
`Usage`，包含 `input`、`output`、`total`、`cached`、`reasoning`；未返回的字段为 `None`，
不推导未报告的 total。提前关闭或失败不替换上次成功记录。关闭消息缓存后的并发调用中，
它代表最后完成的调用；每个调用仍有独立的内部累积状态。
无法解析的 usage 样本返回空；非法计数、嵌套计数结构和辅助元数据不会中断有效回复。
未知计数保持未知，不补造 token 数。元数据更新失败时整次更新丢弃，不保留部分写入。

自定义 codec 现在注册工厂：`MyCodec.register(CustomCodec)`。工厂必须每次创建独立的
codec、encoder、decoder 和 transport，并调用 `BaseCodec.__init__()`。
已有实例保留注册快照；组件可以持有实例状态。只按需构造使用的协议，损失策略与 SDK
客户端均属于 `MyLLM` 实例；共享的有界缓存只保存类型与校验规则。
