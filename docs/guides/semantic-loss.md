# 中间协议与语义损失迁移

`MyCodec.encode()` 返回 SDK 无关的 `ProtocolMiddleWare`。通过 `.messages`、`.tools`、
`.stream`、`.usage` 访问具体模型；`.to_wire()` 用于检查内部投影，不应作为应用会话历史。
完整字段、方向与缓存边界见[协议定义](../design/protocol-middleware.md)。
也可显式设置 `stdio_protocol="middleware"`，直接使用 `.messages` / `.tools` 中的具体模型
作为应用历史，详见[直接输入输出指南](middleware-io.md)。

中间层仅支持文本、用户图片、工具结果图片、思维链、工具调用、函数定义和 usage。
audio、file、用户 metadata、refusal、citations、cache_control 等在源编码时静默丢弃。
按本次需求确认，`strict` 在 Chat / Responses / Anthropic 保留，在 Gemini / Ollama 丢弃。
JSON Schema 内应用属性不按这些名称删除。

如果目标无法表达已支持的值，默认抛出公开的 `SemanticLossError`。例如，Chat 无法表达
工具结果图片；Anthropic / Gemini 无法把外来无有效签名的历史思维直接回放。思维不会拼入
最终回答，图片不会自动变成工具文本。错误提供 `.draft`、`.losses.items` 和 `.context`，
单项的 `.origin_value` 引用原始来源对象，不复制或遍历其中未支持的字段，并提供`$.messages[...].content[...]` 形式的 `.origin_path` 与原因。
decoder 完整翻译后统一报出本批损失，请求在发送前处理，响应在交付前处理。

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
hook 是同步函数，接收 `(draft, losses, context)`，返回值忽略。它可以原地修改、抛错、
重排内容或不做修改。生成请求的 hook 收到完整 SDK 请求；离线 convert 的消息 / 工具 hook 收到相应目标集合。
响应 `context.stage == "result"`
时收到完整原生消息 / 列表，实时流的其他阶段收到本批原生事件列表。
中间 stdio 的结果 draft 为 `AssistantMessage`，事件 draft 为 `MessageEvent` 列表。
实时流只能基于已经收到的块作决定，不能预读未来内容。

例如，应用明确选择把 Anthropic 来源工具结果中的图片搬到 Chat 后续用户消息：

```python
def repair_input(draft, losses, context):
    images = []
    for loss in losses.items:
        value = loss.origin_value
        if not isinstance(value, dict) or value.get("type") != "image":
            raise ValueError(f"unhandled semantic loss at {loss.origin_path}")
        source = value["source"]
        url = source.get("url")
        if source.get("type") == "base64":
            url = f"data:{source['media_type']};base64,{source['data']}"
        images.append({"type": "image_url", "image_url": {"url": url}})
    draft["messages"].append({"role": "user", "content": images})
```

这是应用自行决定的补偿行为。库不会自动添加图片消息，也不会再次翻译 hook 已修改的原生
draft。交给 transport、交付事件以及最终结果使用该 draft；`MessageStream.result`
重复读取返回本次最终结果本身。修改结果不影响其他调用或输入历史。

工具参数仍可从上游逐片到达，但仅在成功读取完整响应后以完整调用输出。
Chat / Ollama 的并行工具调用出现在一个最终 `tool_calls` 列表；Anthropic / Responses
使用原生工具生命周期事件，每个调用携带完整参数，不发送工具参数 delta。

工具结果对象 / 数组在中间层序列化为文本；Gemini 目标使用 SDK 必需的 `{"result": text}`
对象包装。即使文本外观是 JSON，也按文本传递，避免再次猜测应用数据类型。

usage 不出现在公开事件或非流式结果中。流成功结束后，`llm.get_last_usage()` 返回独立的
`Usage`，包含 `input`、`output`、`total`、`cached`、`reasoning`；未返回的字段为 `None`，
不推导未报告的 total。提前关闭或失败不替换上次成功记录。并发时它代表最后完成的调用；
每个调用仍有独立的内部累积状态。

自定义 codec 现在注册工厂：`MyCodec.register(CustomCodec)`。工厂必须每次创建独立的
codec、encoder、decoder 和 transport，并调用 `BaseCodec.__init__()`。
已有实例保留注册快照；组件可以持有实例状态。只按需构造使用的协议，损失策略与 SDK
客户端均属于 `MyLLM` 实例；共享的有界缓存只保存类型与校验规则。
