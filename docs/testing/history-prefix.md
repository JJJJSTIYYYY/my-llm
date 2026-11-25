# 连续请求的消息前缀矩阵

本测试从 `MyLLM.generate` 进入，使用官方 SDK 和本地 HTTP transport，分别检查关闭缓存时
提交完整历史、开启缓存时提交增量消息，是否改变已经发送的消息前缀。
它不访问真实服务，不测供应商缓存命中率。
带日期的执行结果见[无缓存结果](results/2026-11-25-history-prefix.md)与
[开启缓存的结果](results/2026-11-25-cached-history-prefix.md)。

## 运行与判据

在已安装默认开发及 SDK 依赖的仓库根目录执行：

```sh
uv run --no-sync pytest -q tests/integration/test_history_prefix.py
uv run --no-sync pytest -q tests/integration/test_history_prefix.py -k 'not uncached'
```

矩阵的行是 `middleware`、`openai_chat`、`openai_responses`、`anthropic` 四种
`stdio_protocol`，列是 Chat、Responses、Anthropic 三种上游。每格运行十种场景与四种
调用顺序，每种缓存模式共 `4 × 3 × 10 × 4 = 480` 个用例。

每个用例显式参数化 `enable_message_cache=False/True`，执行四轮请求，并在第二轮以相反的
`stream` 参数重试同一份输入，共五次 HTTP 请求。四种顺序分别为全非流式、全流式、
非流式起始交替、流式起始交替。

关闭缓存时每轮提交累计完整历史。开启缓存时首轮提交初始历史，后续只提交上一轮返回消息与
工具结果或用户消息；工具调用与对应结果放在同一增量批次。不能通过重新提交全部历史来
验证缓存模式，否则会重复追加旧历史。流式增量同样取 `finish.messages`。

比较对象是官方 SDK 实际发出的 JSON 请求中的 `messages`，或 Responses 的 `input`。
每轮与所有前轮比较：`current[:len(previous)] == previous`，并要求历史长度增加。
Anthropic 独立的 `system` 也必须保持相同。同一历史的重试比较整个请求对象。
比较保留全部字段、原生 ID、图片引用及内容顺序；JSON 对象的键顺序不作为差异。

流式回填严格使用 `MessageEvent(type="finish").messages`：

- middleware 直接将这些消息追加到调用者历史。
- 原生 stdio 将完整公开事件经 `MyCodec.encode(chunks=..., input_format=...)` 转成
  `MessageEvent`，再将 `finish.messages` 经新的 `MyCodec.decode` 转回原生输入。
- 非流式回填 `generate` 的完整返回消息；第二轮重试的返回值不追加到历史。

流式路径不读取 `MessageStream.result`，也不通过拼接文本或单个工具事件重建消息。
工具结果由调用者按返回的工具 ID 补充，随后按缓存模式提交完整历史或增量。

## 场景

| 场景 | 检查内容 |
| --- | --- |
| `text` | 四轮文本对话，中文、Emoji 与分段 SSE |
| `system-media` | 用户 Base64 图片、指令与后续文本；Anthropic 原生消息没有 system 角色，因此该输入行只携带用户图片 |
| `history` | 调用开始前已有完整工具回合及原生消息身份 |
| `single-tool` | 前两轮连续工具调用，再转两轮文本 |
| `parallel-tools` | 同名工具的两个不同调用，结果逆序回填，检查按 ID 关联 |
| `tool-only` | 工具轮没有正文，检查工具消息仍完整回填 |
| `empty-results` | 空字符串或空内容数组的工具结果 |
| `repair-history` | 首轮历史缺少工具结果，检查自动补齐后反复翻译的一致性 |
| `reasoning-not-replayed` | 显式 `none` 策略删除历史推理，保留历史答案 |
| `tool-images` | 工具结果中的 URL 与 Base64 图片；Chat 原生工具消息不能表达图片，该输入行改用工具文本结果和用户图片 |

除图片补偿边界外，矩阵使用默认语义损失 `raise`。Chat 上游接收其他三种输入的工具图片时，
显式启用 `chat_semantic_loss_hook`；测试限制 hook 只接受工具图片损失，并检查上下文的缓存状态。
该 hook 使用稳定的图片来源 SHA-256 引用，并将图片用户消息追加到完整历史末尾。
相应测试要求同历史重试的整个请求相等、两次图片引用及图片消息相同。
关闭缓存时，新增对话后的原始完整前缀仍因图片消息位置变化而不相等；测试同时验证图片消息之外的
历史前缀一致，不去掉任何 ID 来声称完整前缀一致。哈希规则见[语义损失指南](../guides/semantic-loss.md)。
开启缓存时补偿后的图片消息保存在原来的历史位置，后续增量直接追加在其后，要求完整前缀一致。

每种缓存模式额外的 72 个用例覆盖三种上游 × 原生同协议／middleware × 三种推理策略 × 四种调用顺序。
响应包含合成的推理及反馈信息，前两轮带工具、后两轮只有文本。回填后逐轮检查
`none` 不回传、`always` 全部回传、`when_tool_calls` 只回传工具轮推理，以及消息前缀一致。
这是可原生回放的推理链路；不将跨协议无法表达的签名或反馈信息视为无损转换。

## 补偿 hook 的首轮与第二轮

每种缓存模式额外 60 个用例逐一覆盖 `reasoning_content_feedback`、
`move_tool_images_to_user_message`、`chat_semantic_loss_hook`。组合 hook 分别测试仅推理、
仅图片、推理加图片，其余两个 hook 各测试对应补偿。每种组合覆盖 middleware、Responses、
Anthropic 三种输入与四种流式／非流式顺序；这些 hook 的目标均为 Chat 上游。

所有待补偿内容都在第一轮输入中。测试确认 hook 在首轮 HTTP 请求发出前已被调用，
损失种类符合预期，并在实际发送的消息中验证补偿结果。前缀基线是**首轮补偿后的实际请求**。
第二轮只追加首轮返回消息和新的用户消息，不新增待补偿内容；流式仍使用 `finish.messages`。
随后比较 `second.messages[:len(first.messages)]` 与 `first.messages`。
关闭缓存时，原始历史重新翻译仍可能再次触发 hook；比较不会使用补偿前的草稿。

纯推理补偿要求前缀一致。含图片补偿要求哈希引用及图片消息不变；关闭缓存时如实断言位置变化导致的原始前缀不一致，
保留当前行为记录。补偿从首轮开始，验收比较首轮与第二轮。
开启缓存时还要求 hook 仅首轮被调用，第二轮复用保存的补偿历史，完整前缀一致。

测试还独立检查输出正文、工具名和参数、最终请求的调用／结果 ID、内容出现次数、
输入未被修改、消息缓存存在与否符合开关，以及响应和 HTTP 客户端关闭。独立原生样例位于
`tests/helpers/generation_history.py`，不使用生产转换器生成上游响应或预期正文。

两个公开返回模式都会在内部消费上游 SSE，因此这里的“非流式”指 `generate(stream=False)`。
测试不覆盖供应商原生非流式 HTTP 响应，也不证明真实模型会返回相同答案或服务端缓存会命中。
