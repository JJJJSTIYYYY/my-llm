# SenseNova 接入与端到端测试

通过现有 `ProviderProfile` 接入兼容 API，无需新增 Codec 或 Transport。
完整配置和测试场景见 [sensenova_e2e.py](../../examples/sensenova_e2e.py)。

| 上游协议 | Profile protocol | SDK base_url | 配置 |
| --- | --- | --- | --- |
| Chat Completions | `openai_chat` | `https://token.sensenova.cn/v1` | `max_tokens` 路径为 `$.max_tokens`；思维增量路径为 `$.choices[0].delta.reasoning` |
| Responses | `openai_responses` | `https://token.sensenova.cn/v1` | `reasoning.effort=none` |
| Anthropic Messages | `anthropic` | `https://token.sensenova.cn` | 显式 Bearer 鉴权；`thinking.type=disabled` |

## 运行

默认模型为 `sensenova-6.8-flash-lite`。密钥从 `SENSENOVA_API_KEY` 环境变量读取，
未设置时通过隐藏输入读取；不要写入源码、配置文件或命令行参数。

```sh
uv run python examples/sensenova_e2e.py --report /tmp/sensenova-results.json
```

该命令访问真实服务并产生费用，不随普通 `pytest` 执行。默认执行三种上游与四种
`stdio_protocol`（Chat、Responses、Anthropic、Middleware）的十二条路径，每条验证：

1. `generate(stream=False)` 的正文精确匹配 `SENSENOVA_OK`。
2. `generate(stream=True)` 的正文增量与最终正文一致，并精确匹配 `SENSENOVA_OK`。
3. 流式调用 `add(19, 23)`，检查名称、完整参数和调用 ID；本地得到 `42` 后回传工具结果，
   再次请求，检查最终正文精确匹配 `42` 及其流式增量。

Middleware 另在每种上游验证三个策略场景：

- 同时关闭图片、工具、消息名称与思维能力，检查一次输入 hook 收到完整请求及各项不支持原因，
  过滤后的请求仍能完成真实调用，且调用方输入保持不变。
- `feedback_reasoning_content="none"` 时省略历史思维，不产生输入损失。
- `feedback_reasoning_content="when_tool_calls"` 时省略没有工具调用的历史思维，不产生输入损失。

一次全部成功共 45 个场景、57 次上游请求；失败和补跑会改变实际请求总数。
策略场景使用独立实例和 `hook` 策略；其他场景使用命令行指定的损失策略。
汇总与流式模式都使用库的上游 SSE 执行路径，未独立验证供应商 HTTP 非流式响应。
客户端和响应流通过异步上下文关闭，本例不独立测量连接资源泄漏。

默认每次请求前等待 3 秒以降低 RPM 限流概率，SDK 自动重试关闭。
脚本默认 `semantic_loss_protocol="ignore"`，丢弃目标无法表达的思维或签名，
思维保持独立，正文仍要求精确匹配。库本身的默认策略仍为 `raise`。

可用以下选项缩小范围或调整行为：

| 选项 | 用途 |
| --- | --- |
| `--model MODEL_ID` | 更换模型 |
| `--upstream openai_chat` | 仅测指定上游，可传多个 |
| `--stdio middleware` | 仅测指定输入/输出格式，可传多个 |
| `--case capability_losses feedback_none feedback_when_tool_calls` | 仅测指定场景；策略场景只在 Middleware 格式运行 |
| `--request-interval SECONDS` | 调整每次请求前的等待时间 |
| `--semantic-loss-protocol raise` | 使用严格损失策略 |
| `--retry-failed --report /tmp/sensenova-results.json` | 仅补跑本机报告中的未通过项；成功项保持不变，历史失败保留在 `previous_attempts` |

Middleware 场景直接构造 `UserMessage` 与 `FunctionTool`，不通过原生转换生成输入。
结果须为 `AssistantMessage`，流事件须为 `MessageEvent`；工具以完整 `ToolCall` 交付，
事件与结果中的调用 ID 必须一致。第二轮使用 `ToolMessage` / `ToolResult` 回传结果。

## 2026-10-31 实测结果

完整运行 **45 passed，0 failed**，共 **57 次真实请求**，无需补跑。
原有十二条协议路径的 36 个场景使用 `ignore`；新增的九个 Middleware 策略场景使用 `hook`。
三个上游的能力损失场景均一次收集到五项输入损失，过滤后请求正常完成且输入未被修改；
Chat 和 Responses 实际返回的思维也触发了输出损失 hook。`none` 与 `when_tool_calls`
无工具调用场景均没有输入损失，正文增量与最终正文一致。
本次报告保存在本机 `/tmp`，密钥只通过隐藏输入传入进程，没有写入文件。

## 2026-10-25 实测结果

首次与补跑合并后为 **36 passed，0 failed**，其中 Middleware 场景 **9/9 通过**。
每格包含汇总文本、流式文本和流式工具往返三个场景。

| 输入/输出协议 ↓ / 上游 API → | Chat Completions | Responses | Anthropic |
| --- | --- | --- | --- |
| Chat Completions | 3/3 | 3/3 | 3/3 |
| Responses | 3/3 | 3/3 | 3/3 |
| Anthropic | 3/3 | 3/3 | 3/3 |
| Middleware | 3/3 | 3/3 | 3/3 |

| 轮次 | 本轮执行 | 本轮通过 | 本轮失败及原因 | 合并通过/失败 |
| --- | ---: | ---: | --- | --- |
| 首次，`raise` | 36 | 18 | 14 次限流、4 次语义损失 | 18/18 |
| 加入请求间隔，补跑 Responses / Anthropic 未通过项，仍为 `raise` | 15 | 8 | 6 次语义损失、1 次正文断言失败 | 26/10 |
| 按用户确认用 `ignore` 补跑剩余项 | 10 | 10 | 无 | 36/0 |

26 个场景在 `raise` 下已通过，剩余 10 个在 `ignore` 下通过；这是分轮合并结果。
Middleware 的九项在 `raise` 下就已全部通过。原始 JSON 日志已清理，此处保留结果与原因汇总；
以后复跑请保存本机报告。新增离线测试通过本地 HTTP transport 驱动真实 OpenAI SDK，
覆盖四种输入格式、三类场景，以及补跑、历史保留和错误脱敏。

## 首次四项语义损失的具体情境

下表的方向只描述响应转换。例如“Chat → Anthropic”表示业务使用 Anthropic 格式，
实际调用 Chat 上游，再转换为 Anthropic 格式返回业务。

| 响应转换 | 场景 | 实际返回与失败阶段 | `ignore` 后的结果 |
| --- | --- | --- | --- |
| Chat → Anthropic | 汇总文本 | 提示要求仅答 `SENSENOVA_OK`，上游仍返回 `delta.reasoning`；收齐流后，最终思维块缺少 Anthropic 原生签名，结果转换失败 | 省略无法签名的思维块，正文精确匹配 |
| Chat → Anthropic | 流式文本 | 同样的文本提示；流消费中输出首个外来思维块时，因缺少原生签名失败 | 省略该思维，正文增量与最终正文一致 |
| Chat → Anthropic | 工具往返 | 提示要求先调用 `add(a=19,b=23)`；第一轮返回思维，转换失败，尚未检查完整调用或回传工具结果 | 检查完整调用并回传 `42`，第二轮正文匹配 `42` |
| Anthropic → Chat | 工具往返 | 第一轮返回带签名的思维块；Chat 可表达思维文字，但无法保存 Anthropic 签名，流消费失败，尚未进入第二轮 | 舍弃无法表达的签名，完成工具往返，最终正文匹配 `42` |

前三项错误定位为 `$.chunks[1].choices[0].delta.reasoning`，原因为
`Anthropic cannot replay reasoning without its native signature`。
第四项定位为 `$.chunks[1].content_block`，原因为
`Chat cannot preserve another protocol's reasoning signature`。

这些错误发生在响应表达阶段，没有证明工具名称或参数有误。`ignore` 不会伪造签名，
也不表示跨协议无损。关闭思维的配置未完全阻止返回思维块，因此不能仅靠该配置避免损失。
具体规则见 [Anthropic Decoder](../../src/my_llm/codec/anthropic/decoding/decoder.py) 与
[Chat Decoder](../../src/my_llm/codec/openai/chat/decoding/decoder.py)。

## 其他失败与验证限制

首轮另外 14 项返回 HTTP 429、`rpm exhausted`：Responses 上游的三个 Middleware 场景和
Anthropic 上游的 11 个场景。加入请求间隔后补跑均不再限流；记录不足以确定实际 RPM 阈值。

第二轮 Anthropic → Anthropic 的汇总文本出现一次 `unexpected answer`。
当次未保存实际正文，无法确定具体偏差，也无法归因于模型或翻译；第三轮通过。
脚本现已在此类错误中包含最多 160 字符的正文片段，持久化前仍执行密钥脱敏。
完整原生响应、思维正文和签名值未保留，不能从汇总还原这些内容。

本例只验证不支持图片时的过滤，不验证上游视觉识别；也不覆盖结构化输出、并行工具、
开启思维的力度映射、服务端状态、取消或故障注入。
Responses 工具定义显式带 `strict=False`；正文匹配不代表服务端完全关闭思维或协议转换无损。
