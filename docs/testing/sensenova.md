# SenseNova 端到端测试

本页面向复现与验证。接入配置见 [SenseNova 使用指南](../guides/sensenova.md)，
输入历史与推理回传的详细判据见 [输入修复与推理回传测试](input-e2e.md)。
脚本使用本项目的 Profile、生成、翻译与预置 hook，未在外层重写协议转换。
推理回传的 Transport 包装只记录原生响应和实际 HTTP 请求，不替代原 Transport 或官方 SDK。

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
- `should_feedback_reasoning_item="none"` 时省略历史思维，不产生输入损失。
- `should_feedback_reasoning_item="when_tool_calls"` 时省略没有工具调用的历史思维，不产生输入损失。

原有测试共 45 个场景、57 次上游请求。新增的
[输入修复、工具图片与推理回传测试](input-e2e.md) 共 40 个场景、122 次请求，默认也会执行；
完整成功运行合计 85 个场景、179 次请求。失败和补跑会改变实际请求总数。
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
| `--case reasoning_replay` | 严格检查上一轮非空原生推理与下一轮实际 HTTP 回传一致，覆盖同协议及 Middleware 格式 |
| `--request-interval SECONDS` | 调整每次请求前的等待时间 |
| `--semantic-loss-protocol raise` | 使用严格损失策略 |
| `--validate` | 不读取密钥、不联网，检查新增输入用例的请求准备结果 |
| `--retry-failed --report /tmp/sensenova-results.json` | 仅补跑本机报告中的未通过项；成功项保持不变，历史失败保留在 `previous_attempts` |

Middleware 场景直接构造 `UserMessage` 与 `FunctionTool`，不通过原生转换生成输入。
结果须为 `AssistantMessage`，流事件须为 `MessageEvent`；工具以完整 `ToolCall` 交付，
事件与结果中的调用 ID 必须一致。第二轮使用 `ToolMessage` 回传结果。

## 判断结果

区分请求准备失败、目标协议语义损失、服务端拒绝、限流和模型正文断言失败。
正文断言失败时，只有原生响应与交付结果的独立比较才能判断偏差发生在服务还是翻译阶段；
不能仅据最后的 `unexpected answer` 将失败归因于 Codec。
推理回传的 `raise` 场景应核对非空源数据、交付结果和下一轮实际 HTTP JSON，
不能用补空值或仅检查对象类型代替回传一致性。

报告中的 `results` 是每个场景的最终状态，`previous_attempts` 保留补跑前失败。
汇总场景数应按唯一的上游、stdio 和 case 组合计算；累计请求数需要连同历史尝试统计，
不能把最终行中的 `requests` 简单相加当作整个运行的真实请求量。

## 验证范围

正文与工具场景不覆盖结构化输出、并行工具、服务端状态、取消或故障注入。
图片场景验证发送结构与服务接受情况，不评价视觉识别能力。
脚本使用 SSE 完成流式与汇总调用，不独立验证供应商 HTTP 非流式响应。
结果记录只证明记录日期、服务、模型和配置下的行为。

## 结果记录

- [2026-10 SenseNova 历史结果](results/2026-10-sensenova.md)。
- [2026-11 输入修复与推理回传历史结果](results/2026-11-input-e2e.md)。
- [2026-11-22 SenseNova 验证结果](results/2026-11-22-sensenova.md)。
