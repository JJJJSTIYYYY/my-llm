# SenseNova 接入与端到端测试

使用 [SenseNova 官方文档](https://platform.sensenova.cn/docs)描述的兼容 API，
通过现有 `ProviderProfile` 接入，无需新增 Codec 或 Transport。
可运行配置见 [sensenova_e2e.py](../../examples/sensenova_e2e.py) 的 `provider()`。

| 上游协议 | Profile protocol | SDK base_url | 配置 |
| --- | --- | --- | --- |
| Chat Completions | `openai_chat` | `https://token.sensenova.cn/v1` | `max_tokens` 路径为 `$.max_tokens`；思考增量路径为 `$.choices[0].delta.reasoning` |
| Responses | `openai_responses` | `https://token.sensenova.cn/v1` | `reasoning.effort=none` |
| Anthropic Messages | `anthropic` | `https://token.sensenova.cn` | 显式 Bearer 鉴权；`thinking.type=disabled` |

脚本默认模型为 `sensenova-6.8-flash-lite`。密钥从 `SENSENOVA_API_KEY` 环境变量读取，
未设置时通过隐藏输入提示读取。不要把密钥写入源码、配置文件或命令行参数。

```bash
uv run python examples/sensenova_e2e.py --report /tmp/sensenova-results.json
```

该命令访问真实服务并产生费用，不随普通 `pytest` 运行。默认执行九条路径，
即三种 `stdio_protocol` 与三种上游协议的全部组合。每条路径验证：

1. `generate(stream=False)` 的汇总文本结果。
2. `generate(stream=True)` 的原生文本增量与最终汇总一致。
3. 流式函数调用 `add(19, 23)` 的名称、参数及原生调用 ID 回传；本地执行后再次请求，
   验证最终答案 `42` 及其流式增量。

共 27 个场景、36 次上游请求；SDK 重试关闭。汇总模式和流式模式都使用库现有的
上游 SSE 执行路径，因此这里没有单独验证供应商 HTTP 非流式响应。
客户端和响应流通过异步上下文关闭；本测试不独立测量连接资源泄漏。

可通过 `--model MODEL_ID` 更换模型；通过 `--upstream openai_chat` 或
`--stdio anthropic` 缩小矩阵。

## 2026-10-20 验收结果

最终完整矩阵 **27 passed，0 failed**，共 **36 次上游请求**。
以下每格均包含汇总文本、流式文本、流式工具往返三个场景。
原始记录见 [JSON 报告](../audits/data/sensenova-e2e-2026-10-20.json)。

| 输入/输出协议 ↓ / 上游 API → | Chat Completions | Responses | Anthropic |
| --- | --- | --- | --- |
| Chat Completions | 3/3 | 3/3 | 3/3 |
| Responses | 3/3 | 3/3 | 3/3 |
| Anthropic | 3/3 | 3/3 | 3/3 |

项目离线回归测试 **1952 passed**，新增脚本的 Ruff 与格式检查通过。
本次仅新增接入示例、指南与报告，未改动库的协议转换实现。

## 实测兼容边界

2026-10-20 的测试中，Chat 的 `reasoning_effort=none` 和 Anthropic 的
`thinking.type=disabled` 未完全阻止返回思考块。额外尝试 Chat 的
`thinking="disabled"` 返回 HTTP 400，因此保留最小文档配置。

库保留现有的外来思考翻译语义：Chat / Anthropic 的思考块转换到其他协议时，
部分内容可能成为可见文本。测试对同协议及 Responses 上游的输出要求全文精确匹配；
对上述跨协议路径要求最后一行精确匹配，同时严格比较完整流式文本与最终汇总。
报告的 `rendered_prefix_chars` 记录答案之前附加文本的字符数，不保存思考正文。
场景通过不代表服务端已关闭思考，也不代表跨协议转换无损。

本轮不覆盖图像、结构化输出、并行工具、开启思考的力度映射、服务端状态存储、
取消或故障注入。Responses 工具定义显式带 `strict=False`，符合当前 SDK 输入 Schema。
