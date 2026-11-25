# SenseNova 真实缓存命中矩阵

`examples/sensenova_cache_e2e.py` 从 `MyLLM.generate` 调用真实服务，覆盖四种 stdio 与三种
上游，开启 `enable_message_cache`。它同时检查 HTTP 消息前缀和服务端 usage；
本地前缀一致率与真实缓存命中率分别计算，不相互替代。
已执行结果见 [2026-11-26 真实缓存报告](results/2026-11-26-sensenova-cache.md)。

## 运行

在已安装开发与 SDK 依赖的仓库根目录运行：

```sh
uv run --no-sync python examples/sensenova_cache_e2e.py --report /tmp/sensenova-cache-results.json
```

密钥从 `SENSENOVA_API_KEY` 环境变量或隐藏输入读取，不进入脚本或报告。
默认模型为 `sensenova-6.8-flash-lite`，上游配置沿用 [SenseNova 接入示例](../../examples/sensenova_e2e.py)。
默认每个组合四轮、总计 48 次付费请求，每次间隔两秒，SDK 重试次数为零。
`--upstream`、`--stdio` 可缩小范围；`--records` 调整初始参考数据长度，默认 128 条；
`--request-interval` 控制间隔。原始报告默认保存到 `/tmp`，不包含认证头、正文或推理内容。
补跑使用 `--retry-failed --report /tmp/sensenova-cache-results.json`，保留成功项和原失败，
只重跑未通过的组合。记录数及模型必须与已有报告一致，补跑使用新的会话前缀。

## 负载与判据

每个会话有独立随机标记及由该标记产生的长参考数据，减少不同矩阵格之间的缓存复用。
四轮依次验证短答、记住首轮口令、调用 `add(19,23)`、回填工具结果并回答 42。
每格交替使用公开的流式与非流式返回；全部底层请求仍是服务端 SSE。
流式成功耗尽后从 `stream.result` 回填，后续只提交消息增量，工具声明每轮保持一致。
正文精确匹配单独记录在 `answer_matches`，不匹配仍将该组合标记为失败，但继续完成剩余轮次，
以免将模型正文检查失败误当作缓存无法测量。工具调用格式或参数错误、网络或前缀错误仍终止该组合。

真实 HTTP 请求体仅在进程内保存，第二轮起逐轮比较前一轮消息列表是否仍是完整前缀，
并检查独立 `system` 和工具声明。实际官方 SDK 事件用于采集 usage，事件正文不写入报告。
兼容模式沿用已有真实 E2E 的 `semantic_loss_protocol="ignore"`，避免跨协议原生推理反馈
无法表达时中止；正文、工具名、参数和后续答案仍独立验证。不宣称完整保留原生推理反馈。

| 上游 | 缓存读取 token 来源 | 总输入 token 来源 |
| --- | --- | --- |
| Chat | `usage.prompt_tokens_details.cached_tokens` | `usage.prompt_tokens` |
| Responses | `response.usage.input_tokens_details.cached_tokens` | `response.usage.input_tokens` |
| Anthropic | `usage.cache_read_input_tokens` | `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` |

Anthropic 合并 message_start 和 message_delta 中的 usage。字段缺失保留为 None，不当作零。
流式额外核对 `get_last_usage().cached` 与服务端字段相等；非流式公共接口不暴露 usage，
使用观察到的上游 usage。该观察只转发原事件，不替换响应或调整计数。

首轮单独记录，命中率主要统计第 2–4 轮成功且缓存字段已报告的请求：

- 请求命中率：`cached_tokens > 0` 的请求数／可统计请求数。
- token 命中比例：这些请求的缓存读取 token 之和／总输入 token 之和。

没有匹配字段的请求标记为未报告，失败请求不算作零命中。没有对每轮必然命中设置断言，
服务端预热、调度、缓存容量或策略可能影响结果；本测试没有识别具体原因或验证 TTL／阈值。
耗时包含请求准备、网络、生成及关闭，不用于宣称缓存加速比。

如果实际遇到 HTTP 429，记录异常类型、状态码及传输观察处与调用者捕获的异常是否为同一对象。
不会为了验证限额主动压测；未发生限额时不宣称该分支已经实测。

离线测试 `tests/integration/test_sensenova_cache_e2e.py` 使用官方 SDK 与本地 HTTP 响应，
核对 12 个组合的四轮流程、计数来源、缺失字段及资源关闭；这些模拟值不计入真实结果。
