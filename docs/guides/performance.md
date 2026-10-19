# generate 性能测试

从项目根目录运行：

```sh
uv run --no-sync python -m tests.performance --json /tmp/my-llm-performance.json
```

测试真实 `MyLLM.generate()` 调用及官方 SDK。HTTPX MockTransport 立即返回预先构建的内存字节流，
没有 socket 连接、网络等待、模型推理或模拟 sleep。保留 SDK 请求序列化、SSE / NDJSON 解析、
协议转换、流状态累积、工具 JSON 拼接、响应关闭和结果复制的实际执行成本。

## 计时边界

| 指标 | 边界 |
| --- | --- |
| total_ms | 调用 generate 前，到完整结果返回；stream=True 时包含所有事件消费、响应清理与读取 stream.result。 |
| cpu_ms | 同一范围的进程 CPU 时间。 |
| prepare_ms | 仅流式：调用 generate 前，到 await generate 返回 MessageStream。此时 SDK 请求尚未执行。 |
| first_event_ms | 仅流式：调用 generate 前，到收到第一个输出事件，包含准备阶段；首事件可能是元数据，不能当作首文本 token。 |
| cold | 在新 MyLLM 实例上的首次调用，包含 generate 内部的 SDK 客户端延迟创建。 |
| warm | 完成 cold 和额外预热后，在同一实例上复用客户端的测量样本。 |

MyLLM 构造、实例最终关闭、负载 / 响应字节准备及验证断言均位于计时之外。
计时前先完成一轮独立实例的协议预检，所以 cold 表示首次 SDK 客户端调用，
不表示全新 Python 进程、首次模块导入或首次 Schema 缓存初始化。
保留 Python 默认 GC 行为；进程 CPU 时间与墙钟时间同时输出。

控制台给出 cold、warm 中位数、P95、CPU 中位数与串行 requests/s。
JSON 同时保存每个样本、流句柄 / 首事件时间、实际请求 / 响应字节数、事件 / HTTP 分块数、
负载规模、Git 版本和工作区状态、Python / 平台及各 SDK 版本。
P95 使用 nearest-rank；requests/s 是样本数除以样本总耗时，不是并发吞吐。

## 复杂负载与覆盖

默认运行 60 个场景：

- native：五协议分别使用自己的复杂原生请求，覆盖流式与聚合，共 10 个场景。
- cross：五种上游 × 五种输入输出协议，覆盖流式与聚合，共 50 个场景；采用各协议都能表达的请求特性。

请求和响应由测试独立编写，不使用 MyCodec 生成原生预期。每种输入按其 stdio 协议构建，
响应按上游协议构建。先使用固定协议 Schema 和官方 SDK 类型验证，再检查实际 SDK 发出的请求，
核对图片解码后的字节、完整工具 Schema、历史调用参数和结果数量。每轮测量后验证完成结果的文本、
思考内容、工具名称 / 参数及响应体关闭，避免错误返回或被简化的负载产生虚假的低耗时。

默认规模是 12 轮历史、12 个工具定义、每轮 4 个并行调用、2048 字符用户消息、
8192 字符输出文本、3 层嵌套工具 Schema、8 条嵌套参数记录、96 字符应用增量与 1024 字节 HTTP 分块。
请求包含系统指令、Unicode / 转义字符、图片和多轮工具结果；原生场景还覆盖：

| 协议 | 额外复杂路径 |
| --- | --- |
| Chat | developer / 消息 name、内联 PDF、交错工具参数增量、独立 usage 尾块。 |
| Responses | developer、内联 PDF、reasoning summary / encrypted_content、多项生命周期、交错工具增量与终态快照。 |
| Anthropic | 文本文档、cache_control、thinking / signature / redacted_thinking 历史、引用注释增量。 |
| Gemini | 内联文档、带签名的 thought / function call、多 part 输出、缓存及思考用量。 |
| Ollama | 原生 images / thinking、无 ID 的历史工具配对、对象工具参数和 NDJSON。 |

签名与加密字段使用离线测试值，只验证结构和回放路径。没有统一的“最复杂合法请求”上限：
长度及可组合能力受具体模型限制；这里提供覆盖复杂路径、可以继续放大的压力负载。

## 调整规模

每个场景默认额外预热 2 次、测量 10 次。单测环境与专门的性能测量分开运行，避免其他测试争用 CPU。

```sh
# 单协议的复杂原生请求；增加采样数，观察稳定分布。
uv run --no-sync python -m tests.performance --suite native --upstream anthropic --iterations 30 --json /tmp/anthropic-performance.json

# 指定一条跨协议路径，只测完整流式消费。
uv run --no-sync python -m tests.performance --suite cross --upstream openai_responses --stdio gemini --mode stream --json /tmp/responses-gemini-performance.json

# 放大请求历史、工具和响应；parallel-calls 必须不超过 tools。
uv run --no-sync python -m tests.performance --suite native --rounds 32 --tools 32 --parallel-calls 8 --text-chars 4096 --output-chars 16384 --argument-items 16 --schema-depth 4 --fragment-chars 32 --wire-chunk-bytes 256 --json /tmp/stress-performance.json
```

普通 pytest 使用小规模负载检查全部 60 条路径、无网络保证、计时边界和数据完整性，
不设置依赖机器速度的绝对耗时门槛：

```sh
uv run --no-sync pytest -q tests/performance
```

实现位于 [workloads.py](../../tests/performance/workloads.py)、
[transport.py](../../tests/performance/transport.py)、[runner.py](../../tests/performance/runner.py)
和[测试](../../tests/performance/test_generate.py)。

内部复制、Schema 计划与终态复用的实现及同负载对照结果，见
[优化记录](../audits/performance-optimization-2026-10-19.md)。
历史 extra="allow" 实现阶段的完整矩阵及同进程随机交替对照，见
[性能复测](../audits/performance-retest-2026-10-19.md)。该记录不代表当前支持未声明协议字段透传。
