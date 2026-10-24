# generate 性能测试

## 中间协议直接输入输出

```sh
uv run --no-sync python -m tests.performance.middleware --iterations 15 --warmup 2 --concurrency 4 --json /tmp/middleware-io.json
```

同一源码中比较原生 stdio 与 `stdio_protocol="middleware"`：五种上游分别覆盖聚合和流式，
共十组成对场景，按 N/M、M/N 交替顺序采样。输入包含相同历史、图片、工具 Schema、调用、
结果和原生思维；中间夹具在计时外生成并移除 encoder 来源引用，模拟直接构造模型的调用者。
每一对必须发出完全相同的 SDK 请求字节，响应字节及 HTTP 分块一致。
结果分别按公开格式提取文本、思维和工具参数验证；事件数量因输出格式不同而分别记录。

计时包括公开 generate 调用、输入检查与隔离、真实 SDK 解析、完整消费、响应关闭和结果读取。
夹具构造、断言、实例构造/关闭不计时。内存和缓存探测单独执行，四路响应屏障证明真实请求重叠。
预热后的 tracemalloc 数值包含结果和 HTTP 观测，不含夹具及客户端构造，不是进程 RSS。
此对照衡量直接中间输入输出的调用成本，不包含应用先从原生协议转成中间模型的成本，
也不表示网络请求或模型推理会获得相同比例的提速。

运行记录及原始样本见[直接中间协议验收与性能](../audits/middleware-io-2026-10-24.md)。

从项目根目录运行：

```sh
uv run --no-sync python -m tests.performance --json /tmp/my-llm-performance.json
```

测试真实 `MyLLM.generate()` 调用及官方 SDK。HTTPX MockTransport 立即返回预先构建的内存字节流，
没有 socket 连接、网络等待、模型推理或模拟 sleep。保留 SDK 请求序列化、SSE / NDJSON 解析、
协议转换、流状态累积、工具 JSON 拼接、响应关闭和结果交付的实际执行成本。

## 计时边界

| 指标 | 边界 |
| --- | --- |
| total_ms | 调用 generate 前，到完整结果返回；stream=True 时包含所有事件消费、响应清理与读取 stream.result。 |
| cpu_ms | 同一范围的进程 CPU 时间。 |
| prepare_ms | 仅流式：调用 generate 前，到 await generate 返回 MessageStream。此时 SDK 请求尚未执行。 |
| first_event_ms | 仅流式：调用 generate 前，到收到第一个公开输出事件，包含准备阶段；不表示网络侧首 token。 |
| cold | 在新 MyLLM 实例上的首次调用，包含 generate 内部的 SDK 客户端延迟创建。 |
| warm | 完成 cold 和额外预热后，在同一实例上复用客户端的测量样本。 |

MyLLM 构造、实例最终关闭、负载 / 响应字节准备及验证断言均位于计时之外。
计时前先完成一轮独立实例的协议预检，所以 cold 表示首次 SDK 客户端调用，
不表示全新 Python 进程、首次模块导入或首次 Schema 缓存初始化。
保留 Python 默认 GC 行为；进程 CPU 时间与墙钟时间同时输出。

控制台给出 cold、warm 中位数、P95、CPU 中位数与串行 requests/s。
JSON 同时保存每个样本、流句柄 / 首事件时间、实际请求 / 响应字节数、事件 / HTTP 分块数、
负载规模、Git 版本和工作区状态、Python / 平台及各 SDK 版本。
本轮 `MessageStream.result` 按 hook 身份约定返回最终 draft 本身；旧版读取会复制。计时包含各版实际结果交付成本，
不将这一行为变化归因于缓存加速。
P95 使用 nearest-rank；requests/s 是样本数除以样本总耗时，不是并发吞吐。

## 复杂负载与覆盖

默认运行 60 个场景：

- native：五协议分别使用自己的复杂原生请求，覆盖流式与聚合，共 10 个场景。
- cross：五种上游 × 五种输入输出协议，覆盖流式与聚合，共 50 个场景；请求和响应采用各协议都能表达的文本、图片与工具特性，不含跨协议思维链。

请求和响应由测试独立编写，不使用 MyCodec 生成原生预期。每种输入按其 stdio 协议构建，
响应按上游协议构建。先使用固定协议 Schema 和官方 SDK 类型验证，再检查实际 SDK 发出的请求，
核对图片解码后的字节、完整工具 Schema、历史调用参数和结果数量。每轮测量后验证完成结果的文本、
思考内容、工具名称 / 参数及响应体关闭，避免错误返回或被简化的负载产生虚假的低耗时。

默认规模是 12 轮历史、12 个工具定义、每轮 4 个并行调用、2048 字符用户消息、
8192 字符输出文本、3 层嵌套工具 Schema、8 条嵌套参数记录、96 字符应用增量与 1024 字节 HTTP 分块。
请求包含系统指令、Unicode / 转义字符、图片和多轮工具结果；原生场景还覆盖：

| 协议 | 额外复杂路径 |
| --- | --- |
| Chat | developer、交错上游工具参数增量、独立 usage 尾块。 |
| Responses | developer、reasoning summary / encrypted_content、多项生命周期、交错上游工具增量与终态快照。 |
| Anthropic | thinking / signature / redacted_thinking 历史、上游工具参数增量。 |
| Gemini | 带签名的 thought、多 part 输出、缓存及思考用量。 |
| Ollama | 原生 images / thinking、无 ID 的历史工具配对、对象工具参数和 NDJSON。 |

签名与加密字段使用离线测试值，只验证结构和回放路径。没有统一的“最复杂合法请求”上限：
长度及可组合能力受具体模型限制；这里提供覆盖复杂路径、可以继续放大的压力负载。
负载不包含 audio、file、文档、消息 metadata、引用与 cache_control；它们已不属于固定支持集合。
工具调用参数仍在上游分片传输，公开流的完整工具调用只在结束时输出，usage 只记录。
语义损失及 hook 行为由契约测试覆盖，避免把有损与无损调用混为同一项性能比较。

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

删除未知字段保留前的提交、本轮修复前后对照及原始样本，见
[2026-10-20 性能回退定位](../audits/performance-regression-2026-10-20.md)。
此前 Profile 重构后的测量单独保留在[同日早先复测](../audits/performance-retest-2026-10-20.md)。
五协议调用链、必要差异和抽象契约复核见[当前执行路径审计](../audits/request-path-audit-2026-10-20.md)。

内部复制、Schema 计划与终态复用的历史实现及同负载对照结果，见
[优化记录](../audits/performance-optimization-2026-10-19.md)。
历史 extra="allow" 实现阶段的完整矩阵及同进程随机交替对照，见
[性能复测](../audits/performance-retest-2026-10-19.md)。该记录不代表当前支持未声明协议字段透传。


## 请求链路的时间、内存、缓存与并发对照

```sh
python -m tests.performance.request_chain --iterations 9 --concurrency 4 --json /tmp/request-chain.json
```

该入口沿用上述 60 个场景、默认复杂负载、1024 字节 HTTP 分块和 benchmark/measure 的计时边界。
每个场景包含串行 cold、2 次预热和 9 次采样，以及四路并发的 2 批预热和 9 批采样。
每份响应体在首个字节前经过 asyncio.Barrier，断言 peak_active=4，证明请求实际重叠。
四路请求携带不同 x-request 头，并验证无串扰、结果完整及每个响应体关闭。
Ollama 根据不同 headers 保留四个客户端，其余协议共享一个；两版遵循相同规则。

tracemalloc 只在独立的串行与并发内存探针启用，启用前 gc.collect，不与计时采样混合。
内存指标包含生成结果和测试收集的 HTTP 观察值，排除预制负载与已创建客户端；它是 Python 分配峰值，
不是进程 RSS 或连接池大小。两版使用完全相同的探针。
缓存快照记录五类 LRU 的 hits、misses、maxsize、currsize，可检查暖请求是否只命中规则而不随请求增长。
与串行 cold 一样，内存和并发测量前已预热；不把它们解释为冷进程启动或网络吞吐。

要比较两个版本，在修改运行时源码前保存 src 快照，再用同一环境和相同测试脚本依次运行：

```sh
PYTHONPATH=/tmp/before/src python -m tests.performance.request_chain --iterations 9 --json /tmp/before.json
python -m tests.performance.request_chain --iterations 9 --json /tmp/after.json
```

脚本记录运行时源码和基准脚本 SHA256、Git revision、Python / 平台 / SDK 版本及原始样本。
运行两版时保持负载、解释器和依赖相同，避免并行运行测试或其他 CPU 密集任务。
比较工具强制核对负载、响应字节和 HTTP 分块的 SHA256，以及解释器、SDK、规模、采样数和完整基准代码指纹。
请求规范化可能改变发送字节；工具调用集中输出和 usage 过滤会改变公开事件数，这两项单独报告。
随后检查 P50/P95、准备耗时、并发批次耗时、内存及缓存；少量噪声场景需要交替复测确认。

```sh
python -m tests.performance.compare /tmp/before.json /tmp/after.json --json /tmp/comparison.json
```
可运行的无网络准备示例见 [request_pipeline.py](../../examples/request_pipeline.py)。

本轮改造的完整数据、原生交替复测和尾部诊断见
[2026-10-21 请求链路重构对照](../audits/request-chain-refactor-2026-10-21.md)。
实例构造可单独用 `python -m tests.performance.construction --json /tmp/construction.json` 测量；
Responses 收集尾部定位使用 `python -m tests.performance.gc_diagnostic /tmp/responses-gc.json`，
保留 GC、单独登记回收停顿，诊断数据不替换无回调计时。

固定中间协议和双向语义损失后的首次对照见[2026-10-24 验收与性能报告](../audits/protocol-middleware-2026-10-24.md)。
后续冗余清理、边界复核及清理阶段的复测见[同日清理报告](../audits/code-cleanup-2026-10-24.md)。
需求反向验收补漏后的最新源码和复测见[需求全量验收复核](../audits/requirements-acceptance-2026-10-24.md)。
其后的缓存命中检查与同负载优化试验见[Schema 缓存检查记录](../audits/schema-cache-2026-10-24.md)。
