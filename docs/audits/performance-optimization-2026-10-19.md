# 内部复制、Schema 计划与终态物化优化

> 历史验收记录：本文描述对应审计时点。当前翻译字段契约以[翻译字段边界](../design/translation-boundaries.md)为准；文中未知字段透传、恢复记录及开放模型扩展结论已撤销。

复核日期：2026-10-19。使用真实官方 SDK 和内存 HTTP 响应，排除网络与模型推理等待。

## 已实现的职责与数据边界

- Encoder 的公开转换入口只捕获一次输入；私有钩子复用这份隔离数据。SDK JSON dump 已生成独立树时，不再再次复制。
- 历史校验、来源分组和回放比较只读借用数据；合并时只重建变化的容器。来源快照、公开结果和可变选项仍在实际输出边界隔离。
- Transport 只重建实际替换的请求父容器，过滤器生成独立消息与工具。输入校验使用只读 projected，不再复制仅用于校验的工具参数、结果或 JSON Schema。
- Schema 过滤预编译完整字段和 Union 计划；supported 与 projected 共用类型缓存，保留最高评分、首个同分候选、嵌套 Union 顺序、Annotated 和递归模型语义。缓存不持有请求、Profile 或 SDK 客户端。
- BlockState 缓存当前内容版本的物化结果；修改使缓存失效，嵌套更新按需重建。成功终态复用这些块，工具 JSON 不在 finish/messages/result 中重复解析。
- MyLLM 直接使用完成事件已经验证的消息进行结果转换。Encoder/Decoder/Transport 的职责、五协议共同编排和官方 SDK 生命周期保持一致。

实现细节见[架构](../design/architecture.md#数据所有权与缓存)及[流状态](../design/stream-state.md)。

## 同负载性能对照

环境：Python 3.12.12，macOS-27.0.1-arm64-arm-64bit。每场景额外预热 2 次、测量 10 次，保留默认 GC。下面是完整 generate 返回耗时的 P50，单位 ms；流式包含全部事件消费、响应关闭和 result 复制。

| 协议 | 聚合：优化前 → 后 | 降幅 | 流式完整消费：优化前 → 后 | 降幅 |
| --- | ---: | ---: | ---: | ---: |
| openai_chat | 41.28 → 24.00 | 41.9% | 41.50 → 24.04 | 42.1% |
| openai_responses | 87.53 → 65.03 | 25.7% | 87.31 → 64.67 | 25.9% |
| anthropic | 93.36 → 48.58 | 48.0% | 93.03 → 49.58 | 46.7% |
| gemini | 156.87 → 77.18 | 50.8% | 160.06 → 95.19 | 40.5% |
| ollama | 75.38 → 23.29 | 69.1% | 71.11 → 23.33 | 67.2% |

五个原生场景及 25 对跨协议组合，各自包含聚合/流式，共 60 个场景、优化前后各 600 个暖样本。60 个场景的 P50 全部下降，降幅 21.6%–69.1%，中位降幅 45.3%。

另外对每种原生负载各取一次暖调用启用 `cProfile`，`deepcopy` 的递归调用数减少
61.3%–82.8%；Gemini 从 543,626 次降至 154,069 次。该计数用于确认遍历减少，
不代表完整请求副本的份数，也不把 profiler 放大后的耗时当作真实延迟。

逐场景核对了优化前后的请求字节、响应字节、上游事件数、HTTP 分块数、历史数量、工具定义数和并行调用数，全部一致；负载规模、SDK 版本和采样设置也一致。每轮仍验证协议结构、图片字节、完整工具 Schema、历史调用参数、最终文本与工具参数以及响应关闭。

这是本机默认复杂负载的观测结果；原生场景各自包含不同的协议特性，不用于五种协议之间的绝对效率排名。默认 GC、操作系统调度和小样本会影响 P50/P95，尤其会使聚合与流式的独立样本出现抖动。未设置机器速度相关的测试门槛。

优化前后分别运行：

```sh
uv run --no-sync python -m tests.performance --iterations 10 --warmup 2 --json /tmp/my-llm-optimization-before.json
uv run --no-sync python -m tests.performance --iterations 10 --warmup 2 --json /tmp/my-llm-optimization-after.json
```

两份本机 JSON 保存全部原始样本、P95、CPU 时间、冷调用、Git 工作区状态及 SDK 版本。负载与其他命令见[性能测试指南](../guides/performance.md)。

## 验证

- 全量测试：1583 passed。
- Ruff 检查、格式检查与 git diff --check 通过。
- [所有权测试](../../tests/unit/test_copy_ownership.py)：五协议公开 encoder/decoder 的深层双向隔离、25 对转换、原生快照缺失、回放快照、重复请求及 SDK 配置模型/tuple 可变叶子。
- [Schema 计划测试](../../tests/unit/test_schema_filter_plans.py)：暖缓存不再反射类型、Union 同分与顺序、递归/Annotated、开放 JSON 保留、只读投影无复制、官方 SDK 验证不修改输入及请求数据可回收。
- [终态物化测试](../../tests/contract/test_stream_materialization.py)：工具 JSON 单次解析、缓存版本失效、旧块稳定、25 对协议及未知来源的公开结果/事件不污染缓存、失败部分结果。
- 既有并发、取消/提前退出、SDK 响应关闭、Profile/绑定、字段保留、第六协议注册和职责边界测试均通过。
