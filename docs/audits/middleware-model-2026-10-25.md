# Middleware 模型整理核对（2026-10-25）

本报告重新核对原 M-01–M-06 问题。判断依据为当前实现、实际转换结果及对应回归用例；
重构不保留旧字段别名。原审计把部分实现取舍当成缺陷，以下分别说明。

| 项目 | 核对结论 | 已采用的处理 | 证据 |
| --- | --- | --- | --- |
| M-01 项身份字段 | 字段拼写分裂属实；“五种 id 都是同一身份”的说法不准确，工具调用、源输出项和流地址必须区分 | 消息及所有内容块统一可选 `item: ItemIdentity`，仅含 `id/status`；工具继续使用独立 `call_id` | [模型定义](../../src/my_llm/middleware/models.py)、[字段往返测试](../../tests/unit/test_middleware_models.py) |
| M-02 公开内部 DTO | 原 `AssistantMessageChunk` 混合状态操作和线格式回写，公开暴露没有应用层用途；回写字段原本有用途，并非死代码 | 删除 DTO 及导出；`StreamBatch.events` 直接保存已由 encoder 拥有的操作字典；应用流仍使用 `MessageEvent` | [流模型](../../src/my_llm/middleware/stream.py)、[包导出](../../src/my_llm/__init__.py)、[公开事件与补丁测试](../../tests/unit/test_middleware_models.py) |
| M-03 middleware 解释原生字段 | 属实；检查协议名称字符串不足以证明边界正确 | middleware 不再解释原生隐去思维、摘要对象或协议专有元数据；原生结构在各 codec 内转换，操作元数据按 `path/values` 原样搬运 | [原生 encoder 边界](../../src/my_llm/codec/base/encoder.py)、[原生字段边界检查](../../tests/contract/test_codec_boundaries.py) |
| M-04 响应时间双键 | 属实，且导致 Responses → Chat 时间丢失，并非单纯命名问题 | 中间元数据统一 `created_at`；Chat encoder/decoder 负责原生 `created` 映射 | [Chat encoder](../../src/my_llm/codec/openai/chat/encoding/encoder.py)、[Chat decoder](../../src/my_llm/codec/openai/chat/decoding/decoder.py)、[Responses decoder](../../src/my_llm/codec/openai/responses/decoding/decoder.py) |
| M-05 模型与 wire 双表示 | 双表示本身是接口取舍；真实问题是手写映射遗漏字段、未知内部类型静默消失，以及不必要的流 DTO 往返 | 保留明确的 dataclass 与显式投影；补齐全部模型字段往返测试；内部未知内容、操作和增量字段报错；流操作不再拆解还原 | [完整字段测试](../../tests/unit/test_middleware_models.py)、[中间协议契约](../design/protocol-middleware.md#模型投影与内部流操作) |
| M-06 字段归属文档 | 属实；“不是原生字段并集”不足以解释同协议回放字段 | 字段表逐一列出全部模型字段，标明通用、协议回放或运行时定位用途，说明可选身份与投影所有权 | [完整字段表](../design/protocol-middleware.md#支持集合)、[直接输入输出指南](../guides/middleware-io.md)、[API 参考](../reference/api.md) |

## 已核实的实际数据损失

原实现中，Responses `function_call_output` 的 `id/status` 由 encoder 写入临时块，
随后 `ToolResult` 解析器没有接收，导致同协议转换也丢失源项身份。
当前 `ToolResult.item` 与其他内容统一保存这两个值；工具关联仍使用 `call_id`，
不能用源项 id 替代。

原 Responses 元数据写入 `created_at=100`，Chat decoder 却只读取 `created`，
输出为 `created=0`。当前两者都读取统一中间时间，仅在 Chat 原生出口改键名。

原 `ReasoningContent(redacted=True)` 投影分支只保留密文，
即使模型同时具有 `reasoning/signature/summary`，往返后也会静默丢失。
当前所有思维均使用同一中间结构，`redacted` 为独立布尔标记，摘要为 `list[str]`，
完整字段往返不再受原生块形状影响。各 codec 决定其原生表达与损失报告。

继续检查调用链还发现两处状态损失：Responses 摘要增量首次发布时会被暂时标为 `raw`，
以及缺少基数的 usage 更新会覆盖先前已知计数。当前 encoder 在发布前传入摘要标记，
usage 更新仅覆盖实际提供的计数分组；原生缓存写入计数继续由内部状态保留供目标 codec 输出。
公共 `Usage` 的五个字段保持明确，未将所有原生计数扩展成公共模型字段。

## 保留的取舍与边界

`item.id` 标识源项，`call_id` 关联调用和结果，`source_ref` 定位流块，
`response_id` 标识整个响应。这些信息具有不同作用域，保留它们有实际用途。
消息级身份与内容级身份也不同：多个源项合为一个 assistant 回合时，内容仍需保留各自源项身份。

公共 dataclass 让应用直接使用具体字段；显式 wire 投影供转换管线组装字典。
这里没有采用通用反射序列化：`Origin.value` 引用原始输入，不能被递归遍历或复制到 payload。
`origin` 不作为 wire 键输出，类型别名 `JsonObject` 也不会增加运行时包装。
已有中间键名与工具 `functions` 分组继续构成明确契约；不为这些投影新增泛化序列化层。

`StreamBatch` 是 encoder 与状态机之间的操作载体；元数据是由原生 encoder 筛选后的内部数据。
middleware 不解释任意协议专有键，也不因此承诺透传用户原生 metadata。
`to_wire()` 借用内部操作及部分嵌套应用值，公开生成结果、公开事件和最终 SDK 请求仍按各自边界隔离。

未知原生字段继续由源 encoder 静默过滤。未知中间内容类型、操作种类或增量字段，
则抛出 `MessageValidationError`，避免已进入支持契约的数据悄然消失。
工具结果的中间子块必须为文本或图片；已知上游产出的必需字段直接读取，
默认值由模型构造及原生 encoder 负责，不在内部解析器重复兜底。

## 验证入口

```bash
uv run pytest -q tests/unit/test_middleware_models.py tests/unit/test_semantic_loss.py tests/contract/test_codec_boundaries.py
uv run pytest -q tests/contract tests/integration/test_middleware_io.py tests/integration/test_documentation.py
uv run ruff check src tests examples
```

字段用例显式声明全部 dataclass 字段；新增字段未纳入完整往返用例时测试会失败。
流用例覆盖每种操作、部分工具补丁和未知类型；边界检查同时禁止协议名分支和已知原生字段耦合。
性能结果需要同解释器、依赖和负载下的前后比较，不能从模型简化或单次耗时推断。

## 本轮验证与性能对照

模型整理阶段全量测试 **2611 passed**；Ruff 检查、171 个 Python 文件的格式检查、
本地文档链接和公开导出检查通过。原生与中间 stdio 均覆盖五协议、工具 EOF 交付、
取消和提前关闭、并发、签名回放、所有权以及三种语义损失策略。
工具参数可变负载在 native encoder 的流出口捕获一次；对应计数用例验证没有新增整批深拷贝。

使用未改动的 `tests.performance.middleware`：五种上游 × 收集/流式，共十组，
每组原生与 middleware stdio 按 N M M N 顺序交替，预热 2 次、计时 9 次；
真实官方 SDK 经本地 HTTP 解析相同字节，无外网请求。
计时包含请求准备、完整消费、结果读取和响应关闭，保留默认 GC；
tracemalloc、缓存快照及四路真实重叠并发单独测量，不混入串行计时。
前后十组的请求字节、响应字节摘要全部一致。

环境：Python 3.12.12，macOS 27.0.1 arm64；openai 2.54.0、anthropic 0.125.0、
google-genai 1.75.0、ollama 0.6.3、httpx 0.28.1、pydantic 2.14.1。
负载包含 12 轮历史、12 个工具、4 个并行调用、2048 字符输入、8192 字符输出、
96 字符增量、8 项参数数组和 3 层 Schema。

以下为完整调用耗时中位数，单位 ms；负百分比表示耗时下降。

| 上游 | 模式 | middleware 整理前 | middleware 整理后 | middleware 变化 | 原生 stdio 变化 |
| --- | --- | ---: | ---: | ---: | ---: |
| openai_chat | 收集 | 20.067 | 18.532 | -7.65% | -7.50% |
| openai_chat | 流式 | 21.347 | 19.542 | -8.46% | -6.03% |
| openai_responses | 收集 | 57.408 | 53.833 | -6.23% | -5.10% |
| openai_responses | 流式 | 59.241 | 55.734 | -5.92% | -6.46% |
| anthropic | 收集 | 34.093 | 33.176 | -2.69% | -1.63% |
| anthropic | 流式 | 35.862 | 34.044 | -5.07% | -3.12% |
| gemini | 收集 | 23.118 | 22.086 | -4.47% | -4.62% |
| gemini | 流式 | 24.737 | 23.587 | -4.65% | -3.89% |
| ollama | 收集 | 11.742 | 10.828 | -7.79% | -6.21% |
| ollama | 流式 | 12.798 | 11.667 | -8.84% | -6.33% |

middleware 中位数在本轮十组测量中下降 **2.69%–8.84%**，原生 stdio 下降 **1.63%–7.50%**。
串行 tracemalloc 峰值基本持平：最大绝对增量是 Responses 的 **13,160 字节，约 0.19%**，
本轮同时增加了嵌套项身份与标记，不能声称所有场景内存减少。
五类 Schema LRU 在暖请求探针中均没有增加条目，仍为有界类型规则缓存。
所有场景实际重叠 4 路；携带不同请求头时 Ollama 使用 4 个客户端，其余协议复用 1 个，
前后行为一致，响应及客户端关闭断言通过。

9 个样本不足以判断稳定尾延迟。尤其 Chat 收集的 middleware P95 从 **20.226 ms**
升到 **52.203 ms**，Gemini 收集从 **59.974 ms** 升到 **60.298 ms**；
不将中位数改善解释为所有请求或尾延迟都改善，也不外推到真实网络吞吐。

基线为本轮开始前 `719bf40` 的运行时源码，最终源码与基准均记录摘要：

- 整理前源码 SHA256：`dafab5c1c70f25b50bb5ff075ff2a23ffba8138737d25a261a81d885665aaea7`
- 整理后源码 SHA256：`a70054228e550278efc5db722c69ad391d0496acd52152aa16029bb3963bd7e1`
- 相同 harness SHA256：`a3003cf8a5c751799682fcc64003b3fe5a3013453369b30ed9675832f761a4a7`
- 相同 driver SHA256：`06c17872609c22d3b1c7148a066f84b1457967ecef4e9087943f1ab47edc2ac8`

复测命令；先后分别选择对应源码，保持同一环境和测试脚本：

```bash
.venv/bin/python -m tests.performance.middleware --json /tmp/my-llm-middleware-results.json --iterations 9 --warmup 2 --concurrency 4
```

本报告只保留汇总，原始样本在本机临时目录，不向 `docs/audits` 增加 JSON 日志。

## 性能回退定位与局部简化

### 原回退的归因

前一节每组只有 9 个样本，采用 nearest-rank 的 P95 等于最大值；
一个第 2 代 GC 停顿就足以让 P95 从约 20 ms 跳到约 52 ms。
本次对 `719bf40` 与模型整理后的源码各运行两个独立进程，顺序为 A B B A，
每进程同一 stdio 预热 10 次、采样 100 次；进程内原生和 middleware 按 N M M N 交替。
Chat、Gemini 分别得到每版本 200 个 middleware 收集样本。
GC 保持开启，阈值为 `(700, 10, 10)`，单独记录代数、停顿与进程 CPU 时间。
这些带 GC 回调的数字仅用于归因，不替换未加观察器的主计时。

| middleware 收集场景 | 整理前 P50 / P95 ms | 模型整理后 P50 / P95 ms | 第 2 代 GC 次数（前 / 后，每侧 200 样本） |
| --- | ---: | ---: | ---: |
| Chat | 19.899 / 20.498 | 18.152 / 18.893 | 1 / 2 |
| Gemini | 22.912 / 54.377 | 21.654 / 52.817 | 26 / 26 |

Chat 整理后最慢样本为 **56.866 ms**，CPU 为 **56.852 ms**，其中 GC 为 **38.669 ms**；
Gemini 对应为 **56.573 / 56.569 / 34.910 ms**。
这些长停顿与第 2 代 GC 重合，未复现先前九样本报告所暗示的持续 P95 回退。
两版使用相同请求、响应、SDK 和基准；模型整理后普通请求的中位数仍更低。
本机进程级 GC 扫描是这些尾部样本的主要成本，不能归为网络等待或缓存重新编译。

五类 Schema LRU 的 `misses`、`currsize` 在所有暖请求诊断中均保持不变；
每个 stdio 实例的 SDK 客户端在多次调用中复用，未发现客户端重建或缓存淘汰。
状态物化缓存的替换/增量失效和工具 JSON 单次解析也由
[物化测试](../../tests/contract/test_stream_materialization.py)继续验证。
未发现能解释这些几十毫秒停顿的兜底分支；可选输入和不支持原生块的处理仍属于必要边界。

### 已采用的局部简化

- `StreamState` 的已规范化数据改用现有 `capture()`，复用不可变叶子并复制可变容器，
  自定义对象仍走其深拷贝语义。保留状态、输入、公开事件和结果间的隔离，未改成共享调用者容器。
  同一复杂 Chat 负载、20 次调用的 cProfile 中，通用 `deepcopy` 函数调用次数（含递归）
  从 **33,760** 降为 **17,180**；这是减少通用复制分派，不能等同于端到端提速比例。
- 状态机未发布事件时，流入口直接继续读上游，避免对空批次创建目标转换上下文和调用 decoder。
  EOF 工具集中交付、usage、完整结果和资源关闭规则保持不变。

复制仍有必要：原始操作可被调用者修改，状态需拥有独立值；公开结果也不能反向修改状态缓存。
本次没有用省略这些隔离复制来换取性能。

### 不加 profiler 的前后对照

以下只比较本节两项局部简化之前与之后，和上文“重构前 vs 模型整理后”属于不同对照。
五协议 × 收集/流式共十组，每进程每组预热 5 次、采样 25 次，进程顺序 A B B A；
合并后每版本每组每 stdio **50 个样本**。计时沿用 `middleware.paired()` 和 `runner.measure()`，
不启用 profiler、tracemalloc 或 GC 回调；内存/并发探针不穿插其间，计时期间未运行单测。
各组请求字节、响应字节、负载摘要和公开事件数均相同，harness 与 driver 摘要一致。

| 上游 | 模式 | middleware 简化前 ms | middleware 简化后 ms | middleware 中位数变化 | 原生 stdio 中位数变化 |
| --- | --- | ---: | ---: | ---: | ---: |
| openai_chat | 收集 | 18.277 | 18.271 | -0.03% | +0.26% |
| openai_chat | 流式 | 19.473 | 19.455 | -0.09% | -0.45% |
| openai_responses | 收集 | 54.114 | 54.238 | +0.23% | -0.18% |
| openai_responses | 流式 | 55.736 | 55.359 | -0.68% | -1.47% |
| anthropic | 收集 | 32.914 | 33.151 | +0.72% | +0.56% |
| anthropic | 流式 | 34.152 | 34.173 | +0.06% | -1.29% |
| gemini | 收集 | 21.793 | 21.809 | +0.08% | +0.20% |
| gemini | 流式 | 23.269 | 23.278 | +0.04% | -0.05% |
| ollama | 收集 | 10.982 | 11.012 | +0.27% | -0.31% |
| ollama | 流式 | 11.789 | 11.628 | -1.37% | -1.34% |

大多数变化不足 1%，不足以宣称稳定的整体提速；收益主要是移除可确认的局部工作。
此次主计时仍观察到 Responses middleware 收集 P95 从 **54.993 ms** 跳到 **87.798 ms**：
50 点中超出中位数 20 ms 以上的慢样本从 **2 个**变为 **3 个**，刚好跨过 P95 的 5% 位置。
不能只根据这两个 P95 值判定持续回退。

随后针对该路径独立做 A B B A 的 GC 观察，每版本 200 点：
简化前 P50/P95 为 **53.728 / 54.426 ms**，简化后为 **53.553 / 54.328 ms**；
两侧均有 **2 次**第 2 代 GC，慢样本中的 GC 停顿约 **31–33 ms**，暖缓存无新增 miss。
该诊断未复现持续变慢，也不意味着所有运行的尾延迟都保证下降。

局部简化前源码 SHA256：`a70054228e550278efc5db722c69ad391d0496acd52152aa16029bb3963bd7e1`；
简化后：`fa35d852d2dcaaeb0fae95139f2744ecdcc28675aef571a9afec39aca7f5f1e6`。
全量 **2624 passed**，Ruff 和格式检查通过。新增验证覆盖五协议两种 stdio 的空批次跳过，
以及诊断器的预热排除、GC 回调和计时函数在成功/异常路径上的恢复。

可复用的 GC 观察入口现支持全部五协议与 native/middleware stdio；例如：

```bash
.venv/bin/python -m tests.performance.gc_diagnostic /tmp/chat-gc.json --protocol openai_chat --stdio middleware --iterations 100 --warmup 10
.venv/bin/python -m tests.performance.gc_diagnostic /tmp/gemini-gc.json --protocol gemini --stdio middleware --iterations 100 --warmup 10
.venv/bin/python -m tests.performance.gc_diagnostic /tmp/responses-gc.json --protocol openai_responses --stdio middleware --iterations 100 --warmup 10
```

在相同解释器和 driver 下用 `PYTHONPATH` 选择保存的两版源码，按 A B B A 运行，
核对摘要后比较采样分布。全部 JSON 与 profiler 原始数据仍只写本机临时目录。
