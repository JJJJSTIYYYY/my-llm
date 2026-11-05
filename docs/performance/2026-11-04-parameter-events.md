# Parameter 与 event_type 性能审查（2026-11-04）

> 历史报告：数据来自移除 Ollama / Gemini 协议前的版本，保留用于复核；当前支持范围与测试场景见[性能测试指南](../guides/performance.md)。

本次类型拆分和输出事件路由在 10 个原生场景的交替复测中，P50 变化中位数为 **+0.22%**，
范围 **-1.80%～+2.06%**。存在可简化的重复匹配和提前读取回退字段；本轮实验未证明它们能
明显缩短完整 generate 调用。运行时代码未修改，候选改动只应用于临时源码副本。

完整 60 场景的首轮对照为 **+1.48%**：原生 10 项 **+2.71%**，交叉 50 项 **+1.14%**。
先后两轮易受运行顺序影响，所以保留原始数据，再用交替运行判断差异。
首轮最大的 Ollama → Chat 聚合/流式回退为 +7.33% / +7.01%；按当前/旧/旧/当前复测后，
分别为 **-1.53% / +0.53%**，未复现首轮幅度。

## 测量范围

- 基线：改动前 HEAD `611c12cff9d6d28376d1ee8ed7ef70d5fe1f8386` 的 src；当前：开始审查时的未提交 src 快照。
- 五种上游 × 五种 stdio，聚合和流式；包含 10 个独立原生场景与 50 个交叉场景。
- 完整主测每项 15 个暖样本、3 次预热。原生复测按 **旧/当前/去重候选/去重候选/当前/旧**，
  每轮每项 21 个样本、5 次预热，每版本每项合并 42 个样本。
- 最大波动的 Ollama → Chat 单独按当前/旧/旧/当前复测，每轮每项 31 样本、5 次预热，
  每版本每项合并 62 个样本。
- 使用相同解释器、SDK、驱动、负载及 HTTP 分块；主测 60/60 请求字节、响应字节、负载指纹、
  HTTP 分块数量和公开输出事件数一致。交替复测也逐项检查一致。
- 真实 SDK 通过内存 HTTPX MockTransport 解析 SSE/NDJSON。包含完整 generate、消费、关闭和结果读取；
  不含网络、模型推理、夹具构造、实例构造及最终实例关闭。GC 保持启用，计时阶段不运行测试或其他基准。
- 默认规模：12 轮历史、12 个工具、4 路工具调用、2048 字符输入、8192 字符输出、96 字符增量、
  3 层工具 Schema、1024 字节 HTTP 分块。交叉夹具不包含 reasoning。

| 原生上游 | 模式 | 旧 P50 ms | 当前 P50 ms | 变化 |
| --- | --- | ---: | ---: | ---: |
| openai_chat | 聚合 | 23.635 | 23.210 | -1.80% |
| openai_chat | 流式 | 24.666 | 24.695 | +0.12% |
| openai_responses | 聚合 | 61.031 | 60.699 | -0.54% |
| openai_responses | 流式 | 63.092 | 63.291 | +0.32% |
| anthropic | 聚合 | 39.516 | 39.850 | +0.85% |
| anthropic | 流式 | 40.679 | 40.923 | +0.60% |
| gemini | 聚合 | 29.879 | 29.915 | +0.12% |
| gemini | 流式 | 31.902 | 31.912 | +0.03% |
| ollama | 聚合 | 14.667 | 14.748 | +0.55% |
| ollama | 流式 | 16.030 | 16.360 | +2.06% |

百分比为每个场景计算后再取中位数，不是把所有异构样本混合计算。
当前 GC 诊断的 Responses 原生/middleware 成对运行记录到 2 次二代回收，
暂停中位数 39.96 ms、最大 40.02 ms。
P95 的数十毫秒尖峰不能仅凭短样本解释为事件匹配回退。GC 回调仅用于独立诊断，
其耗时不与无回调主测混用。

## 可优化点与职责边界

**1. 消除已匹配分支中的第二次事件匹配，作为低风险简化。**

[Chat](../../src/my_llm/codec/openai/chat/encoding/encoder.py#L272)、
[Responses](../../src/my_llm/codec/openai/responses/encoding/encoder.py#L407)、
[Anthropic 自定义反馈](../../src/my_llm/codec/anthropic/encoding/encoder.py#L338) 和
[Gemini 自定义输出](https://github.com/JJJJSTIYYYY/my-llm/blob/0d13d6842904f1a970866e43e49d767e331ed6c0/src/my_llm/codec/gemini/encoding/encoder.py#L296) 已先调用
`parameter_matches_event()`，之后又给 `parameter_value()` 传 `event=True`，重复查询绑定及匹配类型。
候选只移除这四处重复标记。仍由 BaseEncoder 提供通用匹配，协议 Encoder 决定原生事件条件和提取时机。
未先匹配的 Ollama 与 Gemini 自定义反馈仍保留 `event=True`。

10 个原生场景相对当前版本的完整调用 P50 变化中位数为 **+0.02%**，范围 -1.28%～+1.91%；
没有测出整体提速。Chat 原生样本每请求有 116 个事件，却调用事件匹配 923 次，存在重复工作，
但调用频率不代表它占主要耗时。

**2. 在匹配器内部按需读取 Chat 的 object 回退字段。**

[BaseEncoder.parameter_matches_event()](../../src/my_llm/codec/base/encoder.py#L194) 中
`read(event, "type", read(event, "object", ""))` 会先计算第三个参数，即使已有 type 仍读取 object。
候选先用缺失标记读取 type，仅在字段缺失时读取 object。保持 type 显式为 None 时不回退的现有语义，
不在 Transport 中增加协议判断，也不把事件选择放入 StreamState。
这是仅作用于显式 event_type 的小优化；默认空字符串分支不执行这些读取。

下面是显式 event_type 的独立 Encoder 测量：每样本 500 次单事件调用，匹配/非匹配交替，
5 次预热、31 个样本，当前/去重/惰性回退/惰性回退/去重/当前顺序，每版本合并 62 样本。
五协议的完整中间输出指纹均一致。该测量不包含 SDK、请求准备或 StreamState。

| 协议 | 当前每 500 事件 P50 ms | 去重候选变化 | 惰性回退候选变化 |
| --- | ---: | ---: | ---: |
| openai_chat | 1.674 | -2.17% | -2.14% |
| openai_responses | 1.280 | -2.94% | -3.03% |
| anthropic | 1.086 | +0.47% | -1.61% |
| gemini | 1.741 | -3.63% | -1.76% |
| ollama | 1.710 | -1.80% | -3.10% |

变化只有数十微秒。去重候选的 Ollama 源码未改变，仍出现 -1.80% 波动，表明这一级差异受噪声影响；
不能把约 2% 的局部比例解释为完整调用提速。惰性回退候选没有单独进行完整 generate 对照。

**3. Profile 字段名缓存暂不建议增加。**

[set_attr()](../../src/my_llm/profile/types.py#L165) 每次调用 `fields(self)`。
在临时副本中使用按 Profile 类型、上限 128 的 LRU 缓存字段名，缓存对象仅为声明元数据。
通过当前/候选/候选/当前测量，每项 100 次预热、15 批 × 每批 40 次操作，每版本合并 30 批。
真实 `profiles(protocol)` 声明构造的变化中位数 **+0.15%**，各协议 -1.35%～+0.22%；
该工厂包含协议默认值准备。MyLLM 构造变化中位数 **-0.61%**。收益不足以支持增加缓存维护逻辑，
也未测量大量动态 Profile 子类的额外类型保留成本。

## 热点、并发与验收

每种协议原生流式预热后用 cProfile 记录 25 次完整调用，断言在 profiler 关闭后执行。
下表是诊断累计时间占比，包含 profiler 开销，不能当作无追踪的绝对执行时间。

| 协议 | 每请求原生事件数 | 每请求匹配调用数 | 匹配累计时间占比 |
| --- | ---: | ---: | ---: |
| openai_chat | 116 | 923 | 0.44% |
| openai_responses | 244 | 840 | 0.15% |
| anthropic | 236 | 472 | 0.13% |
| gemini | 110 | 220 | 0.09% |
| ollama | 110 | 330 | 0.28% |

`capture()`、SDK 模型访问、协议投影和状态处理比匹配器更重。
现有复制承担输入/输出/状态间的所有权隔离；本轮未证明存在可删除的多余复制。
后续如审查复制，应先定位一次调用的数据所有者与逃逸边界，再验证早先批次稳定性、
原始对象修改隔离、语义损失 hook 和最终快照一致性。

当前与去重候选分别对全部 10 个原生场景执行独立内存、缓存及四路并发探针：
每项通过响应首字节屏障证实 peak_active=4，验证每份响应关闭和请求头隔离。
两份报告暖请求的规则缓存 miss/条目数均未增长；逐场景串行及并发 tracemalloc 峰值差异均小于 0.02%。
并发批次 P50 变化中位数为约 +1.00%，每项仅 5 批，不据此宣称并发提速或回退。
内存探针与计时分开，tracemalloc 数值不是 RSS。

三个候选分别通过 **180 项相关测试**，覆盖方向与身份、继承、类型校验、自定义事件、
默认值、非匹配事件、SDK SSE 解析和公开输出。运行时源码指纹在完成审查后再次核对，保持未变。
本轮只新增本报告、原始数据及文档索引。

## 原始数据与复测

[原始样本、环境、诊断、候选补丁及临时驱动](2026-11-04-parameter-events.json)
保留完整 60 场景主测、原生交替复测、显式事件 Encoder 测量、构造、内存、缓存、并发和测试结果。
运行时代码与测量 harness 分别按有序相对路径、NUL、文件字节计算 SHA256：

- 基线：`51a1f88feeed6fa8cf43e4e3ecacffa2e735f5c7b1694427bd77752f94098362`。
- 当前：`60f8771fa0d3a430515e00b2af79e07c6cae0de58212361117232a2fd61ec452`。
- 去重候选：`36a35a2bfaeb23c6b777ddb162505db76862e8c6be433b1a6c3f28ca01dee816`。

完整测量入口：

```sh
.venv/bin/python -m tests.performance --iterations 15 --warmup 3 --json /tmp/parameter-current.json
.venv/bin/python -m tests.performance.profile_setup --batches 15 --iterations 40 --json /tmp/parameter-setup.json
.venv/bin/python -m tests.performance.gc_diagnostic /tmp/parameter-gc.json --protocol openai_responses --iterations 41 --warmup 5
```

比较版本时用 PYTHONPATH 选择独立 src 快照、复用同一 harness，按原始数据中的交替顺序运行。
保存的驱动源码可还原局部实验；更完整的计时边界见[性能指南](../guides/performance.md)。
