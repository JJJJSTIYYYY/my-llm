# Schema 缓存检查与优化试验

> 本文记录当时源码快照的测量结果。原始 JSON 日志已清理，保留结果汇总、测量条件与复测方法；不能再据本文逐样本复算，也不将历史结果视为当前工作区实测。

日期：2026-10-24。基线为[需求验收完成](requirements-acceptance-2026-10-24.md)的源码，
本轮重新测量两版，没有使用历史时延样本代替 before。

## 检查结论

**本轮没有新增缓存，也没有保留实验代码。当前运行时与需求验收完成时逐字节一致。**
两种试验分别存在整体收益不足、临时内存明显增加的问题，因此保持现有实现。

现有缓存已经覆盖类型分析、过滤计划、验证器、SDK 参数声明和路径分析。
暖调用没有缓存容量不足的证据，因此保留原有容量和所有权边界。

before 的 60 场景暖内存探针中，五类缓存的 miss 增量均为 0。
整个矩阵的缓存项数最高为 fields 379、adapter 13、parameter_types 5、plan 34、declared_path 21，
都小于各自的 512 项上限。fields 与 parameter_types 的分析也没有在暖请求中重新执行。

第一种候选试验修改 [schema.py](../../src/my_llm/cache/compiled_cache.py) 中的 Union 过滤计划
（该实现现已迁至 cache/compiled_cache.py）：
类型编译时记录结构候选的 Literal 约束是否全部有路由索引；执行时若已完整检查这些约束，
且只剩一个结构候选，直接执行其过滤函数。省去对唯一候选的字段计数评分及候选列表构造。
重叠候选、缺少判别字段时的同分顺序、标量候选、递归类型和非法判别值仍受原规则约束。

局部微测中，Chat 消息过滤从约 0.133 ms 降到 0.094 ms，节省约 0.04 ms；
但完整矩阵和原生交替复测没有证明值得保留的整体收益，候选代码与为其增加的测试均已撤回。
原有协议边界、缓存键、容量、复制隔离和 SDK 执行路径全部保留。

| 状态 | 缓存内容与生命周期 |
| --- | --- |
| Schema 规则 | fields、adapter、plan、parameter_types、declared_path；有界 LRU，每类最多 512 项，只保存类型与规则 |
| Profile 类型分析 | 声明与构造器分析，最多 128 项；参数路径解析最多 512 项 |
| SDK 字段视图 | 当前转换中按需读取的字段，不跨请求保存 |
| 响应物化 | 当前 StreamState / BlockState 内部，更新时失效 |
| SDK 客户端 | 当前 MyLLM 实例内复用，由 aclose 释放；不放入全局 Schema 缓存 |

Profiling 中 `_plan` / `_SchemaKey` 查找成本很小，较多成本来自官方 SDK 的类型转换和实际数据处理。
增加 LRU 容量无法省去这些逐请求工作；缓存编码后的消息、工具参数或响应会破坏所有权与请求隔离。

## 未采用的批量校验

另在独立临时源码中测试 `TypeAdapter(list[schema])` 一次校验、序列化整批值。
它同时持有整批解析结果和整批 JSON 字节；当前实现逐项校验并强制检查 SDK 的懒 Iterable。

| 原生消息校验 | 原逐项峰值 | 整批峰值 |
| --- | ---: | ---: |
| Chat | 35,487 B | 882,588 B |
| Responses | 143,672 B | 951,662 B |
| Anthropic | 201,900 B | 977,353 B |
| Gemini | 77,670 B | 1,063,740 B |
| Ollama | 53,492 B | 952,738 B |

部分局部校验耗时下降，但这些临时分配不值得保留，因此没有将批量校验写入运行时。
上表保留当时的诊断汇总。局部微测不能代表完整请求加速。

## 验证

路由候选阶段使用 8 项临时反例，检查唯一 role 不能盖过另一个非法标签，并检查带结构分支的 Union 仍保留标量候选；
完整测试 **2441 passed in 18.28s**。撤回候选代码与临时测试后，再验证保留的原实现。
已有测试覆盖 SDK 对象、同分顺序、动态类型淘汰、未知字段不读取、委派路径隔离和请求 payload 可回收。

最终保留实现 **2433 passed in 18.48s**。Ruff、165 个 Python 文件格式、git diff --check、
166 个本地 Markdown 链接、两个离线示例、wheel / sdist 构建及 Twine 严格检查均通过。

before 源码 SHA256：`4fae8edb3daf0362d61998411fb39889619b2f457dbdb2425613a0fba2eeb3aa`。
候选 after 源码 SHA256：`9c9104246009fdf161e5158e98b09015b1d884496885d1a5ad2d386ef8d64165`，**未采用**。
最终保留源码与 before 相同。
harness SHA256：`f8d5ab40e9184751af15e49c0acd0c08528a43cbf503706b34cfd5516a584ff0`。

## 完整性能对照

两版使用相同解释器、官方 SDK、默认复杂负载和 1024 字节 HTTP 分块。
各运行 60 个场景，每场景串行 2 次预热 / 9 次采样，四路并发 2 批预热 / 9 批采样，
内存和缓存另行探测。计时保留 GC，不包含网络和模型推理。
原生协议再按 before / after / after / before 顺序交替运行，每版每场景共 62 个暖样本。

比较器已核对两版负载、响应字节、分块、环境和测量脚本指纹。
60/60 场景请求 SHA256 一致，公开事件数量一致；两版暖缓存 miss / 项数增量均为 0。
四路并发各场景均达到 peak_active=4，头部隔离、完整结果和响应关闭检查通过。

下表先计算各场景的百分比变化，再取组内中位数；负数表示耗时降低。

| 候选相对原实现 | 串行 P50 | 串行 P95 | 四路并发批次 P50 |
| --- | ---: | ---: | ---: |
| 全部 60 场景 | +0.40% | +0.31% | +0.23% |
| 原生 10 场景 | +0.21% | −0.90% | +0.07% |
| 交叉 50 场景 | +0.53% | +0.67% | +0.25% |
| 聚合 30 场景 | +0.66% | +0.77% | +0.29% |
| 流式 30 场景 | +0.22% | +0.08% | +0.16% |

暖内存峰值的组内变化中位数近乎 0；这不包含预热前已编译的类型规则。
原生 ABBA 每版每场景共 62 个样本，场景 P50 变化的中位数为 **−0.15%**，单场景为 −0.85% 至 +0.53%。
顺序矩阵与交替复测未给出一致、实际可观的整体加速。候选额外分支不值得为了局部微测保留。
这些结果只覆盖本地官方 SDK 调用链，不包含真实网络或模型推理，也不代表任意工作负载的统计保证。

## 复测入口

修改源码前分别保留 src 快照，用同一工作区的测试脚本运行：

```sh
PYTHONPATH=/tmp/before/src python -m tests.performance.request_chain --iterations 9 --concurrency 4 --json /tmp/before.json
PYTHONPATH=/tmp/after/src python -m tests.performance.request_chain --iterations 9 --concurrency 4 --json /tmp/after.json
python -m tests.performance.compare /tmp/before.json /tmp/after.json --json /tmp/comparison.json
```

原生交替复测使用 `python -m tests.performance --suite native --iterations 31`，
每次用 PYTHONPATH 选择相应源码快照，保存独立 JSON；比较器的 `--abba-before` / `--abba-after`
各接收对应的两份结果文件。
