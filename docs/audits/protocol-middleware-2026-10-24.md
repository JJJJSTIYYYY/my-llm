# 中间协议重构验收与性能对照

日期：2026-10-24。需求来源为根目录 `my-llm-refactor-requirements.md`。
重构前版本 `9a61f5183cd3a424dfb6cfe2eb5754524333c02e`；已逐字节核对独立基线快照的 84 个源码文件与该提交一致。
本次变更保留在工作区，需求原文件没有修改。

## 需求选择与交付

先定义 [ProtocolMiddleWare](../design/protocol-middleware.md)，再接入两条翻译链路。
`strict` 按用户明确选择：Chat、Responses、Anthropic 保留，Gemini、Ollama 静默丢弃。
需求 5.4 的“空函数”适用于显式选择 hook 且未提供函数；总体默认策略仍是 P0 / 5.2 规定的 raise。

| 需求 | 实现及直接验证 |
| --- | --- |
| C1 固定支持集合 | `middleware.py` 的具体模型；`test_semantic_loss.py` 检查根与子模型、开放 JSON Schema 所有权；`test_native_refactor.py` 覆盖五协议支持字段与未支持字段投影 |
| C2 工厂与实例 | `registry.py` 只保存工厂；MyCodec 保存工厂快照、按需创建本实例组件；`test_runtime_refactor.py` / `test_protocol_registration.py` 检查组件和可变 defaults 不共享、拒绝 singleton 工厂、二次设置隔离 |
| C3 丢弃与报错 | 原生 encoder 只投影固定支持字段，再校验支持结构；decoder 使用 report_loss；工具图片不再变 JSON 文本，外来思维不再混正文 |
| C4 完整 draft / 定位 | 调用独立 ContextVar 收集器；消息与工具全部译完再抛；测试检查多损失、完整尾部、原始 dict / SDK 对象身份及 Responses 终态流位置，不复制或遍历未知来源字段 |
| C5 三策略与 hook | BaseCodec 管理 BaseSemanticLossAdaptor；默认 raise、无记录 ignore、按方向 hook；测试空 hook、原地搬图片、返回值忽略、用户抛错、重入及最终 MyLLM 结果对象身份 |
| C6 思维链 | 25 源 / 目标组合验证 unsigned reasoning；同协议 opaque / 签名回放；Anthropic / Gemini 不伪造签名；跨签名按损失处理，正文与思维分别核对 |
| C7 工具图片 | Anthropic / Responses 两个来源 × 五目标；原生保留或逐图片收集损失；文本保留，原路径准确；50 个源 / 目标 / 对象与数组组合验证完整 SDK 准备；hook 示例由应用搬移图片 |
| C8 工具 schema | name / description / parameters 原样翻译；Ollama SDK 会丢约束时聚合 SemanticLossError；strict 三保留、两忽略；Schema 内应用字段不误删 |
| C9 未支持字段 | 五协议 × 五目标检查 audio / file / metadata / 引用等删除；畸形未支持内容不先被 SDK 校验拒绝；UnreadableExtension 验证未支持字段不被读取 / 深拷贝 |
| C10 响应链路 | MyLLM._render / MyCodec.result 均带 output 上下文；响应默认 raise、调用 output hook；Google SDK 签名损失释放当前响应且保留可复用客户端 |
| 5.6 工具与 usage | 工具在成功 EOF 后完整输出；Chat / Ollama 并行调用合入一个列表，Anthropic / Responses 不发送参数 delta；usage 仅内部收集，get_last_usage 返回副本，缺失字段 None，非流式结果不含 usage |
| 5.8 生命周期与并发 | 四路性能探针使用首字节 Barrier，断言真实 peak_active=4、请求头隔离和每份响应关闭；集成测试另覆盖八路交错、失败、取消、提前退出和最后完成 usage |

完整使用与迁移见[指南](../guides/semantic-loss.md)。公共编排没有增加具体协议名分支；
目标的表达、签名准入、Schema 限制仍归原生 decoder。

## 缓存与所有权

- 共享缓存保留类型 / 过滤 / 校验 / 路径规则的有界 LRU：Schema 相关五类 maxsize=512，Profile 类型声明 maxsize=128，路径 maxsize=512。
- 不缓存消息、图片、工具参数、Profile 实例、响应、损失集合或 hook。损失收集器在同步翻译结束后 reset，并发任务不共享集合。
- codec、组件、可变 defaults 与 adaptor 属于 MyLLM；未使用协议不实例化。注册快照不受后续注册影响。
- SDK 客户端保留实例内复用与显式关闭；不通过 LRU 淘汰活跃客户端。Ollama 请求 headers 绑定客户端，因此四路不同 headers 使用四个客户端，其余协议一个。
- SDK 开放 JSON 叶复用已有 capture 路径，保留别名、循环与自定义复制语义，避免逐原子调用通用 deepcopy；不增加 payload 缓存。
- 中间模型捕获应用 payload；来源定位只保留只读原始引用，既不当作恢复快照，也不遍历未支持字段。原生 draft 与输入独立，hook 修改的同一个对象进入后续链路。
- MessageStream.result 交付本调用最终 draft 本身，满足 hook 身份约定；重复读取不再额外深拷贝。get_last_usage 仍每次返回独立副本。

## 测量条件

两份隔离源码快照使用相同解释器、SDK、基准代码、固定 Schema 和负载。请求 / 响应夹具独立编写，
逐次验证最终文本、独立思维、工具名 / 完整参数、48 条工具结果、图片字节和响应关闭。
native 保留同协议 signed reasoning；cross 使用双方共同支持的文本、图片、工具集合，不含跨协议思维。
原始旧负载包含现已排除的文件等字段，其单独基线仅留作历史参考，不混入本次对照。

60 场景分别覆盖五条原生与 25 条跨协议路径的 stream / collect；每场景串行 cold、2 次预热、9 次计时，
再测四路 2 批预热和 9 批计时。默认 GC；计时不启用 tracemalloc。单独内存探针在 gc.collect 后测 Python
分配峰值，不是 RSS。构造计时单独测 500 个实例，内存探针保留 100 个实例。
原生延迟再以 A-B-B-A 顺序、每轮每场景 31 个样本复测。P95 是 nearest-rank 样本统计，不是置信区间。

比较器要求源码以外的 harness 指纹、全部夹具指纹、响应字节及 HTTP 分块完全一致；请求规范化与公开事件数变化单独列出。
usage 删除和完整工具事件减少是本次行为变化，不能据此声称两版事件数完全一致。
这里测的是本地无网络官方 SDK 处理成本，不包含模型推理或真实网络吞吐。

## 最终测量

最终源码 SHA256：`13ffa2cc013e7acb99d25e49638e9895d6de6cf7cb0597b95698c54574e2b3dd`；基线源码 SHA256：`909e6b072bcf6eedcc7717a6a97aa5035ec711402965861234e72184cc90cf84`。
测量 harness SHA256：`f8d5ab40e9184751af15e49c0acd0c08528a43cbf503706b34cfd5516a584ff0`。两版解释器、SDK、负载、响应 SHA256 与 HTTP 分块检查均通过，
该源码指纹对应首次重构的独立快照。后续代码清理及复测见
[清理复核报告](code-cleanup-2026-10-24.md)。本节历史结果有性能回退，不能描述为加速。

下表先计算每个场景相对变化，再对场景取中位数；正数表示耗时 / 峰值增加。

| 场景 | 数量 | 串行 P50 | 串行 P95 | CPU P50 | 四路批次 P50 | 串行峰值 | 四路峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部 | 60 | +16.80% | +19.73% | +16.73% | +17.73% | +0.57% | +1.23% |
| 原生 | 10 | +14.06% | +6.28% | +14.02% | +15.56% | +0.57% | +1.29% |
| 交叉 | 50 | +17.02% | +36.83% | +17.02% | +17.99% | +0.57% | +1.19% |

暖请求前后缓存大小和 misses 增长的场景数，两版都是 **0 / 60**。
每版 60 / 60 场景确认 peak_active=4、独立请求头、客户端数量正确且全部响应关闭。
串行与并发内存增量较小，但 CPU 时间与延迟回退是实际测量结果；不能由“缓存没有增长”推断性能没有回退。

原生 A-B-B-A 每场景每版合计 62 个 warm 样本，中位变化 **+13.82%**：

| 协议 | 模式 | 重构前 P50 ms | 重构后 P50 ms | 变化 |
| --- | --- | ---: | ---: | ---: |
| openai_chat | collect | 21.509 | 25.606 | +19.05% |
| openai_chat | stream | 21.589 | 25.759 | +19.31% |
| openai_responses | collect | 58.297 | 66.080 | +13.35% |
| openai_responses | stream | 57.933 | 66.207 | +14.28% |
| anthropic | collect | 39.535 | 41.773 | +5.66% |
| anthropic | stream | 39.475 | 41.953 | +6.28% |
| gemini | collect | 44.780 | 95.752 | +113.83% |
| gemini | stream | 44.993 | 95.758 | +112.83% |
| ollama | collect | 14.328 | 16.110 | +12.43% |
| ollama | stream | 14.433 | 16.162 | +11.98% |

Gemini 是主要回退项：原生交替复测约为原来的 **2.13 倍**；完整矩阵最差的交叉路径为
Ollama 格式输入 → Gemini API → Ollama 格式返回，collect P50 从 **24.38 ms** 变为 **84.12 ms**，
增加 **245.1%**。其请求从 1,048,978 bytes 变为 973,330 bytes；Gemini 原生请求字节则保持一致，
所以不能仅用请求变大解释回退。

本轮明确采用类型化中间值、完整来源定位和统一双向策略；工具结果对象经中间文本表示后，
Gemini 目标还需恢复 SDK 所需的响应对象。诊断显示输入准备与 SDK 转换存在额外处理成本，
已复用 JSON capture 减少通用 deepcopy 调度，但本次数据仍证明整体及 Gemini 路径更慢。
这里不把观察到的整体回退全部归因于某一个函数，也不声称复制优化已经消除回退。

**24 / 60** 场景的 SDK 请求 SHA256 完全一致，其余差异包含字段删除、工具结果规范化与必需空字段。
工具图片、完整工具调用和文本数据均通过独立断言核对。公开事件数按 usage 过滤及工具事件变化单独保存在比较数据中。
由于请求结构和结果交付语义发生变化，本比较代表完整新旧实现的成本，不能当作纯同字节转发器微基准。

实例构造是独立的一次性成本，未混入 generate 计时：

| 协议 | 前 ms | 后 ms | 增加 ms | 前保留 KB / 实例 | 后保留 KB / 实例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| openai_chat | 0.0787 | 0.1540 | 0.0753 | 8.81 | 14.96 |
| openai_responses | 0.0809 | 0.1572 | 0.0763 | 9.36 | 15.70 |
| anthropic | 0.0808 | 0.1549 | 0.0741 | 9.36 | 15.70 |
| gemini | 0.0900 | 0.1649 | 0.0749 | 10.68 | 17.01 |
| ollama | 0.0816 | 0.1555 | 0.0740 | 9.18 | 15.33 |

保留内存来自单独 tracemalloc 探针的 100 实例总量除以 100，单位为十进制 KB，不是 RSS。
按需 codec 工厂、独立可变 defaults 与 adaptor 增加了实例成本；没有构造 SDK 客户端。

## 原始数据与复现

- [完整比较、构造与 ABBA 汇总](data/protocol-middleware-2026-10-24-comparison.json)。
- [60 场景基线](data/protocol-middleware-2026-10-24-before.json)、[60 场景重构后](data/protocol-middleware-2026-10-24-after.json)。
- [构造基线](data/protocol-middleware-2026-10-24-construction-before.json)、[构造重构后](data/protocol-middleware-2026-10-24-construction-after.json)。
- ABBA：[A1](data/protocol-middleware-2026-10-24-native-before-1.json)、[B1](data/protocol-middleware-2026-10-24-native-after-1.json)、[B2](data/protocol-middleware-2026-10-24-native-after-2.json)、[A2](data/protocol-middleware-2026-10-24-native-before-2.json)。
- [旧原始负载基线](data/protocol-middleware-2026-10-24-original-before.json)仅为历史参考，不参与上面的匹配比较。

在修改运行时源码前保留 before/src；两份源码使用同一份当前 tests 和 Schema，以及同一解释器。
从仓库根目录运行，依次完成前后测量，避免与测试 / profiling 并行：

```sh
PYTHONPATH=/tmp/before/src python -m tests.performance.request_chain --iterations 9 --concurrency 4 --json /tmp/before.json
PYTHONPATH=/tmp/after/src python -m tests.performance.request_chain --iterations 9 --concurrency 4 --json /tmp/after.json
python -m tests.performance.compare /tmp/before.json /tmp/after.json --json /tmp/comparison.json
```

原生复测按 before、after、after、before 的顺序各运行一次：

```sh
PYTHONPATH=/tmp/before/src python -m tests.performance --suite native --iterations 31 --warmup 3 --json /tmp/native-before-1.json
PYTHONPATH=/tmp/after/src python -m tests.performance --suite native --iterations 31 --warmup 3 --json /tmp/native-after-1.json
PYTHONPATH=/tmp/after/src python -m tests.performance --suite native --iterations 31 --warmup 3 --json /tmp/native-after-2.json
PYTHONPATH=/tmp/before/src python -m tests.performance --suite native --iterations 31 --warmup 3 --json /tmp/native-before-2.json
```

构造使用 `python -m tests.performance.construction --iterations 500 --json ...`，同样分别选择源码。
`compare` 支持 `--construction-before / --construction-after` 及各含两个文件的
`--abba-before / --abba-after`，生成本文附加汇总，并拒绝源码、driver、夹具或 SDK 不匹配的数据。

## 最终检查

运行时代码完整测试 **2282 passed**。Ruff 与格式检查、105 个本地 Markdown 链接、两个离线示例、
wheel / sdist 构建以及 `twine check --strict` 全部通过。
