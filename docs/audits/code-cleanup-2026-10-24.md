# 重构代码清理与职责复核

日期：2026-10-24。范围为本次重构后的全部 `src/my_llm`，对照实际调用链检查生产引用、
数据所有权、私有方法输入与原生协议边界。

本报告记录清理阶段的独立快照；后续需求反向验收发现的遗漏、修复及最新验证见
[需求全量验收复核](requirements-acceptance-2026-10-24.md)。

## 已清理的问题

| 问题 | 修改 |
| --- | --- |
| 工具结果已经转为文本，Origin 仍深拷贝整份源对象 | Origin 只借用原始引用；文本自身与原输入独立，定位不枚举未支持字段 |
| 原生 encoder 先深拷贝参数 / Schema，中间模型随后又捕获一次 | 在中间模型统一建立应用数据所有权；删除原生解析阶段的重复复制 |
| 请求多次经过公开列表转换 | 请求归一化一次后进入私有校验与解析；公开入口仍验证外部输入 |
| 长消息片段合并不断重拷贝已合并内容 | 首次合并复制容器，后续追加，消除二次复杂度 |
| 默认能力完整时逐事件重建没有变化的字典 | 内部只读借用；最终公开输出维持独立所有权 |
| 工厂迁移后 Profile 解析临时创建 codec，MyLLM 随后再创建一份 | Profile 解析使用本实例的 defaults getter，只按需创建一次所需 codec |
| Gemini 把确定的工具结果文本再猜测解析成对象 | 固定 SDK `{"result": text}` 包装；JSON 外观文本保持文本 |
| Gemini config 对工具 Schema 重复建模 | 工具单独校验，其他 config 字段仍完整校验，包括非法 temperature |
| 中间层保留 Gemini 签名字段别名 | 统一 `signature`；`thought_signature` 仅在 Gemini 原生入出边界映射 |
| 不支持字段已不进入中间协议，流状态仍处理 annotations / refusal | 删除不可达处理、旧计数分支及相应空常量 / 恒等函数 |
| 私有确定输入上的类型兜底、旧全量 codec 实例枚举 | 删除重复检查、无调用入口及旧字段缺省；有限内部操作直接分派 |
| 投影对象额外的实例字典 / 临时字典 | 中间模型和来源投影使用 slots；复制时直接填目标容器 |
| 损失去重逐项扫描已有列表 | 调用内有序字典去重，完整错误生成时才创建公开列表 |

共享工具中的单协议工具结果格式化也移入 Responses decoder。没有为了表面相似而合并各协议的
原生规则。按生产代码引用扫描私有函数，未发现零引用函数；Ruff 检查未发现未使用导入或变量。

## 保留的边界

`generate → Encoder.request → ProtocolMiddleWare → Decoder.prepare → {client, request, local} → Transport`。
响应经过原生 encoder、调用独立流状态、目标 decoder 及损失策略。公共编排、Profile、中间模型、
损失收集和流状态中未发现按原生协议名称选择翻译行为的条件分支。
注册表和可选 SDK 目录中的名称只用于注册、依赖发现与路由。

原生字段识别 / 过滤归 encoder；目标字段、签名准入、Schema 限制和 SDK 请求组装归 decoder；
组件及策略由 codec 管理；SDK 客户端和响应资源归 transport。原生模块中的私有内容 helper
仍是所属 encoder / decoder 的实现细节。

保留以下有实际职责的复制和检查：

- 中间模型捕获工具参数和 Schema，避免调用者后续修改原输入。
- 原生 SDK 请求、公开转换结果与流状态快照建立各自所有权。
- Profile / defaults / PreparedOptions 的快照支持独立实例与重复调用。
- 公开入口验证输入；支持集合投影过滤未知字段；配置禁用支持字段时记录损失。
- SDK 校验后的序列化用于触发 `Iterable` 的延迟校验，不能直接删除。
- BaseCodec 处理 decoder 主动抛出完整损失异常，以及收集结束后的异常；两者载荷封装位置不同。

私有方法不再为已经确定的输入结构维护旧格式兼容。共享缓存继续只保存有界类型规则，
没有加入请求、来源对象、损失集合或 codec 实例缓存。

## 验证与性能

最终完整测试 **2304 passed**；包含原始对象所有权、三类 encoder 自定义校验钩子的两个入口、
五协议按需单次构造、签名入出边界、公开 config 非法值、并发、取消和资源关闭。
Ruff / 格式检查通过。以下性能数据对应完整测试后的冻结源码。

清理后源码 SHA256：`dc950fc5227929b7944ebe21b9d2bd22513abbe4b80d7960d372eab195a27c09`；清理前：`13ffa2cc013e7acb99d25e49638e9895d6de6cf7cb0597b95698c54574e2b3dd`。
测量 harness SHA256：`f8d5ab40e9184751af15e49c0acd0c08528a43cbf503706b34cfd5516a584ff0`，与首次重构测量完全一致。
两个对照使用相同解释器 / SDK / 负载 / 响应 / HTTP 分块，比较器检查全部指纹。
60 场景串行各 9 个 warm 样本、四路各 9 批；计时不启用 tracemalloc。
内存、缓存和真实重叠并发分别探测。原生另做 A–B–B–A，每场景每版合计 62 个 warm 样本。

下表为每场景变化百分比的中位数，负数表示降低。首次重构的两个历史矩阵保留原始结果，
本轮没有重跑它们；原生交替复测则重新运行清理前 / 后两份冻结源码，降低时段波动影响。

| 相对版本 | 场景 | 串行 P50 | 串行 P95 | CPU P50 | 四路 P50 | 串行峰值 | 四路峰值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 本次清理前 | 全部 60 | -3.41% | -3.36% | -3.50% | -3.39% | -0.06% | -0.01% |
| 本次清理前 | 原生 10 | -3.40% | -2.31% | -3.48% | -2.61% | -0.06% | -0.56% |
| 本次清理前 | 交叉 50 | -3.69% | -3.37% | -3.70% | -3.65% | -0.03% | +0.00% |
| 最初重构前 | 全部 60 | +9.41% | +7.78% | +9.31% | +10.14% | +0.57% | +1.20% |
| 最初重构前 | 原生 10 | +8.05% | -9.38% | +7.98% | +11.05% | +0.60% | +1.09% |
| 最初重构前 | 交叉 50 | +9.53% | +11.91% | +9.53% | +9.85% | +0.57% | +1.27% |

本次清理前后的原生交替复测，中位变化 **-2.64%**：

| 协议 | 模式 | 清理前 ms | 清理后 ms | 变化 |
| --- | --- | ---: | ---: | ---: |
| openai_chat | collect | 25.760 | 25.120 | -2.49% |
| openai_chat | stream | 26.172 | 25.129 | -3.99% |
| openai_responses | collect | 66.541 | 65.304 | -1.86% |
| openai_responses | stream | 66.565 | 65.084 | -2.23% |
| anthropic | collect | 42.003 | 41.152 | -2.02% |
| anthropic | stream | 41.939 | 41.089 | -2.03% |
| gemini | collect | 95.850 | 31.442 | -67.20% |
| gemini | stream | 95.760 | 31.391 | -67.22% |
| ollama | collect | 16.370 | 15.912 | -2.79% |
| ollama | stream | 16.434 | 15.675 | -4.61% |

Gemini 原生从约 **95.8 ms** 降到 **31.4 ms**，交替复测约降低 **67.2%**。
此前最慢交叉项 Ollama 格式 → Gemini API → Ollama 返回，collect 从 **84.12 ms** 降到
**25.64 ms**；最初重构前为 **24.38 ms**，当前仍高 **5.18%**。
相对最初重构前，全部场景串行 P50 中位仍高 **9.41%**、四路 **10.14%**，
最差串行项高 **17.74%**。本轮不能据此宣称已恢复全部原始性能，或不存在进一步优化空间。

清理前后 SDK 请求 SHA256 相同 **48 / 60**，相对最初重构前相同 **20 / 60**。
差异包含 Gemini 工具结果改为固定文本包装以及公开空字段规范化。Gemini 原生请求从
970,001 bytes 变为 1,042,961 bytes，仍更快；收益来自减少解析 / 对象遍历等整体变化，
不能将本次测量解释成严格同字节转发器比较，也不把全部收益归因于某一个函数。
负载、工具参数、工具结果、图片及模型输出均经过相同独立断言核对。

暖请求缓存大小 / misses 增长均为 **0 / 60**。全部场景真实 `peak_active=4`，
请求头独立，响应全部关闭。内存峰值是 Python tracemalloc 分配，不是 RSS；P95 为样本统计，
不是置信界。没有调用真实模型或外部网络。

构造独立测 500 个实例，保留内存来自另一次 100 实例探针，十进制 KB / 实例：

| 协议 | 清理前 ms | 清理后 ms | 变化 | 前保留 KB | 后保留 KB |
| --- | ---: | ---: | ---: | ---: | ---: |
| openai_chat | 0.1540 | 0.1180 | -23.38% | 14.96 | 14.24 |
| openai_responses | 0.1572 | 0.1206 | -23.31% | 15.70 | 14.97 |
| anthropic | 0.1549 | 0.1214 | -21.66% | 15.70 | 14.97 |
| gemini | 0.1649 | 0.1300 | -21.21% | 17.01 | 16.28 |
| ollama | 0.1555 | 0.1200 | -22.82% | 15.33 | 14.60 |

1 万片段合并的单独诊断中位数从 **98.374 ms** 到 **2.360 ms**，只用于验证二次复制问题；
它不等于完整请求链路的加速倍数。

## 原始数据与复现

- [本次清理前后完整比较](data/protocol-cleanup-2026-10-24-comparison.json)、[相对最初重构前](data/protocol-cleanup-2026-10-24-original-comparison.json)。
- [清理后 60 场景](data/protocol-cleanup-2026-10-24-after.json)、[清理前 60 场景](data/protocol-middleware-2026-10-24-after.json)、[最初基线](data/protocol-middleware-2026-10-24-before.json)。
- [清理后构造](data/protocol-cleanup-2026-10-24-construction-after.json)、[清理前构造](data/protocol-middleware-2026-10-24-construction-after.json)。
- 原生交替复测：[A1](data/protocol-cleanup-2026-10-24-native-before-1.json)、[B1](data/protocol-cleanup-2026-10-24-native-after-1.json)、[B2](data/protocol-cleanup-2026-10-24-native-after-2.json)、[A2](data/protocol-cleanup-2026-10-24-native-before-2.json)。

沿用[性能指南](../guides/performance.md)及[原始测量条件](protocol-middleware-2026-10-24.md#测量条件)。
两份源码使用同一套当前 tests / SDK，分别以 `PYTHONPATH` 选择 `src`。不要与测试或剖析同时计时。

```sh
PYTHONPATH=/tmp/before/src python -m tests.performance.request_chain --iterations 9 --concurrency 4 --json /tmp/before.json
PYTHONPATH=/tmp/after/src python -m tests.performance.request_chain --iterations 9 --concurrency 4 --json /tmp/after.json
python -m tests.performance.compare /tmp/before.json /tmp/after.json --json /tmp/comparison.json
```

原生按 before、after、after、before 依次运行 `python -m tests.performance --suite native --iterations 31 --warmup 3 --json ...`。
构造运行 `python -m tests.performance.construction --iterations 500 --json ...`。
比较器的 `--abba-before / --abba-after`、`--construction-before / --construction-after` 接收对应原始文件，
核对源码及测量指纹后生成附加汇总。
