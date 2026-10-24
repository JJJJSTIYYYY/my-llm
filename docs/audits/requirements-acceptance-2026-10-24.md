# 需求全量验收复核

日期：2026-10-24。逐条对照根目录 `my-llm-refactor-requirements.md`，独立检查实际调用路径，
补充反例测试后重新运行完整测试。需求原文件保持原样。

本文记录验收补漏完成时的源码。其后的缓存检查与优化试验见
[Schema 缓存检查](schema-cache-2026-10-24.md)；试验未采用，运行时仍为本文的源码。

## 验收口径

以下三项以用户在本会话中的明确答复为准：

1. `strict`：Chat / Responses / Anthropic 保留，Gemini / Ollama 静默丢弃，覆盖原文 5.1、5.5、7.1 的冲突表述。
2. 工具调用：保留原生事件格式，在源 EOF 后集中交付完整调用；允许原生生命周期使用多个事件，不发送参数增量。
3. `extra_body` 保留原有开放控制接口，`Parameter` 保留原有功能。固定支持集合约束消息 / 工具 / 响应翻译，不对上游控制字典新增这类裁剪。

另按原文 2.3、3.3、C5、5.2 一致的默认 `raise` 执行；5.4 的“空函数体”适用于显式 hook 模式未提供 hook。
hook 可以不修改 draft、原地修改或自行抛错；三者都受测试覆盖。

## 功能验收结论

**按上述口径，C1–C10、5.1–5.6 和 5.8 均已落地并通过当前验收。原文没有 5.7 条目。**

| 需求 | 结果 | 实现与独立验证 |
| --- | --- | --- |
| 1.1 支持输入 / 输出 | 通过 | 文本、用户 / 工具图片、独立思维、函数工具、完整调用及私有 usage；五协议请求、响应及并发测试 |
| 1.2 固定协议对 | 通过 | MyLLM stdio 绑定后只读；upstream 使用 ResolvedProfile 快照；生成接口没有单次路由参数 |
| 2.1 / C1 固定具体中间模型 | 通过 | SDK 无关 ProtocolMiddleWare 与具体子模型；支持集合硬编码，Profile 不改变内容识别集合；加密独有思维正确标记 encrypted |
| 2.2 / C3 丢弃与损失边界 | 通过 | encoder 先投影再校验支持结构；不支持字段不产生损失条目；decoder 对无法表达的支持值收集 SemanticLossError |
| 2.3 / C4 完整 draft / 原始定位 | 通过 | 调用内收集全部损失后统一处理；异常携带完整 draft、方向、原始对象引用与原始路径；跨 messages / tools / chunks 多处损失聚合 |
| 3.1 / C2 工厂与实例 | 通过 | 注册表仅保存工厂；codec、组件、defaults、adaptor 按实例独立；所需协议仅创建一次，未使用协议延迟创建 |
| 3.2 支持集合与组件边界 | 通过 | encoder 只解析自己的源格式，decoder 只构造自己的目标；公开编排无具体协议翻译特判；原生自身身份仅用于签名 / 生命周期身份准入，不调用其他源协议的规则 |
| 3.3 / C5 三策略 | 通过 | 默认 raise、ignore 无持久记录、hook 按方向；二次设置隔离；hook 原地编辑后交付同一对象，返回值忽略；非流式仅在最终完整结果回译一次 |
| 3.4 同协议翻译 | 通过 | 同协议仍经过中间模型和统一策略，无同协议透传捷径 |
| C6 思维链 | 通过 | 五协议 unsigned 矩阵、同协议 signature / encrypted_content 回放；外来 opaque owner 优先于消息来源；不拼正文、不伪造签名 |
| C7 工具结果 | 通过 | 三个原生图片来源包括 Gemini function_response.parts；Responses / Anthropic 保图，其余目标报损失；对象 / 数组成为文本，应用 type 字段不会误删 |
| C8 工具 Schema | 通过 | name / description / parameters 翻译；明确 {} 与缺省 / None 分开；可表达时保留，Ollama 强加约束时统一报损失并保留后续 draft；strict 按确认处理 |
| C9 不支持字段 | 通过 | audio / file / 用户 metadata / refusal / citations 等投影删除；畸形未知字段不被读取或复制；非 function 调用不伪造成函数调用 |
| C10 响应方向 | 通过 | 实时批次与最终结果都经 BaseCodec 的 output 策略；非流式跳过丢弃的事件回译，在 EOF 后处理完整结果 |
| 6 禁止项 | 通过 | 不支持 audio / file 内容翻译，不伪造签名，不公开工具参数增量，不动态路由，不自动搬移图片，无公共协议特判 |
| 7 迁移 | 通过 | 公开 SemanticLossError、损失类型、adaptor、双向 hook 与二次设置；工厂注册和行为变化有迁移说明 |

原生 decoder 检查自己是否拥有签名、response item 身份或 stop sequence，是中间值的所属协议约束；
没有按其他源协议的名称选择转换算法。相关差异全部位于各原生 codec 实现中。
Origin 只引用原始对象；未知字段不会因此被枚举、复制、翻译或作为独立损失记录。

## 验收章节对照

| 验收 | 状态 | 直接证据 |
| --- | --- | --- |
| 5.1 支持字段与丢弃 | 通过 | [原生固定集合矩阵](../../tests/unit/test_native_refactor.py)、[新增原生反例](../../tests/unit/test_requirements_native_acceptance.py)；应用 Schema / 参数内同名字段保持开放 JSON |
| 5.2 策略与 hook | 通过 | [策略与身份](../../tests/unit/test_semantic_loss.py)、[完整 draft 反验收](../../tests/unit/test_requirements_runtime_acceptance.py)、[合法 falsey callable hook](../../tests/unit/test_requirements_policy_acceptance.py) |
| 5.3 思维链 | 通过 | [签名 / unsigned 矩阵](../../tests/unit/test_native_refactor.py)、[显式 owner 优先级](../../tests/unit/test_requirements_native_acceptance.py)、[加密 mark](../../tests/unit/test_requirements_models_acceptance.py)；消息与响应事件均覆盖 |
| 5.4 工具结果图片 | 通过 | [三来源图片与原始位置](../../tests/unit/test_requirements_native_acceptance.py)、[hook 主动搬移图片](../../tests/unit/test_semantic_loss.py)；未配置策略时仍为默认 raise |
| 5.5 工具定义 | 通过 | [strict 五目标与 Schema 约束](../../tests/unit/test_native_refactor.py)、[空 Schema / 可选 Schema](../../tests/unit/test_requirements_native_acceptance.py) |
| 5.6 工具 / usage | 通过 | [五 stdio EOF 与完整双工具](../../tests/unit/test_requirements_runtime_acceptance.py)、[usage 缺省 / 非流式 / 提前关闭](../../tests/unit/test_runtime_refactor.py)、[并发 / 失败 / 取消](../../tests/integration/test_concurrency.py) |
| 5.8 生命周期 | 通过 | [组件 / 策略隔离](../../tests/unit/test_runtime_refactor.py)、[仅一次构造](../../tests/unit/test_instance_default_resolution.py)、[自定义协议注册](../../tests/integration/test_protocol_registration.py)、[客户端关闭](../../tests/unit/test_client_lifecycle.py) |

## 本轮发现并修复的遗漏

此前的常规测试确实遗漏了以下情况，本轮均增加独立回归：

1. 非流式调用提前回译事件，hook / 默认异常拿不到后续完整正文。现在只累积，在 EOF 后处理最终结果。
2. Gemini 工具结果 `function_response.parts` 中的图片被遗漏。现在读取图片、丢弃非图片媒体，并保持真实 SDK / dict 来源与原索引。
3. 空工具 Schema `{}` 被真值默认分支改成 object，Gemini 缺省 / None Schema 也被造出 `{}`。现在分别保留、缺省或报告目标约束损失。
4. 应用数组内的 `type` 字段被误认为原生内容块而丢弃。现在各 encoder 声明原生标签，应用数组正常序列化为文本。
5. Chat custom 调用被误造为 function。现在丢弃未支持的调用类型，保留合法函数参数片段的内部收集。
6. 消息来源能盖过思维 / 工具值的显式签名 owner。现在各 decoder 优先遵守值的 owner，外来签名进入损失策略。
7. 合法 callable hook 的布尔值为 False 时被当成空 hook。现在只以 None 判断是否未提供。
8. 仅有 encrypted_content 的思维被标记为 summary。现在使用 encrypted，摘要与 opaque payload 同时存在时各自保持。
9. SDK 合法的空参数 / 空片段未归一化：Gemini `args=None` 现在表示空参数对象，Chat `arguments=None` 视为未提供片段。

四个新增验收文件共增加 **129 项用例**；共用原生夹具归入 tests/helpers，测试模块互不导入。

## 最终验证与性能数据

当前冻结源码 SHA256：`4fae8edb3daf0362d61998411fb39889619b2f457dbdb2425613a0fba2eeb3aa`。
完整测试 **2433 passed in 18.22s**。Ruff 与格式检查通过。
最终源码重新完成 **60 场景**，每场景串行 2 次预热 / 9 次采样、四路 2 批预热 / 9 批采样，
另有独立内存和缓存探针。harness SHA256 为 `f8d5ab40e9184751af15e49c0acd0c08528a43cbf503706b34cfd5516a584ff0`，与两份基线完全一致。
比较器核对 Python、SDK、规模、采样数、负载 / 响应 SHA256 及 HTTP 分块。
两个 before 使用各自已有的独立历史快照结果，本轮重新测量最终 after。

下表先计算每场景百分比变化，再取场景中位数；负数为降低。

| 相对版本 | 场景 | 串行 P50 | 串行 P95 | 四路 P50 | 串行峰值 | 四路峰值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 上一轮清理后 | 全部60 | -2.80% | -2.44% | -2.29% | +0.00% | +0.00% |
| 上一轮清理后 | 非流式30 | -5.54% | -4.21% | -5.89% | +0.00% | +0.00% |
| 上一轮清理后 | 流式30 | -0.02% | -0.51% | +0.37% | +0.00% | +0.00% |
| 最初重构前 | 全部60 | +6.92% | +7.53% | +7.06% | +0.57% | +1.20% |
| 最初重构前 | 非流式30 | +2.51% | +3.49% | +3.87% | +0.57% | +1.20% |
| 最初重构前 | 流式30 | +9.08% | +8.94% | +10.56% | +0.57% | +1.20% |

相对最初重构前，串行 P50 仍高 **6.92%**、四路 P50 高 **7.06%**，最差串行项高 **18.68%**。
本轮不宣称已恢复全部原始性能。需求只要求完成并报告性能对照，没有规定加速比例或硬性上限。
相对上一轮清理后，非流式 P50 中位降低 **5.54%**，流式约 **-0.02%**；
这与删除非流式丢弃的事件回译相符，但单次矩阵不能把全部波动归因于该修改。

本轮与上一轮清理后 SDK 请求字节完全一致 **60 / 60**，
相对最初重构前相同 **20 / 60**，此前已记录的 Gemini 工具结果文本规范化等差异继续单独保留。
全部场景 `peak_active=4`，请求头隔离、客户端数量与响应关闭检查通过。
两份基线及最终 after 的暖请求缓存大小 / misses 增长均为 **0 / 60**。

本轮没有重新做原生 A–B–B–A；旧交替数据只说明当时的冻结源码。
计时不启用 tracemalloc；内存是独立 Python 分配峰值，不是 RSS。P95 不是置信界。
这里测量本地官方 SDK 处理，不包含真实网络 / 模型推理，没有使用 live provider 凭据。

构造单独采样 500 次，内存探针另保留 100 个实例。保留内存单位为十进制 KB / 实例：

| 协议 | 上一轮 ms | 最终 ms | 上一轮 KB | 最终 KB |
| --- | ---: | ---: | ---: | ---: |
| openai_chat | 0.1180 | 0.1189 | 14.24 | 14.24 |
| openai_responses | 0.1206 | 0.1203 | 14.97 | 14.97 |
| anthropic | 0.1214 | 0.1209 | 14.97 | 14.97 |
| gemini | 0.1300 | 0.1290 | 16.28 | 16.28 |
| ollama | 0.1200 | 0.1179 | 14.60 | 14.60 |

原始数据与可复现比较：

- [最终60场景](data/requirements-acceptance-2026-10-24-after.json)、[最终构造](data/requirements-acceptance-2026-10-24-construction-after.json)。
- [相对上一轮清理](data/requirements-acceptance-2026-10-24-comparison.json)、[相对最初重构前](data/requirements-acceptance-2026-10-24-original-comparison.json)。
- 基线：[上一轮](data/protocol-cleanup-2026-10-24-after.json)、[最初重构前](data/protocol-middleware-2026-10-24-before.json)。

沿用[性能指南](../guides/performance.md)的源码快照选择方式：

```sh
PYTHONPATH=/tmp/after/src python -m tests.performance.request_chain --iterations 9 --concurrency 4 --json /tmp/after.json
PYTHONPATH=/tmp/after/src python -m tests.performance.construction --iterations 500 --json /tmp/construction-after.json
python -m tests.performance.compare /tmp/before.json /tmp/after.json --construction-before /tmp/construction-before.json --construction-after /tmp/construction-after.json --json /tmp/comparison.json
```

每个报告保存 runtime / harness 指纹；最终 after 源码摘要与本文及当前工作区一致。


原先[重构验收记录](protocol-middleware-2026-10-24.md)和[清理复核记录](code-cleanup-2026-10-24.md)
分别记录各自当时的冻结源码与测量结果，不能作为本轮修补后的源码验证。
