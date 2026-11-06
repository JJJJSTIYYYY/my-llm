# 性能测试记录

更新日期：2026-11-06。本文件统一保存近期性能变化、历史结论和最新完整结果；测试运行方法见[性能测试指南](../guides/performance.md)。

最新一次为**仅省去消息缓存过滤前全量深拷贝**的对照：缓存准备阶段 P50 变化中位数为增量 **-8.56%**、连续重试 **-7.65%**，开启缓存的完整 generate 为 **-1.12%**。24 轮原生历史的缓存准备内存峰值下降约 **18%～37%**。

## 最近的性能变化

图中每个点都是该批测试相对其自身基线的变化，负数表示耗时下降。每条线先计算逐场景 P50 变化，再取三种上游的中位数。仅选取 `openai_chat`、`openai_responses`、`anthropic` 的原生场景，聚合和流式各 3 项。

三批的代码、计时诊断及缓存状态不同，连线仅用于按测试批次浏览，**不能累乘为累计提速，也不能比较跨批的绝对耗时**。前两批来自移除 Gemini／Ollama 前的历史版本；最后一批开启消息缓存。ReasoningContent 图点使用主重构的 ABBA 快照，不包含后来 Responses 反馈所有权补丁。

<figure style="margin: 24px 0; padding: 16px; border: 1px solid #d0d7de; border-radius: 12px; background: #fff; color: #24292f;">
<figcaption style="font-weight: 600; margin-bottom: 8px;">最近三批完整 generate 耗时变化</figcaption>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 380" role="img" aria-labelledby="performance-title performance-desc" style="display: block; width: 100%; height: auto; font-family: sans-serif;">
  <title id="performance-title">三协议原生 generate 的 P50 相对变化</title>
  <desc id="performance-desc">每批分别相对自己的修改前基线。每条线取三个上游场景变化率的中位数。负数表示耗时下降。最后一批开启消息缓存，与前两批不同，不能累计计算提速。</desc>
  <rect x="0" y="0" width="900" height="380" fill="#fff"/>
  <rect x="595" y="60" width="235" height="240" rx="8" fill="#fff7ed"/>
  <text x="80" y="25" font-size="14" fill="#57606a">P50 相对各批基线的变化（%）↓ 越低越快</text>
  <line x1="535" y1="23" x2="565" y2="23" stroke="#0969da" stroke-width="3"/>
  <text x="574" y="28" font-size="14" fill="#24292f">聚合</text>
  <line x1="665" y1="23" x2="695" y2="23" stroke="#b45309" stroke-width="3" stroke-dasharray="7 5"/>
  <text x="704" y="28" font-size="14" fill="#24292f">流式</text>
  <line x1="80" y1="80.00" x2="830" y2="80.00" stroke="#eaeef2" stroke-width="1"/>
  <text x="68" y="85.00" text-anchor="end" font-size="13" fill="#57606a">+1.0%</text>
  <line x1="80" y1="115.00" x2="830" y2="115.00" stroke="#eaeef2" stroke-width="1"/>
  <text x="68" y="120.00" text-anchor="end" font-size="13" fill="#57606a">+0.5%</text>
  <line x1="80" y1="150.00" x2="830" y2="150.00" stroke="#8c959f" stroke-width="2"/>
  <text x="68" y="155.00" text-anchor="end" font-size="13" fill="#57606a">+0.0%</text>
  <line x1="80" y1="185.00" x2="830" y2="185.00" stroke="#eaeef2" stroke-width="1"/>
  <text x="68" y="190.00" text-anchor="end" font-size="13" fill="#57606a">-0.5%</text>
  <line x1="80" y1="220.00" x2="830" y2="220.00" stroke="#eaeef2" stroke-width="1"/>
  <text x="68" y="225.00" text-anchor="end" font-size="13" fill="#57606a">-1.0%</text>
  <line x1="80" y1="255.00" x2="830" y2="255.00" stroke="#eaeef2" stroke-width="1"/>
  <text x="68" y="260.00" text-anchor="end" font-size="13" fill="#57606a">-1.5%</text>
  <line x1="80" y1="290.00" x2="830" y2="290.00" stroke="#eaeef2" stroke-width="1"/>
  <text x="68" y="295.00" text-anchor="end" font-size="13" fill="#57606a">-2.0%</text>
  <text x="837" y="155" font-size="12" fill="#57606a">基线</text>
  <line x1="580" y1="60" x2="580" y2="300" stroke="#9a6700" stroke-dasharray="4 5"/>
  <polyline data-series="collect" points="190,115.76 450,188.06 710,208.94" fill="none" stroke="#0969da" stroke-width="3" stroke-linejoin="round"/>
  <circle cx="190" cy="115.76" r="5" fill="#0969da" stroke="#fff" stroke-width="2"><title>11-04 · ReasoningContent 聚合：+0.49%</title></circle>
  <text x="190" y="102.76" text-anchor="middle" font-size="15" font-weight="600" fill="#0969da">+0.49%</text>
  <circle cx="450" cy="188.06" r="5" fill="#0969da" stroke="#fff" stroke-width="2"><title>11-04 · Parameter 聚合：-0.54%</title></circle>
  <text x="450" y="175.06" text-anchor="middle" font-size="15" font-weight="600" fill="#0969da">-0.54%</text>
  <circle cx="710" cy="208.94" r="5" fill="#0969da" stroke="#fff" stroke-width="2"><title>11-06 · 缓存深拷贝 聚合：-0.84%</title></circle>
  <text x="710" y="195.94" text-anchor="middle" font-size="15" font-weight="600" fill="#0969da">-0.84%</text>
  <polyline data-series="stream" points="190,134.19 450,127.95 710,235.72" fill="none" stroke="#b45309" stroke-width="3" stroke-linejoin="round" stroke-dasharray="7 5"/>
  <circle cx="190" cy="134.19" r="5" fill="#b45309" stroke="#fff" stroke-width="2"><title>11-04 · ReasoningContent 流式：+0.23%</title></circle>
  <text x="190" y="157.19" text-anchor="middle" font-size="15" font-weight="600" fill="#b45309">+0.23%</text>
  <circle cx="450" cy="127.95" r="5" fill="#b45309" stroke="#fff" stroke-width="2"><title>11-04 · Parameter 流式：+0.32%</title></circle>
  <text x="450" y="114.95" text-anchor="middle" font-size="15" font-weight="600" fill="#b45309">+0.32%</text>
  <circle cx="710" cy="235.72" r="5" fill="#b45309" stroke="#fff" stroke-width="2"><title>11-06 · 缓存深拷贝 流式：-1.22%</title></circle>
  <text x="710" y="222.72" text-anchor="middle" font-size="15" font-weight="600" fill="#b45309">-1.22%</text>
  <text x="190" y="320" text-anchor="middle" font-size="14" fill="#24292f">11-04</text>
  <text x="190" y="342" text-anchor="middle" font-size="14" fill="#24292f">ReasoningContent</text>
  <text x="450" y="320" text-anchor="middle" font-size="14" fill="#24292f">11-04</text>
  <text x="450" y="342" text-anchor="middle" font-size="14" fill="#24292f">Parameter</text>
  <text x="710" y="320" text-anchor="middle" font-size="14" fill="#24292f">11-06</text>
  <text x="710" y="342" text-anchor="middle" font-size="14" fill="#24292f">缓存深拷贝</text>
  <text x="80" y="370" font-size="12" fill="#57606a">前两批：历史五协议版本中选取三协议原生场景；最后一批：三协议原生场景，消息缓存开启。</text>
</svg>
</figure>

| 测试批次 | 聚合 P50 变化中位数 | 流式 P50 变化中位数 | 图中范围 |
| --- | ---: | ---: | --- |
| 11-04 · ReasoningContent | +0.49% | +0.23% | 三协议 × 聚合／流式，共 6 场景 |
| 11-04 · Parameter | -0.54% | +0.32% | 三协议 × 聚合／流式，共 6 场景 |
| 11-06 · 缓存深拷贝 | -0.84% | -1.22% | 三协议 × 聚合／流式，共 6 场景 |

<details>
<summary>图表使用的逐场景原始数据（P50 ms）</summary>

| 批次 | 上游 | 模式 | 修改前 | 修改后 | 变化 |
| --- | --- | --- | ---: | ---: | ---: |
| 11-04 · ReasoningContent | openai_chat | 聚合 | 22.640500 | 22.927521 | +1.27% |
| 11-04 · ReasoningContent | openai_chat | 流式 | 24.208063 | 24.453437 | +1.01% |
| 11-04 · ReasoningContent | openai_responses | 聚合 | 60.731146 | 60.595104 | -0.22% |
| 11-04 · ReasoningContent | openai_responses | 流式 | 62.489604 | 62.630730 | +0.23% |
| 11-04 · ReasoningContent | anthropic | 聚合 | 39.139750 | 39.331209 | +0.49% |
| 11-04 · ReasoningContent | anthropic | 流式 | 40.679104 | 40.721063 | +0.10% |
| 11-04 · Parameter | openai_chat | 聚合 | 23.635376 | 23.210187 | -1.80% |
| 11-04 · Parameter | openai_chat | 流式 | 24.665563 | 24.695124 | +0.12% |
| 11-04 · Parameter | openai_responses | 聚合 | 61.030958 | 60.699083 | -0.54% |
| 11-04 · Parameter | openai_responses | 流式 | 63.091813 | 63.290563 | +0.32% |
| 11-04 · Parameter | anthropic | 聚合 | 39.515625 | 39.849834 | +0.85% |
| 11-04 · Parameter | anthropic | 流式 | 40.679438 | 40.923249 | +0.60% |
| 11-06 · 缓存深拷贝 | openai_chat | 聚合 | 18.309750 | 18.155583 | -0.84% |
| 11-06 · 缓存深拷贝 | openai_chat | 流式 | 20.039708 | 19.794313 | -1.22% |
| 11-06 · 缓存深拷贝 | openai_responses | 聚合 | 54.529875 | 54.240271 | -0.53% |
| 11-06 · 缓存深拷贝 | openai_responses | 流式 | 56.656583 | 56.082583 | -1.01% |
| 11-06 · 缓存深拷贝 | anthropic | 聚合 | 35.123792 | 33.710167 | -4.02% |
| 11-06 · 缓存深拷贝 | anthropic | 流式 | 36.556333 | 35.443313 | -3.04% |

</details>

支持内嵌 HTML／SVG 的 Markdown 渲染器可直接显示上图；表格提供相同数据。

## 历史测试摘要

### 2026-11-04：ReasoningContent 重构

- 完整主测 60 场景，P50 变化中位数 **-1.18%**；独立 ABBA 复测 18 场景 **+0.40%**，范围 -1.01%～+1.42%。主测原生 10 项为 -0.20%，交叉 50 项为 -1.20%；ABBA 原生 10 项为 +0.27%，选定交叉 8 项为 +0.87%。
- 主测每场景 3 次预热、15 样本；ABBA 每轮 5 次预热、41 样本，每版本每场景合并 82 样本。ABBA 带有两版一致的 GC 回调，绝对时间不与无回调主测混用。
- 最后补齐 Responses 原生反馈所有权后，A／B／C／C／B／A 单独复测，最终聚合／流式相对原始实现为 **+0.44%／+0.22%**；相对主重构快照为 +0.72%／+0.32%。每版本每场景 30 样本。
- 60／60 完整请求字节一致。诊断记录 237 次二代 GC，暂停中位数 37.76 ms、最大 52.04 ms；短样本尾部尖峰不能直接判为回退。安静环境对 Ollama → Chat 复测未复现首轮约 5%～7% 回退。
- 结论：整体成本接近原实现，不能据单轮数据宣称普遍提速。新增的反馈路由和所有权隔离有实际工作成本。
- [历史原始数据](data/2026-11-04-reasoning-content.json)保留全部主测、ABBA、最终补丁对照、GC 样本、环境、源码指纹和源码差异。

### 2026-11-04：Parameter 与事件路由

- 首轮完整主测 60 场景为 **+1.48%**；交替复测 10 个原生场景后为 **+0.22%**，范围 -1.80%～+2.06%。原生交替复测每轮 5 次预热、21 样本，每版本每场景合并 42 样本。
- 首轮 Ollama → Chat 的 +7.33%／+7.01% 经当前／旧／旧／当前复测后为 **-1.53%／+0.53%**，没有复现原幅度。60／60 请求、响应、负载指纹与分块一致。
- 去重事件匹配的候选完整原生 P50 中位变化为 **+0.02%**；独立事件微基准、惰性回退和 Profile 字段缓存未证明完整调用有明显收益。候选仅在临时源码实验，没有进入运行时代码。
- 匹配函数在诊断中的累计时间占比仅约 0.09%～0.44%。两次二代 GC 暂停约 40 ms；逐场景串行／并发内存峰值差异小于 0.02%，规则缓存未增长，四路并发探针 `peak_active=4`，三个候选各通过 180 项相关测试。
- [历史原始数据](data/2026-11-04-parameter-events.json)保留完整对照、候选补丁、局部驱动、构造、并发、缓存及 GC 诊断。

## 最新完整结果：消息缓存深拷贝优化（2026-11-06）

最终仅保留缓存过滤前省去一次全量深拷贝的改动；之前的循环合并、惰性投影及相关接口修改全部撤回。
以下数据来自最终精简代码的复测，替换之前混合优化的报告。

### 保留的改动

- MessageCacheBatch.append 继续捕获新增批次并暂存历史，返回只读借用的历史引用，不再深复制全量历史。
- 组装和过滤只读借用输入；最终过滤仍产生独立请求。hook 对 draft 的普通修改不会自动回写缓存。
- cache.get() 的独立快照以及 replace/extend 的数据捕获保留。
- Decoder 仅补充借用约定的 docstring；AST 去除 docstring 后，仅 message_cache.py 有可执行差异。
- 原有循环、校验顺序、批量接口及协议处理均恢复，公共方法没有新增控制入参。

### 测量方法

- 基线：`44661bb2137225bfaefe172617fc05f94f4de682` 的原始 src；候选只包含上述深拷贝改动。
- 所有计时按旧／新／新／旧串行运行，同一解释器、SDK、驱动和夹具；默认 GC，计时不与测试或内存探针重叠。
- 准备阶段 72 场景：3 上游 × 4 stdio（含 middleware）× 2／24 轮历史 × 关闭缓存／增量／连续重试。每轮预热 7 次、41 样本，每版本每场景合并 82 样本。
- 增量场景逐样本在计时外预填相同历史；重试场景仅预填一次后连续重试，不重建或增长历史。
- 完整 generate 专门开启缓存，覆盖三种上游的原生聚合与流式共 6 场景，默认复杂负载为 12 轮历史、12 工具、4 路调用、2048 字符输入和 8192 字符输出。每轮预热 5 次、21 样本，每版本每场景合并 42 样本。
- 完整调用包含实际 SDK 序列化、消费、结果交付和响应关闭；客户端复用、历史预填位于计时外。使用内存 HTTPX 响应，不含真实网络和模型推理。
- 72 个准备场景的请求及负载指纹、6 个完整调用的请求指纹及输出事件数在四轮均一致；完整输出逐次经过同一套独立验证。准备阶段仅归一化 Responses 生成 item ID 的随机前缀。
- 内存另外对 24 轮原生历史执行三次 tracemalloc 探针，缓存预填在探针外，峰值取中位数，不是进程 RSS。
- 关闭缓存的执行逻辑与基线一致，其计时作为噪声对照。

### 时间结果

组变化先对每个场景计算相对 P50，再取中位数；负数表示耗时下降。

| 场景 | 场景数 | P50 变化中位数 | 单场景范围 |
| --- | ---: | ---: | --- |
| 关闭缓存准备 | 24 | -0.34% | -4.37%～+2.80% |
| 增量缓存准备 | 24 | -8.56% | -17.61%～-1.93% |
| 连续重试准备 | 24 | -7.65% | -30.39%～-2.93% |
| 开启缓存的完整 generate | 6 | -1.12% | -4.02%～-0.53% |

完整开启缓存的调用结果：

| 上游 | 模式 | 修改前 P50 ms | 修改后 P50 ms | 变化 |
| --- | --- | ---: | ---: | ---: |
| openai_chat | 聚合 | 18.310 | 18.156 | -0.84% |
| openai_chat | 流式 | 20.040 | 19.794 | -1.22% |
| openai_responses | 聚合 | 54.530 | 54.240 | -0.53% |
| openai_responses | 流式 | 56.657 | 56.083 | -1.01% |
| anthropic | 聚合 | 35.124 | 33.710 | -4.02% |
| anthropic | 流式 | 36.556 | 35.443 | -3.04% |

### 准备阶段完整明细（72 场景）

下表保留全部上游 × 输入格式 × 历史规模 × 缓存模式的结果。P50、P95 和 CPU P50 均由每版本两轮合计 82 个样本计算，单位 ms；P95 使用 nearest-rank。CPU 为进程 CPU 时间。每组展开可查看全部 24 场景。

<details>
<summary>关闭缓存：24 场景</summary>

| 上游 | stdio | 历史轮数 | 旧 P50 | 新 P50 | 变化 | 旧 P95 | 新 P95 | 旧 CPU P50 | 新 CPU P50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openai_chat | openai_chat | 2 | 0.571 | 0.564 | -1.13% | 0.649 | 0.621 | 0.571 | 0.565 |
| openai_chat | openai_responses | 2 | 0.738 | 0.728 | -1.45% | 0.914 | 0.865 | 0.739 | 0.728 |
| openai_chat | anthropic | 2 | 0.728 | 0.720 | -1.17% | 0.875 | 0.867 | 0.730 | 0.720 |
| openai_chat | middleware | 2 | 0.427 | 0.426 | -0.27% | 0.484 | 0.473 | 0.427 | 0.426 |
| openai_responses | openai_chat | 2 | 0.734 | 0.732 | -0.39% | 0.899 | 0.974 | 0.734 | 0.732 |
| openai_responses | openai_responses | 2 | 1.033 | 1.023 | -0.89% | 1.075 | 1.073 | 1.034 | 1.023 |
| openai_responses | anthropic | 2 | 0.879 | 0.882 | +0.33% | 0.996 | 1.000 | 0.881 | 0.883 |
| openai_responses | middleware | 2 | 0.671 | 0.671 | +0.08% | 0.761 | 0.738 | 0.671 | 0.671 |
| anthropic | openai_chat | 2 | 0.775 | 0.797 | +2.80% | 0.886 | 0.897 | 0.776 | 0.793 |
| anthropic | openai_responses | 2 | 1.010 | 0.966 | -4.37% | 1.173 | 1.036 | 1.010 | 0.966 |
| anthropic | anthropic | 2 | 1.018 | 1.006 | -1.22% | 1.163 | 1.151 | 1.018 | 1.006 |
| anthropic | middleware | 2 | 0.655 | 0.660 | +0.82% | 0.798 | 0.813 | 0.655 | 0.661 |
| openai_chat | openai_chat | 24 | 3.722 | 3.613 | -2.94% | 9.269 | 3.796 | 3.724 | 3.614 |
| openai_chat | openai_responses | 24 | 5.214 | 5.182 | -0.63% | 5.398 | 5.392 | 5.215 | 5.181 |
| openai_chat | anthropic | 24 | 5.553 | 5.566 | +0.23% | 5.686 | 9.004 | 5.554 | 5.567 |
| openai_chat | middleware | 24 | 2.625 | 2.617 | -0.28% | 2.709 | 2.726 | 2.626 | 2.617 |
| openai_responses | openai_chat | 24 | 5.124 | 5.229 | +2.05% | 5.450 | 5.967 | 5.123 | 5.229 |
| openai_responses | openai_responses | 24 | 7.676 | 7.750 | +0.96% | 8.005 | 8.260 | 7.678 | 7.750 |
| openai_responses | anthropic | 24 | 7.122 | 7.106 | -0.23% | 7.780 | 7.242 | 7.123 | 7.108 |
| openai_responses | middleware | 24 | 4.997 | 5.013 | +0.31% | 5.195 | 5.264 | 4.997 | 5.006 |
| anthropic | openai_chat | 24 | 6.031 | 5.955 | -1.26% | 7.938 | 6.219 | 6.030 | 5.955 |
| anthropic | openai_responses | 24 | 7.936 | 7.828 | -1.36% | 22.537 | 21.588 | 7.929 | 7.822 |
| anthropic | anthropic | 24 | 8.915 | 8.846 | -0.78% | 17.005 | 9.052 | 8.916 | 8.848 |
| anthropic | middleware | 24 | 5.695 | 5.680 | -0.27% | 6.319 | 8.395 | 5.697 | 5.681 |

</details>

<details>
<summary>增量缓存：24 场景</summary>

| 上游 | stdio | 历史轮数 | 旧 P50 | 新 P50 | 变化 | 旧 P95 | 新 P95 | 旧 CPU P50 | 新 CPU P50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openai_chat | openai_chat | 2 | 0.361 | 0.346 | -4.07% | 0.377 | 0.363 | 0.362 | 0.348 |
| openai_chat | openai_responses | 2 | 0.389 | 0.369 | -4.94% | 0.477 | 0.380 | 0.390 | 0.370 |
| openai_chat | anthropic | 2 | 0.381 | 0.374 | -1.93% | 0.400 | 0.463 | 0.383 | 0.374 |
| openai_chat | middleware | 2 | 0.317 | 0.307 | -3.43% | 0.332 | 0.341 | 0.320 | 0.308 |
| openai_responses | openai_chat | 2 | 0.536 | 0.514 | -4.28% | 0.570 | 0.536 | 0.538 | 0.514 |
| openai_responses | openai_responses | 2 | 0.622 | 0.594 | -4.42% | 0.662 | 0.608 | 0.623 | 0.596 |
| openai_responses | anthropic | 2 | 0.557 | 0.537 | -3.58% | 0.690 | 0.557 | 0.559 | 0.538 |
| openai_responses | middleware | 2 | 0.552 | 0.527 | -4.38% | 0.578 | 0.555 | 0.553 | 0.528 |
| anthropic | openai_chat | 2 | 0.665 | 0.591 | -11.06% | 0.776 | 0.612 | 0.665 | 0.592 |
| anthropic | openai_responses | 2 | 0.682 | 0.618 | -9.51% | 0.738 | 0.647 | 0.683 | 0.620 |
| anthropic | anthropic | 2 | 0.789 | 0.651 | -17.43% | 0.828 | 0.678 | 0.792 | 0.652 |
| anthropic | middleware | 2 | 0.646 | 0.587 | -9.13% | 0.662 | 0.622 | 0.649 | 0.589 |
| openai_chat | openai_chat | 24 | 1.347 | 1.197 | -11.14% | 1.441 | 1.261 | 1.351 | 1.199 |
| openai_chat | openai_responses | 24 | 1.378 | 1.243 | -9.84% | 1.435 | 1.348 | 1.381 | 1.245 |
| openai_chat | anthropic | 24 | 1.372 | 1.225 | -10.72% | 1.462 | 1.313 | 1.375 | 1.228 |
| openai_chat | middleware | 24 | 1.318 | 1.172 | -11.08% | 1.415 | 1.278 | 1.321 | 1.175 |
| openai_responses | openai_chat | 24 | 3.070 | 2.844 | -7.36% | 3.272 | 2.916 | 3.071 | 2.846 |
| openai_responses | openai_responses | 24 | 3.628 | 3.413 | -5.92% | 3.728 | 3.695 | 3.630 | 3.414 |
| openai_responses | anthropic | 24 | 3.092 | 2.845 | -7.99% | 3.212 | 2.946 | 3.093 | 2.846 |
| openai_responses | middleware | 24 | 3.551 | 3.316 | -6.61% | 4.524 | 3.502 | 3.554 | 3.318 |
| anthropic | openai_chat | 24 | 4.607 | 3.951 | -14.24% | 5.422 | 4.183 | 4.608 | 3.953 |
| anthropic | openai_responses | 24 | 4.794 | 4.075 | -14.98% | 5.325 | 4.822 | 4.795 | 4.077 |
| anthropic | anthropic | 24 | 5.193 | 4.548 | -12.43% | 5.418 | 4.751 | 5.195 | 4.550 |
| anthropic | middleware | 24 | 5.464 | 4.502 | -17.61% | 6.392 | 4.785 | 5.466 | 4.505 |

</details>

<details>
<summary>连续重试：24 场景</summary>

| 上游 | stdio | 历史轮数 | 旧 P50 | 新 P50 | 变化 | 旧 P95 | 新 P95 | 旧 CPU P50 | 新 CPU P50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openai_chat | openai_chat | 2 | 0.376 | 0.344 | -8.38% | 1.563 | 0.401 | 0.377 | 0.345 |
| openai_chat | openai_responses | 2 | 0.520 | 0.362 | -30.39% | 1.575 | 0.414 | 0.510 | 0.363 |
| openai_chat | anthropic | 2 | 0.375 | 0.361 | -3.69% | 0.456 | 0.444 | 0.376 | 0.362 |
| openai_chat | middleware | 2 | 0.311 | 0.301 | -3.11% | 0.357 | 0.344 | 0.312 | 0.302 |
| openai_responses | openai_chat | 2 | 0.532 | 0.517 | -2.94% | 0.606 | 0.581 | 0.533 | 0.518 |
| openai_responses | openai_responses | 2 | 0.615 | 0.592 | -3.77% | 0.673 | 0.673 | 0.616 | 0.592 |
| openai_responses | anthropic | 2 | 0.545 | 0.529 | -2.93% | 0.642 | 0.617 | 0.546 | 0.530 |
| openai_responses | middleware | 2 | 0.552 | 0.520 | -5.84% | 0.613 | 0.580 | 0.553 | 0.520 |
| anthropic | openai_chat | 2 | 0.657 | 0.592 | -9.94% | 0.817 | 0.733 | 0.657 | 0.593 |
| anthropic | openai_responses | 2 | 0.672 | 0.612 | -8.92% | 0.818 | 0.778 | 0.672 | 0.612 |
| anthropic | anthropic | 2 | 0.704 | 0.650 | -7.70% | 0.840 | 2.264 | 0.705 | 0.651 |
| anthropic | middleware | 2 | 0.644 | 0.579 | -10.16% | 0.784 | 0.604 | 0.645 | 0.579 |
| openai_chat | openai_chat | 24 | 1.314 | 1.198 | -8.86% | 1.369 | 1.260 | 1.316 | 1.200 |
| openai_chat | openai_responses | 24 | 1.346 | 1.230 | -8.63% | 1.439 | 1.273 | 1.347 | 1.230 |
| openai_chat | anthropic | 24 | 1.342 | 1.253 | -6.63% | 1.453 | 1.338 | 1.343 | 1.253 |
| openai_chat | middleware | 24 | 1.275 | 1.179 | -7.60% | 1.357 | 1.218 | 1.276 | 1.179 |
| openai_responses | openai_chat | 24 | 3.054 | 2.822 | -7.59% | 3.992 | 2.893 | 3.056 | 2.822 |
| openai_responses | openai_responses | 24 | 3.563 | 3.358 | -5.76% | 3.651 | 3.492 | 3.565 | 3.359 |
| openai_responses | anthropic | 24 | 3.004 | 2.881 | -4.10% | 3.136 | 3.241 | 3.005 | 2.880 |
| openai_responses | middleware | 24 | 3.547 | 3.326 | -6.25% | 3.816 | 3.734 | 3.547 | 3.329 |
| anthropic | openai_chat | 24 | 4.819 | 4.106 | -14.80% | 5.189 | 4.255 | 4.821 | 4.106 |
| anthropic | openai_responses | 24 | 4.755 | 4.129 | -13.16% | 5.002 | 4.714 | 4.756 | 4.130 |
| anthropic | anthropic | 24 | 5.197 | 4.558 | -12.30% | 6.168 | 4.911 | 5.197 | 4.558 |
| anthropic | middleware | 24 | 5.289 | 4.546 | -14.04% | 6.201 | 5.044 | 5.287 | 4.547 |

</details>

### 完整 generate 的尾延迟与 CPU（6 场景）

以下与前述完整调用 P50 表使用相同的 42 样本／版本；P95 易受 GC 影响，保留原值供复核，不单独据此判断尾延迟改善。

| 上游 | 模式 | 旧 P95 ms | 新 P95 ms | 旧 CPU P50 ms | 新 CPU P50 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| openai_chat | 聚合 | 19.872 | 18.861 | 18.311 | 18.155 |
| openai_chat | 流式 | 21.211 | 22.319 | 20.042 | 19.794 |
| openai_responses | 聚合 | 58.143 | 56.514 | 54.529 | 54.235 |
| openai_responses | 流式 | 61.922 | 66.575 | 56.639 | 56.076 |
| anthropic | 聚合 | 62.555 | 36.927 | 35.124 | 33.680 |
| anthropic | 流式 | 37.392 | 36.338 | 36.558 | 35.443 |

### 内存与隔离

嵌套工具参数的捕获计数：新批次由 3 次降至 2 次，重试由 2 次降至 1 次。
回归测试同时修改已准备请求的嵌套参数，确认输入、缓存和后续请求保持独立。

| 上游 | 模式 | 修改前峰值 KiB | 修改后峰值 KiB | 变化 |
| --- | --- | ---: | ---: | ---: |
| openai_chat | 关闭缓存准备 | 878.6 | 878.6 | +0.00% |
| openai_chat | 增量缓存准备 | 172.1 | 121.4 | -29.47% |
| openai_chat | 连续重试准备 | 169.9 | 119.2 | -29.85% |
| openai_responses | 关闭缓存准备 | 1107.2 | 1107.2 | +0.00% |
| openai_responses | 增量缓存准备 | 320.2 | 262.7 | -17.98% |
| openai_responses | 连续重试准备 | 317.8 | 262.1 | -17.52% |
| anthropic | 关闭缓存准备 | 617.1 | 617.1 | +0.00% |
| anthropic | 增量缓存准备 | 564.8 | 356.3 | -36.93% |
| anthropic | 连续重试准备 | 563.2 | 360.8 | -35.94% |

### 验证与复现

- 当次代码验证：全部 2651 项测试通过；Ruff、格式和差异检查通过。

```sh
.venv/bin/python -m tests.performance.preparation --iterations 41 --warmup 7 --json /tmp/cache-copy-preparation.json
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

旧版通过 PYTHONPATH 选择保存的 src 快照，运行顺序为旧／新／新／旧。
开启缓存的完整调用驱动、内存探针、源码差异、环境指纹及所有样本见
[完整原始数据](data/2026-11-06-message-cache-copy.json)。准备驱动见
[preparation.py](../../tests/performance/preparation.py)。

- 基线源码指纹：`6806d122254ca44bcd01437fdff0a74d60772bdb3f6a385c909fb9a9aa8f59ac`。
- 当前源码指纹：`d81a3b1c94df6b2a89b73966418832f6b077e81696b94107c44f30ed50729466`。

### 环境与数据保留

- Python：`3.12.12 (main, Oct 28 2025, 11:52:25) [Clang 20.1.4 ]`；平台：`macOS-27.0.1-arm64-arm-64bit`。
- SDK 与校验依赖：OpenAI `2.54.0`、Anthropic `0.125.0`、Pydantic `2.14.1`。
- 最新 JSON 原样保留 4 轮运行的 72 个准备场景与 6 个完整调用场景、每个计时样本、请求／负载／驱动指纹、9 个内存场景及三次探针、复现驱动和最终源码补丁。
- 所有结果均为本机离线测量，真实 SDK 通过内存 HTTPX 消费响应；不包含远程网络和模型推理。多轮真实服务 E2E 是正确性验证，不纳入性能折线。
- 本次仅整理既有结果并从原始样本汇总表格，没有重新运行性能基准。
