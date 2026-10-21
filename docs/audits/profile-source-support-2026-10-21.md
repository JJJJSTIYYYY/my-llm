# Profile 固定身份与参数支持状态重构

本页性能数据对应逻辑复核前的采样快照。后续修复与验证见[逻辑复核记录](profile-source-support-review-2026-10-21.md)；修复后未重新测量性能。

验证日期：2026-10-21。基线为本轮动工前保存的完整 src/tests 工作区快照，包含用户已有的未提交修改，
不是直接使用 HEAD。重构保留已有修改，没有提交或推送。

## 行为变化

- 删除 Parameter.phase。Parameter.source 固定为 input（用户输入／API 请求）或 output（API 返回）。
- Parameter.name/source 与 Config.name 不接受构造参数；Profile 字段及 metadata 固定身份，
  @set_attr 初始化时检查，允许复用同一身份的声明，拒绝跨字段重新绑定。冻结对象禁止二次赋值。
- Parameter.supports 统一管理参数支持状态。False 强制 default_value=omit，不参与自动请求路由或返回字段读取。
  reasoning_enable.supports 替代 supports_reasoning，reasoning_effort.supports 替代 supports_reasoning_effort。
  请求不支持的 reasoning 报错；普通禁用路由保留手动原生字段，与 omit 的既有行为一致。
- replay_reasoning_content_when_tool_calls、reasoning_effort_mapping 等行为配置保留 Config。
  supports_stream、supports_function_calling、supports_message_name 控制协议能力和处理分支，当前没有对应的可路由 Parameter，也保留 Config。
- Profile 仍按 Codec.defaults → Provider → Model 解析；None 继承，omit 禁用，Parameter 整份替换，Config 字典按键合并。
  解析时复制可变默认值，并保留已绑定身份；Schema 缓存不持有 Profile 或请求数据。
- 自动生成的 token 控制、reasoning_enable 配置值、reasoning_effort 映射值优先于同路径的原生请求字段。
  普通 Parameter.default_value 仍只补缺。修正旧逻辑中原生字段覆盖生成 reasoning 开关的问题，并覆盖 SDK extra_body overlay 的别名路径。

## 性能方法

使用同一解释器、SDK 和 tests/performance/request_chain.py，真实官方 SDK 读取内存 HTTP 响应。
60 个场景包含 5 个原生协议 × 2 种返回模式，以及 5×5 协议组合 × 2 种返回模式。
每场景预热 2 次、采样 15 次；4 路并发通过 asyncio.Barrier 确认响应读取实际重叠，并检查响应关闭与请求头隔离。

完整 generate 计时包含输入转换、请求准备、SDK 解析、结果累积、响应清理与结果副本；
不包含 Profile／实例构造、测试断言、网络或服务器延迟。默认 GC 开启。
内存探针独立于计时，用 tracemalloc 测 Python 分配，不等于进程 RSS。

Profile 声明创建和配置解析由新增 tests/performance/profile_setup.py 分开测量；
MyLLM 构造沿用 tests/performance/construction.py，验证不会创建 SDK 客户端。
原始报告保留源码与测量脚本 SHA256、环境和逐次样本。Profile fixture 只做新声明接口的等价迁移。

## 结果

已核对全部 60 个场景的请求字节数、响应字节数、上游事件、输出事件、HTTP 分块、输入消息及工具数量一致。
四路并发 peak_active=4、SDK 客户端数量一致，预热后的 Schema 缓存大小在探针中保持稳定。

下表为各场景相对变化百分比的中位数；负数表示降低，不是把所有场景耗时相加后的比例。

| 场景 | 数量 | 串行 P50 | 四路批次 P50 | 串行峰值内存 | 四路峰值内存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原生，收集式 | 5 | -2.31% | -1.50% | 0.00% | 约 0% |
| 原生，流式 | 5 | -1.23% | -1.33% | 0.00% | 约 0% |
| 协议组合，收集式 | 25 | -1.56% | -1.80% | 0.00% | 约 0% |
| 协议组合，流式 | 25 | -1.27% | -1.54% | 0.00% | 约 0% |
| 全部 | 60 | -1.58% | -1.64% | 0.00% | 0.00% |

主矩阵单项串行 P50 最大升幅 3.39%，并发 P50 最大升幅 4.03%。
小幅变化可能包含系统负载、分配器和 GC 波动；这些数据支持请求期没有明显整体回退，不能据此承诺固定比例的加速。

### 创建与解析成本

按 before → after → after → before 交替测量，避免只做单方向前后对比。
每协议 Profile 创建／解析每轮预热 100 次、测 50 批 × 100 次，表内合并每版本 100 批样本后取中位数；
MyLLM 构造每轮预热 10 次、测 1000 次，合并每版本 2000 次。
下表单位为微秒。Profile 创建包括一组 ProviderProfile + ModelProfile 及其 Parameter/Config；
解析与 MyLLM 构造使用预建 Profile，不包含 Profile 创建，三个边界不能全部相加（构造已包含解析）。

| 协议 | Profile 创建：前 → 后 | 解析：前 → 后 | MyLLM 构造：前 → 后 |
| --- | ---: | ---: | ---: |
| Chat | 6.15 → 16.05 | 37.23 → 34.81 | 70.08 → 67.92 |
| Responses | 6.18 → 15.83 | 38.84 → 36.43 | 72.67 → 70.17 |
| Anthropic | 6.27 → 15.86 | 39.11 → 36.32 | 72.46 → 70.04 |
| Gemini | 6.21 → 15.83 | 38.75 → 36.17 | 81.67 → 78.00 |
| Ollama | 6.28 → 15.92 | 37.17 → 34.66 | 71.54 → 69.54 |

固定身份绑定需要复制未绑定 wrapper，Profile 创建增加约 9.6–9.9µs（约 +153% 至 +161%），
属于一次性声明成本。解析降低约 6.2%–7.1%，预建 Profile 下的 MyLLM 构造降低约 2.8%–4.5%。
按独立测量的中位数估算，Profile 创建加 MyLLM 构造的总启动成本增加约 6.0–7.7µs，约 +6.8% 至 +10.1%。

独立 tracemalloc 探针每次保留 100 组 Profile／ResolvedProfile：Profile 声明保留内存每组增加约 710 字节；
ResolvedProfile 每份增量约 0–80 字节。该数据包含探针的列表和分配器记账，不能当作对象精确大小。
这些成本没有转移到每次 generate 的循环中。

### 尾延迟复核

主矩阵 Anthropic 原生收集 P95 从 41.26ms 升至 75.39ms（+82.71%）；
Anthropic → Gemini 流式 P50 从 37.31ms 升至 38.58ms（+3.39%）。
因此对这两个场景各按 before → after → after → before 复测，每轮 150 次、每版本合计 300 次，
复用 tests/performance/gc_diagnostic.py 在计时区间记录 GC 回调。回调会增加开销，诊断数值不能替代无回调主矩阵。

| 场景 | 前 P50 | 后 P50 | 前 P95 | 后 P95 | 每版本样本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Anthropic 原生收集 | 39.915ms | 39.853ms | 70.503ms | 70.281ms | 300 |
| Anthropic → Gemini 流式 | 36.570ms | 36.571ms | 37.989ms | 37.639ms | 300 |

原生收集两版都出现 18 次第 2 代回收，18 个超过 60ms 的样本全部伴随超过 10ms 的 GC，
最大单次 GC 停顿分别为 33.56 / 33.22ms。
跨协议流式两版都出现 12 次第 2 代回收；最大 GC 停顿分别为 33.80 / 33.51ms。
P95 的明显上升和单项 P50 回退在交替长采样中均未持续复现。
请求期应按基本持平评估，主矩阵约 1%–2% 的改善不能外推为固定加速收益。

### 原始数据与复现入口

- [主矩阵：前](data/profile-source-support-2026-10-21-before.json)、[主矩阵：后](data/profile-source-support-2026-10-21-after.json)
- [汇总、逐场景变化、创建成本与 GC 复核](data/profile-source-support-2026-10-21-summary.json)
- 同目录 setup-1-before / setup-2-after / setup-3-after / setup-4-before 保存创建／解析原始批次。
- construction-1-before 至 construction-4-before 保存交替构造样本；tails-1-before 至 tails-4-before 保存诊断样本。
  中间两轮均为 after。汇总 JSON 包含本轮诊断适配脚本原文与数据来源的 SHA256。

```bash
python -m tests.performance.request_chain --iterations 15 --json /tmp/profile-chain.json
python -m tests.performance.profile_setup --json /tmp/profile-setup.json
python -m tests.performance.construction --iterations 1000 --json /tmp/profile-construction.json
```

使用 PYTHONPATH 选择对应 src 与 tests 快照，可对两版执行同一测量入口；
更换快照时使用 python -P，避免当前工作目录的包优先于指定快照。



## 验证

覆盖固定 name/source、拒绝构造注入与跨字段绑定、dataclasses.replace、pickle、可复用声明、
自定义字段 source、client 参数来源、supports 布尔校验、False 强制 omit、
五协议输出字段禁用、reasoning 能力控制、生成控制优先级和原始输入所有权。
同步更新 README、配置指南、公共 API 示例、协议接入指南与 SenseNova 示例；本轮没有调用付费模型接口。

最终离线回归：2049 项通过；Ruff lint/format、git diff --check、16 个文档 Python 代码块解析、
源码及示例编译、sdist/wheel 构建、Twine 检查与隔离 wheel 导入均通过。
