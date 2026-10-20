# 请求链路重构与同负载性能对照

验证日期：2026-10-21。范围为当前工作树，重构前基线是动工前保存的 src 快照；
当时只有已有的版本号修改，保留该修改，两版使用同一个解释器和已安装 SDK。
原始源码、脚本 SHA256、Git revision、Python / 平台 / SDK 版本保存在数据中。
本次未提交或推送。

## 请求链路的变化

此前 generate 自行规范化、验证并编码消息，复制一份生成参数，再由 Transport 准备选项，
Decoder 组装原生请求，Transport 再过滤、验证并重新读取连接设置。

现在 generate 将完整输入打包为字典，连同 ResolvedProfile 交给 Codec：

```text
完整调用字典 + ResolvedProfile
  → 输入 Encoder.request：验证输入，生成本次中间字典
  → 上游 Decoder.prepare：每次准备一次选项，映射路径、默认值和覆盖规则
      → 消息／工具翻译 → 原生请求组装 → Schema 过滤与验证
      → {client: SDK构造参数, request: SDK调用参数, local: 本地控制项}
  → MyLLM 延迟取得／复用客户端
  → Transport.upstream(client, request)：官方 SDK 调用及本次响应资源清理
```

连接配置在 resolve_profiles 中验证一次，在 Decoder.client_parameters 中按实例映射一次。
每次调用只准备 body、headers 和生成控制；Ollama 按 headers 生成独立客户端参数及复用键，
不会临时修改共享客户端。客户端仍由 MyLLM 缓存、统一关闭。
输入和输出分属两个协议边界，分别验证；没有取消 SDK 的嵌套 Iterable 验证。

连接 Parameter 通过字段 metadata 的 scope 声明，解析器一次建立 request_parameters 与 connection。
通用请求过滤不再按 api_key 字段名排除认证；测试使用不同字段名的自定义认证声明验证这一点。
已有 None / omit、推理路径、默认值、保护字段和 previous_response_id 规则保持。

删除 generate 的参数整份预拷贝。Decoder 内部直接使用只读翻译钩子，
由最终 SDK Schema 边界捕获可变消息和工具，避免先复制公开转换结果再复制请求。
根计划委派消息／工具叶，Anthropic 开放 input_schema 也先委派，再复制完整 JSON Schema 一次。
离线公开转换、PreparedOptions 的独立使用、结果副本仍有各自必要的所有权边界。

## 基准与计时边界

主矩阵使用同一份 [request_chain.py](../../tests/performance/request_chain.py)，
复用已有 runner、workloads、validation 和真实官方 SDK，经内存 HTTP 响应执行，保持默认 GC。
包含 10 个复杂原生场景与 50 个协议组合场景，各含收集式／流式；每场景预热 2 次、采样 9 次。
负载为 12 轮历史、12 个工具、每轮 4 个调用、2048 字符输入、8192 字符输出、1024 字节 HTTP 分块。

串行计时从 generate 入口到完整结果、响应清理及结果副本；构造／关闭实例、断言不计时。
四路并发额外预热 2 批、采样 9 批；每个响应体在首字节前经过 Barrier，确认 peak_active=4。
每路使用不同 x-request，检查请求头隔离、结果完整、响应关闭及客户端创建次数。
Ollama 使用四个 header 变体客户端，其余协议共享一个，重构前后完全相同。

tracemalloc 在独立的串行和并发探针中运行，启用前 gc.collect，计时采样不启用追踪。
它测 Python 分配峰值，包含返回结果和测试观察值，排除预制负载及已经创建的客户端；不等于 RSS。
构造开销另外测量：每协议预热 10 个实例、采样 200 个，断言没有创建 SDK 客户端。

已核对全部 60 个场景的请求字节、响应字节、上游事件、输出事件、HTTP 分块、输入消息及工具数量一致。
不是通过减少请求内容、返回事件或跳过校验得到的低耗时。

## 主矩阵结果

下表为各场景相对变化百分比的中位数；负数表示降低。完整逐项数值见汇总 JSON。

| 场景 | 数量 | 串行 P50 | 四路批次 P50 | 串行峰值内存 | 四路峰值内存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原生，收集式 | 5 | -1.15% | -1.40% | -7.43% | -15.03% |
| 原生，流式 | 5 | -4.42% | -0.87% | -0.01% | 约 0% |
| 协议组合，收集式 | 25 | -4.22% | -5.24% | -6.49% | -13.64% |
| 协议组合，流式 | 25 | -5.15% | -5.08% | -0.01% | 约 0% |
| 全部 | 60 | -4.32% | -4.88% | -2.99% | -3.89% |

流式准备阶段 P50 的场景中位变化为 -10.31%。主矩阵单项串行 P50 最大升幅 3.70%，
并发 P50 最大升幅 1.21%。这些结果未显示持续的明显请求耗时回退；流式峰值主要由响应累积决定，变化很小。

## 原生场景交替复测与尾部核查

9 个样本会让 Gemini 的隔次 GC 双峰分布改变中位数所在位置，因此补充 before → after → after → before
四轮原生复测，每轮预热 2 次、采样 30 次。下表合并每个版本的 60 个样本，单位毫秒：

| 上游／模式 | 前 P50 | 后 P50 | P50 变化 | 前 P95 | 后 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chat 收集 | 22.385 | 21.998 | -1.73% | 52.665 | 23.297 |
| Chat 流式 | 22.315 | 22.139 | -0.79% | 25.889 | 22.929 |
| Responses 收集 | 59.345 | 58.850 | -0.83% | 61.610 | 91.913 |
| Responses 流式 | 58.597 | 58.627 | +0.05% | 59.913 | 59.774 |
| Anthropic 收集 | 42.197 | 40.986 | -2.87% | 43.814 | 41.867 |
| Anthropic 流式 | 42.052 | 40.992 | -2.52% | 75.684 | 43.063 |
| Gemini 收集 | 47.278 | 46.035 | -2.63% | 85.510 | 83.485 |
| Gemini 流式 | 46.601 | 45.390 | -2.60% | 84.744 | 84.490 |
| Ollama 收集 | 15.639 | 14.350 | -8.24% | 51.070 | 15.044 |
| Ollama 流式 | 15.586 | 14.392 | -7.66% | 50.985 | 15.068 |

保留 Responses 收集 P95 升高这项原始观察。随后对同一场景按 ABBA 顺序各测 150 次，
合计每版本 300 次；通过 gc.callbacks 记录计时区间内的停顿，不禁用或改变 GC 阈值。
长测的前／后 P50 为 58.808 / 57.991，P95 为 60.568 / 58.808，均值为 60.235 / 58.078。
基线出现 10 次第 2 代回收，最大停顿 34.700ms；重构版本轮计时区间没有第 2 代回收，最大停顿 0.870ms。
基线 11 个超过 80ms 的样本中，10 个伴随超过 20ms 的 GC；重构版没有超过 80ms 的样本。

初始 P95 升高未在长测复现，结果对样本分位和 GC 相位敏感。GC 回调本身会增加开销，
这些数值用于定位，不能替换无追踪主基准，也不构成任意负载下 P95 恒定的保证。
正常断言会在计时区间外分配对象，GC 可能落入下一条请求；没有用扣除 GC 后的时间作为最终成绩。

## 实例准备与缓存

| 实例构造 | 前，微秒 | 后，微秒 | 一次性增加 |
| --- | ---: | ---: | ---: |
| Chat | 68.375 | 70.000 | 1.625 |
| Responses | 68.480 | 73.667 | 5.187 |
| Anthropic | 68.375 | 73.250 | 4.875 |
| Gemini | 69.875 | 81.875 | 12.000 |
| Ollama | 68.417 | 72.333 | 3.916 |

SDK 连接映射提前到了实例构造，因此构造略增；请求基准排除这段时间，两项分别报告。
相对于本次 14–59ms 的典型请求耗时，一次性增加为 0.002–0.012ms，不是每次调用的重复开销。

Schema 缓存继续使用 shared/transport.py 的有界 LRU，仅类型／规则被缓存。最后一个场景的串行内存探针完成时快照如下（之后该场景的并发样本不在此快照内）：

| 缓存 | 前 currsize | 后 currsize | 说明 |
| --- | ---: | ---: | --- |
| fields | 380 | 380 | SDK 字段分析。 |
| TypeAdapter | 13 | 13 | 两版总 hits 均为 16805，保留输入与输出验证。 |
| parameter_types | 5 | 5 | SDK 方法类型分析。 |
| _compiled_plan | 34 | 35 | 唯一新增为 Anthropic 开放 input_schema 委派规则。 |
| _declared_path | 21 | 21 | SDK 类型中的路径规则。 |

各类 maxsize 均为 512。全部 60 个预热后探针的 misses、currsize 均未增加；
请求内容没有成为缓存键、缓存值或上次 Union 分支。弱引用回收、动态类型、递归 Schema、Union 顺序、
开放字典、不可哈希 Annotated 和有界淘汰仍由原有缓存测试覆盖。

## 验证与扩展接口变化

完整测试 2020 passed；Ruff 规则与 144 文件格式、协议参考离线校验（24 Schema / 59 模板 / 90 帧）、
两个本地示例、sdist / wheel 构建、Twine strict 和隔离 wheel 导入均通过。
测试包含相同 SDK 上的同／跨协议、流式与收集、并发屏障、取消、提前关闭、
失败隔离、默认值、None / omit、签名／密文回放、媒体、工具 Schema、缓存及发行包导入。
新增测试验证完整输入字段清单、连接只准备一次、每调用选项只准备一次、三个目的地和开放 Schema 单次复制。

扩展协议需把原 Transport 的 Schema、验证、过滤钩子移到 Decoder，并在 Encoder 声明输入消息／工具类型。
Transport 改为 create_client(parameters)、upstream(client, request)；不再接收 Profile 或生成选项。
公开 MyLLM.generate 签名和返回结构不变，离线 MyCodec.request 仍返回原生请求字典。
接口说明见[接入指南](../guides/adding-protocol.md)，可运行示例见[请求准备](../../examples/request_pipeline.py)。

## 复现与原始数据

```sh
# baseline 是改动前保存的 src；两版使用同一基准、依赖和规模，顺序运行。
PYTHONPATH=/tmp/baseline/src python -m tests.performance.request_chain --iterations 9 --json /tmp/before.json
python -m tests.performance.request_chain --iterations 9 --json /tmp/after.json

# 原生交替复测按 before / after / after / before 各运行一次。
python -m tests.performance --suite native --iterations 30 --json /tmp/native.json
python -m tests.performance.construction --iterations 200 --json /tmp/construction.json
python -m tests.performance.gc_diagnostic /tmp/responses-gc.json
```

- [主矩阵基线](data/request-chain-2026-10-21-before.json)与[重构版](data/request-chain-2026-10-21-after.json)。
- [逐项汇总及构造结果](data/request-chain-2026-10-21-summary.json)。
- [原生 ABBA 原始样本](data/request-chain-2026-10-21-native-abba.json)。
- [构造原始样本](data/request-chain-2026-10-21-construction.json)。
- [Responses GC 长测原始样本](data/request-chain-2026-10-21-gc.json)。

基准没有 socket、真实服务器等待或模型推理，不用这些数字推断线上首 token、网络吞吐或 RSS。
后续扩展协议与依赖升级仍需用同样负载复测，不能把本次有限样本当作硬性延迟上限。
