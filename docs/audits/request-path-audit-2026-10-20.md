# 五协议请求路径与新增协议契约审计

核对日期：2026-10-20。范围是 Profile 参数标记重构后的当前工作树，
包含本轮冗余拷贝与路径收敛修改。本文替代历史请求路径报告中已撤销的恢复记录与未知字段保留结论。
此前工作区的性能数据见[同日复测报告](performance-retest-2026-10-20.md)，
随后针对未知字段保留删除后的性能回退，定位与修复验证见
[性能回退修复报告](performance-regression-2026-10-20.md)。

## 请求与返回链路

五协议共用下面的编排。同协议输入、跨协议输入，以及流式和收集式结果，均不绕过翻译层。

```text
MyLLM.generate
  → 已绑定 stdio_protocol；未绑定时唯一识别
  → 输入协议 Transport.validate_input
  → 输入协议 Encoder.messages / tools
  → MyCodec.request → BaseCodec.request
      → 上游 Transport.validate_parameters → PreparedOptions
      → 上游 Decoder.request
          → messages / tools → request_parameters → assemble_request(request=...)
      → 上游 Transport.request
          → validate_history → filter_request（消息/工具叶委派给后续钩子）
          → filter_messages / filter_tools → SDK 类型校验
          → validate_request → 恢复 Profile 显式声明的非标准叶字段
  → MyLLM._stream
      → MyLLM 缓存客户端 → 上游 Transport.upstream → SDK(**request)
      → 每个上游事件交上游 Encoder.chunks
      → 请求独立 StreamState.feed
      → stdio 协议 Decoder.chunks → 对外流事件
      → StreamState.finish → stdio 协议 Decoder.messages / result
      → 响应资源关闭 → 最终结果
```

入口证据：[MyLLM.generate 与 _stream](../../src/my_llm/runtime/llm.py)、
[BaseCodec.request](../../src/my_llm/codec/base/codec.py)、
[BaseDecoder.request](../../src/my_llm/codec/base/decoder.py)、
[BaseTransport.request](../../src/my_llm/codec/base/transport.py)。
stream=False 会消费同一个 MessageStream；没有另一套非流式 SDK 执行路径。

| 协议 | 请求位置与组装差异 | SDK 调用与清理差异 | 差异所属位置 |
| --- | --- | --- | --- |
| Chat | messages、tools；extra_body 作为 SDK body overlay；按需请求 stream_options 用量尾帧 | chat.completions.create；响应流 close | Chat Decoder / Transport |
| Responses | input、tools；extra_body overlay；previous_response_id；最终返回 items 列表 | responses.create；响应流 close | Responses Codec / Encoder / Decoder / Transport |
| Anthropic | messages、tools；system 独立组装；max_tokens 必填；原生 thinking 与预算约束 | messages.create；响应流 close | Anthropic Codec / Encoder / Decoder / Transport |
| Gemini | contents、config.tools；system_instruction 和 HTTP headers 放入 config；禁止 SDK 自动执行工具 | generate_content_stream；请求独立 ExitStack 跟踪响应体与迭代器，客户端分别关闭异步和同步接口 | Gemini Codec / Encoder / Decoder / Transport |
| Ollama | messages、tools；options.num_predict；请求头绑定客户端并影响缓存键；工具 Schema 表达能力检查 | chat；响应迭代器 aclose | Ollama Codec / Encoder / Decoder / Transport |

这些是协议和 SDK 的实际差异，不需要再增加公共执行分支。
共享的 Profile、MyLLM、StreamState 不导入原生实现来决定字段或 SDK 行为。
MyCodec 仅在内置注册处引用五种 Codec，正常执行按注册表路由。

## 公共机制中只由部分协议使用的分支

仅搜索协议名不足以证明收敛。本次同时检查了公共算法读取的字段、原生操作的生产者和消费方。

| 公共机制 | 目前的主要生产者 | 是否仍含协议专属规则 |
| --- | --- | --- |
| 嵌套 message_path / tool_path | Gemini 的 config.tools | 路径由 Transport 声明，公共层只沿字典路径操作。 |
| snapshot 与实时块协调 | Responses 终态 output | Encoder 生成统一 BlockRef 与已投影内容；共享状态机仅检查块身份、遗漏与已发文本前缀。 |
| coalesce_messages | Responses 相邻 assistant items | Encoder 标记 coalesce，其他目标 Decoder 显式消费；公共函数只合并已标记且来源、角色一致的中间态。 |
| allow_external_calls | Responses 的服务端历史引用 | Transport 声明能力；公共历史检查仍拒绝重复结果与未完成本地调用。 |
| block_stop、block_annotation、reason 与 finish 分开 | Anthropic 的内容块、引用增量与消息终止 | 原生事件识别和字段投影留在 Encoder；公共层仅执行块生命周期和已识别内容操作。 |
| 终态后的 allow_tail | Chat 的用量尾块、Gemini 的元数据尾块 | 允许的操作类型由源 Encoder 声明，不按厂商名判断。 |
| metadata 的嵌套更新 | Responses 的 item 头和响应设置；独立第六协议的 trace | 已改为通用 path / values 操作，共享层不再识别协议字段。 |

原实现的 StreamState._metadata 虽然没有协议名判断，仍固定认识 response_items、response_parameters
及其子字段；因此当时不能声称所有元数据差异均已收敛。
本轮将 ID 归一化、字段投影、原生项头和设置的更新位置移回五家 Encoder。
共享状态机现在仅消费 `metadata(values, path=())`：沿路径复制必要容器，捕获本次 values，浅合并目标字典。
完整终态通过更新父字典来替换完整子树；原本未被消费的 merge 标记已移除。
values 只装入 Encoder 已识别的字段，不装原始事件，也不恢复未知字段保留。

SDK overlay 的字段名和空 body 省略行为也已改由原生 Decoder 声明，
共享 options 不再根据 extra_body 名称推断某家 SDK 的封装方式。

## 冗余逻辑与保留的必要边界

| 本轮删除或收敛的行为 | 保留的行为及原因 |
| --- | --- |
| select、arguments(dict) 及部分原生内容转换内部重复复制嵌套 payload | messages / tools 公开出口捕获结果；两个方向的流钩子捕获实际借入的可变字段，直接返回独立事件。整条 generate 仍有不同所有权边界。 |
| 每个流事件在 Encoder / Decoder 出口再次复制完整新建外壳 | `_chunks` 明确承担结果所有权；开放工具参数、元数据、logprobs 字节列表等仍在借入处捕获。内部只读字段投影可以借用开放字典。 |
| Chat / Responses 渲染已投影内容前无用的再次 select | 输出仍由明确支持的字段构造，未知字段不进入公开结果。 |
| Ollama 已过滤数据的再次投影、options 再次过滤；Anthropic system / thinking / output_config 的重复过滤 | 各自 SDK 所需的 Image 包装、原生配置校验与工具 Schema 能力检查仍归 Transport。 |
| 显式 stdio_protocol 时构造历史加工具的协议识别候选列表 | 未指定协议时仍进行整批唯一识别；模糊输入继续报错。 |
| StreamState 内协议项头和响应设置的第二套字段清单 | 原生 Encoder 显式投影，共享状态机只实施统一更新。 |
| Union 为每个内容块遍历全部候选 | 编译已声明 Literal 的候选索引，排除不匹配分支；剩余候选继续按原评分和声明顺序选择。 |
| 根请求与消息/工具钩子重复投影同一叶节点，包含 Gemini config.tools | 根计划只借入被委派的消息/工具叶，后续原有钩子投影并建立独立副本一次。其他配置容器仍完整过滤、捕获。 |
| SDK `_ModelView` 对同一字段重复读取属性、重建嵌套列表与子视图 | 每次转换的视图缓存字段值、缺失标记及包装值；存在性检查不创建子视图，下次转换重新读取源对象。 |

以下行为仍有实际用途：

- 输入 Transport 验证原生输入，输出 Transport 验证翻译后的目标请求；它们验证不同的数据与 SDK 契约。
- SDK TypedDict 的惰性 Iterable 校验仍需要 dump_json 强制执行嵌套验证，不能只检查最外层。
- request、消息与工具的公开过滤结果必须隔离可变调用方数据；开放应用字典保留其键值。
- StreamState 是公开可输入中间操作的累积器，需要捕获输入内容与元数据，防止调用方事后修改状态。
- None、字段缺失和 omit 仍有不同语义；标准协议中的合法可选字段、跨协议表达能力与 SDK 缺省值仍需要明确处理。
- BlockRef 的列表归一化保留 JSON 往返能力；它不用于猜测来源字段或恢复未知内容。

此前记录的 Gemini `config.tools` 重复投影已经修复。Transport 的根请求过滤沿
`message_path` / `tool_path` 生成不可变 `delegated_paths`：只委派对应的消息/工具叶节点，
外层 `config` 等容器仍按 SDK 声明过滤、复制，随后原有 `filter_messages` / `filter_tools`
完成叶节点投影并继续校验。覆盖钩子仍执行，公开过滤结果的隔离也继续保留。

全局类型计划缓存按 Schema 和 `delegated_paths` 分区；默认过滤与委派过滤、
同类型位于不同路径时的规则互不污染。`_FilterPlan` 的四个槽位为 `apply`、`project`、`score`、
`literals`，其中 Literal 索引只排除明确不匹配分支，保留重叠判别值、无限制候选及同分时的声明顺序。
这些计划只保存类型规则。`_ModelView` 另存当前转换实际读取的字段数据，随转换释放，
不能将它理解成跨请求缓存；未知且未配置字段仍不读取、不枚举、不调用整模型序列化。

没有为短小的原生请求信封组装再增加抽象层；同名内容字段也没有跨协议相互导入实现。

## 抽象契约与第六协议证明

逐一核对运行时消费端后，当前请求与返回模型所要求的接口均由基类声明或提供默认实现。

| 基类 | 必需接口 |
| --- | --- |
| BaseCodec | protocol_or_api_style、defaults、encoder、decoder、transport |
| BaseEncoder | protocol、match_item、_messages、_tools、_chunks、usage |
| BaseDecoder | protocol、_messages、_tools、_chunks、usage、assemble_request |
| BaseTransport | request_schema、message_path、tool_path、message_type、tool_type、validate_request、create_client、upstream |

后续配置重构将 Decoder.request_parameters 固定为 final 基类流程，由 Decoder.request 统一调用一次，
assemble_request 接收已有 request 字典并填入消息和工具，不再接收 PreparedOptions。

validate_parameters、validate_input、filter_messages / filter_tools、validate_profile、client_key、
close_client、allows_external_calls 和 Decoder.result 有可覆盖的默认实现。
例如默认结果是首条消息，Responses 自己覆盖为 items 列表；这不是缺少接口声明。

本次审计时，ProtocolDefaults、ProviderProfile 和 ModelProfile 通过可继承的 dataclass 声明
ParameterPath / ParameterDefaultV / Config；解析器发现继承字段并检查分类、配对和路径语法。
后续配置重构已由 `Parameter` 集中声明路径、默认值和阶段，当前接口见[配置指南](../reference/configuration.md)。
Transport 依据 request_schema 检查 SDK 根参数。
新协议的默认值属于自己的 Codec.defaults，不要求改共享默认值表。

本轮修复两个约束缺口：

1. 注册原来只拒绝实例 __dict__，带 `__slots__=("pending",)` 的有状态组件仍能注册。
   现在检查完整 MRO 的数据槽，直接声明及从父类继承的非空 slots 均被拒绝，覆盖 Codec 和三个组件。
2. 依赖边界测试原来只识别 encoding、decoding 与 transport，遗漏协议 utils。
   现在按注册 Codec 所属包识别完整协议目录，utils 与 facade 也进入跨协议依赖和 SDK 客户端归属检查。
   编码/解码互不依赖的方向检查仍限定在对应方向目录。

[独立第六协议测试](../../tests/integration/test_protocol_registration.py) 从四个基类直接实现新协议，
不继承现有五家的原生实现，覆盖：

- 注册和创建时快照；自有 defaults、usage_mode、request_schema 和字段路径。
- 新协议作为上游时，自己的格式与五种内置 stdio 格式，分别使用流式和收集模式。
- 五种内置协议作为上游、新协议作为 stdio 时的流式和收集结果。
- 原生请求字典、工具声明、参数默认值隔离、响应关闭与客户端关闭。
- 自己显式识别 trace.token / trace.phase，使用 metadata 路径操作累积并输出；
  未配置相邻字段的 deepcopy 会直接报错，测试证明它未被读取或复制。

抽象基类约束的是可调用接口，不能独自保证语义正确。Python 的 final 是静态标记，
方法签名、只读借用输入、标准字段投影、完整终态和资源关闭仍必须由实现及契约测试验证。
第六协议证明当前中间内容和操作模型内的新协议可以接入；未来协议引入当前中间态没有表达的新语义时，
仍需先扩展相应内容或操作契约，不能承诺所有未来协议的全部功能无需公共模型调整。

## 此前审计验证

请求链、注册、边界、流状态、字段投影、协议原生结构和所有权相关测试：

```sh
.venv/bin/python -m pytest -q \
  tests/integration/test_protocol_registration.py \
  tests/contract/test_codec_boundaries.py \
  tests/contract/test_stream_state.py \
  tests/contract/test_stream_materialization.py \
  tests/contract/test_translation_boundaries.py \
  tests/contract/test_protocol_conformance.py \
  tests/contract/test_audit_regressions.py \
  tests/unit/test_copy_ownership.py
```

结果：**797 passed in 5.66s**；相关源码与测试 Ruff 检查、格式检查通过。
该数字属于此前审计时的工作区和上述测试范围，不代表后续修复的最终验证数量。
全量测试、构建与本地官方 SDK 性能数据由[同日复测报告](performance-retest-2026-10-20.md)统一记录。
新增 Literal 剪枝、委派路径缓存分区、覆盖钩子及 SDK 视图局部缓存的契约验证，
与完整性能复测，统一记录在[性能回退修复报告](performance-regression-2026-10-20.md)。
本审计与验证不请求远端模型。
