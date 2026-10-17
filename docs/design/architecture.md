# 项目架构与职责边界

2026-10-18 更新 codec 布局与抽象契约。按标准协议输入范围和后续明确要求，
核心行为要求已落地。与最初示例在职责、目录和命名上的差异见
[需求对照报告](../audits/refactor-requirements-2026-10-17.md)。
无损边界为同一 protocol 与中间态之间，不保证跨协议无损。

## 目录职责

```text
src/my_llm/
├── __init__.py                 # 对外导出
├── exceptions.py               # 配置、输入、流错误
├── invoke.py                   # MyLLM、一次性 generate、请求状态编排
├── stream/
│   ├── message_stream.py       # MessageStream 生命周期与标准结果快照
│   └── _state.py               # 每次请求独立的累积状态
├── profile/
│   ├── provider_profile.py     # 连接、统一 protocol、默认配置
│   ├── model_profile.py        # 模型与覆盖项
│   ├── configuration.py        # 配置类型、覆盖字段与字典合并
│   └── resolution.py           # 解析、快照与校验
└── codec/
    ├── converter.py            # MyCodec，五协议注册与转换路由
    ├── base/
    │   ├── codec.py            # BaseCodec
    │   ├── encoder.py          # BaseEncoder
    │   └── decoder.py          # BaseDecoder
    ├── shared/                 # 通用字典、工具关联、匹配、流转换辅助
    ├── anthropic/              # codec.py + encoding/ + decoding/
    ├── gemini/                 # codec.py + encoding/ + decoding/
    ├── ollama/                 # codec.py + encoding/ + decoding/
    └── openai/
        ├── chat/               # 独立 codec.py + encoding/ + decoding/
        └── responses/          # 独立 codec.py + encoding/ + decoding/
```

### 原生协议目录

五种协议使用相同的方向边界。转换算法直接实现为 Encoder / Decoder 的方法，
不再将每个方法拆成函数模块，再通过 staticmethod 绑定到类上。
包初始化文件仅导出，codec.py 只组合两个方向并声明协议身份。

```text
<protocol>/
├── codec.py
├── encoding/
│   ├── __init__.py             # 仅导出 Encoder
│   ├── encoder.py              # 识别、消息、工具、流、用量：原生 → 中间态
│   └── content.py              # 内容块、字段消费、工具调用和 Schema 辅助（按需）
└── decoding/
    ├── __init__.py             # 仅导出 Decoder
    ├── decoder.py              # 中间态 → 原生；请求组装、参数与默认策略
    ├── content.py              # 内容渲染、投影与工具辅助（按需）
    └── transport.py            # 官方 SDK 调用和请求资源清理
```

每个方向的主要转换方法都由基类约束；content.py 中的函数是实现细节，
供本方向的消息与流转换复用，不要求为每个辅助函数新增继承层。
Decoder 的 request_parameters、assemble_request、profile_defaults 和 upstream
也都有明确抽象接口；transport.py 是 upstream 使用的 SDK 资源管理实现。
不同协议之间、同一协议的 encoding 与 decoding 之间均不相互依赖。

encode 的方向是标准协议 → 中间态，decode 的方向是中间态 → 标准协议或上游请求格式，
因此上游参数生成归在 decoding。OpenAI 两套 API 的原生参数也分别处理。
共享 options / transport 辅助仅合并配置、校验通用能力并管理资源，不选择厂商或原生字段。
BaseEncoder.matches 固定整批识别流程，协议只实现 match_item。
BaseDecoder.request 固定能力检查、消息解码、工具解码、原生请求组装的顺序，
直接调用 Decoder.request 也必须传中间态；MyCodec.request 只负责路由。
Profile 解析向目标 Decoder 获取协议默认值和用量模式校验；MyLLM 委托 MyCodec 打开原生流、
处理服务端历史能力和收集结果形状，不直接分支选择 SDK 或端点。

[转换器](../../src/my_llm/codec/converter.py)不引用 MyLLM；完整离线流批次可使用本次调用内的
局部累积器，实时流的状态由 MyLLM 为每次请求创建并持有。
[MessageStream](../../src/my_llm/stream/message_stream.py)只管理迭代、关闭和结果快照，不定义消息格式。
请求累积器使用结构化源引用和显式的独立块追加操作；每块的片段、参数与发布状态
由 `BlockState` 统一管理，`feed()` 只分派操作。详见[流状态设计](stream-state.md)。
`exceptions.py` 取代已经没有消息类型的 types 包；没有旧目录的兼容转发模块。

测试按职责放在 `tests/unit/`、`tests/contract/`、`tests/integration/`、`tests/packaging/`，
可复用样例、传输替身及校验器放在 `tests/helpers/`，conftest.py 只注册 fixture。
文档按 `reference/`、`guides/`、`design/`、`audits/` 分类，见[文档目录](../README.md)。
离线示例放在 `examples/`；模块命名和导入规则见[本地开发规范](../guides/development.md)。
新协议的必需接口、中间态约定和验证步骤见[接入协议指南](../guides/adding-protocol.md)。

## 请求与返回路径

请求：stdio 标准格式 → 源 Encoder（profile=None）→ 内部开放字典 →
目标 Decoder（resolved_profile）→ 官方 SDK 请求。

返回：官方 SDK 流事件 → 源 Encoder（resolved_profile）→ 请求独立累积状态 →
目标 Decoder（profile=None）→ stdio 标准流事件与完成消息。

`ProviderProfile.protocol` 决定上游 SDK/API；`MyLLM.stdio_protocol` 决定调用者的输入和输出。
二者独立，不通过当前 provider 反推调用者格式。纯文本输入存在协议重叠时默认使用
openai_chat；显式设置 stdio_protocol 可选择其他标准格式。

## 逐项验收

| 原需求 | 复核结果与证据 |
| --- | --- |
| BaseCodec / BaseEncoder / BaseDecoder 统一抽象 | 已完成。[base](../../src/my_llm/codec/base/codec.py)定义六个对称方法；五个 Codec 均继承基类。见 [契约测试](../../tests/unit/test_native_codecs.py)。 |
| Encoder / Decoder / Codec 无状态，不绑定 profile | 已完成。协议实例不保存 profile 或片段；profile 按调用传入，None 表示标准协议。见 [契约测试](../../tests/unit/test_native_codecs.py)与[实例隔离测试](../../tests/unit/test_ownership.py)。 |
| 不同签名差异收入 parameters | 已完成。六个转换方法均接受 `**parameters`；previous_response_id 在请求组装时处理。见 [BaseCodec](../../src/my_llm/codec/base/codec.py)与[请求参数测试](../../tests/unit/test_encoding.py)。 |
| MyCodec 同时持有五个 Codec，任意格式转换 | 已完成。[MyCodec](../../src/my_llm/codec/converter.py)只保留协议注册表；消息、工具和完整流批次均验证 25 条转换路径。见 [消息/工具测试](../../tests/unit/test_message_conversion.py)及[流契约测试](../../tests/contract/test_decoding.py)。 |
| 未识别字典保留层级 | 已完成。标准协议数据及扩展的原路径、索引与同协议回放有直接检查；见[保留规则](../audits/lossless.md)和[契约测试](../../tests/contract/test_intermediate_preservation.py)。不要求跨协议无损。 |
| Codec 路由内聚，MyLLM 不依赖具体 Codec | 已完成。[invoke.py](../../src/my_llm/invoke.py)只使用 MyCodec；具体 Codec 注册位于 [converter.py](../../src/my_llm/codec/converter.py)。 |
| 识别整个输入格式一次，绑定 I/O 协议 | 已完成。guess_stdio_protocol 仅在未绑定时识别；后续直接使用对应 Encoder，不逐条重新选择源协议。见 [调用测试](../../tests/unit/test_invoke.py)与[输入集成测试](../../tests/integration/test_message_inputs.py)。 |
| provider / model 构造时复制，并保存 resolved_profile | 已完成。[MyLLM](../../src/my_llm/invoke.py)保存配置副本和解析快照，generate 不接受动态 provider/model/连接覆盖。见 [所有权测试](../../tests/unit/test_ownership.py)与[公开接口测试](../../tests/unit/test_public_api.py)。 |
| 请求与返回分别应用正确的 profile | 已完成。调用者侧使用标准格式，上游侧使用解析配置；兼容服务的文本路径在源编码时生效。见 [调用测试](../../tests/unit/test_invoke.py)及[SDK 集成测试](../../tests/integration/test_sdk_transport.py)。 |
| OpenAI Chat / Responses 目录独立 | 已完成。两者各有 encoding/decoding，互不共用消息与流实现。见 [Chat](../../src/my_llm/codec/openai/chat/codec.py)与 [Responses](../../src/my_llm/codec/openai/responses/codec.py)。 |
| 输入输出使用标准格式，不再自定义消息协议 | 已完成。移除 Message / MessageChunk / ToolCall 等自定义消息类型与 types/messages.py。返回字典或 Responses items，流返回标准事件。见 [公开接口测试](../../tests/unit/test_public_api.py)。 |
| protocol / api_type 统一 | 已完成。[Protocol](../../src/my_llm/profile/configuration.py)使用五个同级值，ProviderProfile / ResolvedProfile 只有 protocol。见 [配置测试](../../tests/unit/test_profiles.py)。 |
| 空思维链只在思考开启且当条有工具调用时补充 | 已完成。Chat / Ollama 同时检查回放配置、思考开关、assistant 角色和工具调用；已有可见思考按回放配置处理。见 [参数组合测试](../../tests/unit/test_encoding.py)。 |
| MyLLM 校验消息间工具 ID 一一对应 | 已完成。标准协议输入中的重复、缺少、未知、顺序错误及可选 ID 情况均校验；见[历史测试](../../tests/unit/test_messages.py)。 |
| 不处理消息内部中断恢复，只对外发布完整工具调用 | 已完成。请求状态只负责正常片段累积，工具参数完整后再生成目标事件，不提供部分 JSON 工具调用或中断修复 API。显式 SDK 错误和缺少消息级终止事件继续上抛。见 [流测试](../../tests/contract/test_decoding.py)。 |
| 移除自定义 client，统一官方 SDK | 已完成。MyLLM 无 client 构造参数，每次请求创建并关闭官方 SDK client；成功、异常、取消和提前退出均覆盖。见 [调用测试](../../tests/unit/test_invoke.py)、[OpenAI SDK](../../tests/integration/test_sdk_transport.py)、[原生 SDK](../../tests/integration/test_native_sdk_transport.py)。 |
| 无旧版兼容路径，文档与分发同步 | 已完成。旧模块已移除，公开导出和示例同步；构建测试验证新模块入包、旧 types 包不存在。见 [文档测试](../../tests/integration/test_documentation.py)与[发行包测试](../../tests/packaging/test_distribution.py)。 |

## 累计重构修复

- 拆开之前混在单文件中的双向转换，统一五种协议的目录与职责。
- 保留工具调用内的未知扩展字典，避免在源编码时丢失或被提升到父级；流式工具发布也保留这些字段。
- 修复同名工具混用显式结果 ID 和按名称匹配时的待完成调用队列。
- 将未知流事件的同协议保留扩大到五种来源；跨协议不透传不属于目标协议的事件。
- 清理无调用方的连接参数动态覆盖逻辑，连接配置统一来自构造时的解析快照。
- 补齐嵌套媒体、原生工具调用包装、流事件的原路径扩展保留；Responses 在目标解码阶段合并输出，源编码保留逐项扩展。
- 针对未知键与转换字段重名明确转换值优先；流操作使用保留命名空间，避免误读原生扩展。

## 最终代码复核

本轮发现并已修复以下问题，回归测试直接覆盖触发场景：

- 冲突字段递归合并会把原列表的尾部文本或原字典的额外工具参数混入转换结果。现在命名空间中的转换字段整体优先，保留值不参与其语义合并。
- 流事件扩展可能覆盖已识别的块类型、工具名称或参数缓冲。已保护转换字段；参数片段存入请求状态的独立字典，不再使用消息块的 `__arguments__` 魔法键。
- 媒体字段消费规则曾混用不同块类型；消息工具调用的 `index` 曾被当作流索引消费。现在按实际类型和消息/流上下文区分，未知字段留在原路径。
- Gemini / Ollama 自动生成的可选工具 ID 可能与显式 ID 重复。生成前预留输入中的显式 ID，保留同协议回放与按名称关联。
- 协议匹配遇到畸形嵌套容器时可能抛出 TypeError / AttributeError。现在返回不匹配；合法的空值调用列表按无调用处理，同协议仍还原 null。

清理项包括：内部保留键集中到 [keys.py](../../src/my_llm/codec/shared/keys.py)，移除重复的嵌套扩展拷贝、未使用的辅助参数和历史校验数据，统一异常导入，清理测试中的旧 api_type 变量名，并修正 Decoder 文档对跨协议字段保留的过度承诺。

先前验证：714 项测试通过；Ruff、格式检查、差异空白检查、离线示例、wheel / sdist 构建及 Twine 严格检查通过。源码未发现旧协议路由、旧消息类型或自定义 client 的运行时兼容路径；未发现剩余的已知需求缺口。

测试使用本地 HTTP transport 驱动真实官方 SDK，没有调用真实模型服务。
签名回放和未知字段的保留不赋予私有字段新的跨厂商语义，目标 SDK/API 仍会校验其支持的字段。
构建与验证命令见[构建指南](../guides/building.md)，使用方式见 [API 指南](../reference/api.md)。

## 官方协议定义复核

2026-10-17 按官方在线定义重新独立编写原生样例，补充 SDK 类型、未知字段拒绝、
延迟内容块验证和事件生命周期检查，修正了 usage、Responses 必填项、Gemini 工具名称
及外来签名/工具字段等偏差。详情、SDK 版本和范围见[协议复核](../audits/protocol-conformance.md)。

## 并发调用验证

五协议上游与五协议 I/O 的 25 组组合分别验证同一实例 8 路混合调用，
另验证五协议取消、失败、提前退出的请求隔离和后续复用。专项共 30 项，
使用真实 SDK、本地 transport 与同步屏障，共执行 230 次调用；同时修复
Gemini 提前退出时 HTTPX 响应体未即时关闭的问题。见[并发复核](../audits/concurrency.md)。

## 独立审计修复与协议归属复核

A01—A11 的修复及后续代码审核已补齐。请求、SDK 传输、格式识别、默认参数、用量、
内容投影、原生字段消费及 stream/result 形状的差异均由对应 Encoder / Decoder 负责。
共享层只保留不区分厂商的机制。详见[修复验收记录](../audits/protocol-fixes-2026-10-17.md)，
其验证结果取代本页较早的累计测试数量。
