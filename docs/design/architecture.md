# 项目架构与职责边界

2026-10-19 更新。Encoder / Decoder 负责翻译与信息保留，Transport 负责请求验证、
字段过滤及官方 SDK 执行。请求参数映射和原生请求结构由 Decoder 翻译。
历史重构记录见[需求对照报告](../audits/refactor-requirements-2026-10-17.md)。

## 目录职责

```text
src/my_llm/
├── __init__.py                 # 仅导出公共接口
├── invoke.py                   # 生成编排、SDK 客户端缓存
├── registry.py                 # 协议声明注册及快照
├── stream/                     # 请求独立的流累积、结果与资源关闭
├── profile/                    # Provider / Model 配置、合并与快照
└── codec/
    ├── converter.py            # MyCodec，协议注册与转换路由
    ├── base/
    │   ├── codec.py            # 组合翻译与请求流程
    │   ├── encoder.py          # 原生 → 中间态、未修改结果的恢复
    │   ├── decoder.py          # 中间态与生成选项 → 原生请求、保留转换剩余信息
    │   └── transport.py        # 请求验证、过滤及 SDK 生命周期
    ├── shared/
    │   ├── conversion.py       # 中间态、保留键、消息 / 工具翻译及来源分组
    │   ├── options.py          # 配置路径、生成选项翻译与独立校验辅助
    │   ├── replay.py           # 随值保存的恢复记录、修改检测与工具 ID 恢复
    │   ├── stream.py           # 原生事件操作、块身份与用量计数
    │   └── transport.py        # SDK 字段过滤、工具关联验证与响应资源关闭
    ├── anthropic/
    ├── gemini/
    ├── ollama/
    └── openai/
        ├── chat/
        └── responses/
```

### 原生协议目录

每种协议包含 `codec.py`、`transport.py`、`encoding/` 和 `decoding/`。
两个方向的主要实现放在 `encoder.py` / `decoder.py`，内容辅助放在本方向的 `content.py`。
Codec 统一组合三个无状态组件；实际 SDK 客户端由 MyLLM 实例持有。

| 组件 | 职责 |
| --- | --- |
| BaseEncoder | 识别格式；翻译消息、工具、流事件与用量；恢复未修改的转换分组。 |
| BaseDecoder | 翻译消息、工具、流事件、用量和生成选项；组织原生请求结构；保留扩展及原生恢复数据；决定收集结果形状。 |
| BaseTransport | 验证输入格式、工具关联、模型能力及生成参数；返回请求独立的已解析选项；过滤不支持字段并校验原生结构；管理客户端创建、关闭和请求执行。 |
| BaseCodec | 声明协议身份、ProtocolDefaults 和三个组件，统筹翻译、验证及请求流程。 |
| MyCodec | 持有注册快照，路由转换与请求；内置五种协议，允许启动时注册新协议；完整离线流使用本次调用内的累积器。 |
| Profile / resolve_profiles | 声明配置和覆盖规则，合并协议、供应商与模型配置，校验通用字段并建立独立快照。 |
| MyLLM | 持有配置快照与 SDK 客户端缓存，编排请求的翻译、执行、累积和输出。 |
| StreamState / BlockState | 按请求独立维护块身份、内容、签名、工具参数和终态；不持有 SDK 客户端。 |
| MessageStream | 包装本次请求的异步生成器，提供迭代、完成结果副本和响应关闭；独立 generate 的流还负责临时实例的关闭回调。 |

Encoder / Decoder 不承担请求参数验证、字段过滤或 SDK 执行。
Decoder 的 request_parameters 与 assemble_request 只映射选项和组织原生请求结构。
Transport.validate_parameters 只解析一次共同选项与声明式绑定，并返回公开类型 PreparedOptions。
BaseCodec 显式将其传给 Decoder.request；request_parameters(profile, options) 与
assemble_request(..., options=options) 直接复用结果，不重新解析或提供隐式兜底。
绑定模板只保存一份已展开占位符的 bindings，omit 保留到公共 merge_bindings 按层递归合并时消费。
五协议共用删除语义，Profile 的两个推理绑定字段也会保留顶层删除标记；
不保留 Responses 专属的 raw binding 路径。普通 Profile 字典仍按既有按键覆盖规则处理。
公开 messages / tools / chunks 入口继承基类固定流程；协议实现 `_messages` / `_tools` / `_chunks`。
Decoder._messages / _tools 返回 Translations，显式声明每组输出的来源；公开入口仍返回普通 list。
翻译中的格式解析错误和上游失败事件仍按[异常边界](../reference/api.md#异常边界)报告。

Transport 从公开 SDK 消息与工具类型取得字段定义。过滤仅作用于请求副本，
不修改业务保存的消息，也不递归删除工具参数、结果或 JSON Schema 中的应用键。
兼容服务的显式 extra_body、extra_headers 与思考回放策略仍由 Profile 配置。

## 请求与返回路径

```text
请求：输入标准格式 → Encoder → 中间态 → Transport 校验并准备选项 → Decoder → Transport 过滤、结构校验 → 官方 SDK
返回：官方 SDK 流 → Encoder → 请求独立累积状态 → Decoder → 输入协议的事件与完成消息
```

`ProviderProfile.protocol` 决定实际 API，`stdio_protocol` 决定业务层的输入与输出。
二者独立；未指定 I/O 协议时整批识别，仅唯一匹配时绑定；零匹配或多匹配报 MessageValidationError。
普通文本的 role/content 结构可能匹配多种协议，应显式指定 stdio_protocol；显式指定跳过推测，仍验证所选协议的输入。
标准协议默认值由各 Codec.defaults 的 ProtocolDefaults 声明；Profile 解析从注册表复制声明，
不创建 Codec 或 SDK 客户端。兼容服务差异通过 Profile 提供给 Encoder / Decoder。
Codec 调用 Transport 校验并准备生成选项，再由 Decoder 翻译请求；Transport 在 SDK 边界过滤并校验原生请求。

Protocol 是 str 别名。新增协议实现四个基类后，启动时调用 MyCodec.register(codec)；
公共入口、类型枚举、默认值表和公共编排无需变更。注册拒绝重名、不匹配的组件和有实例字典的实现。
MyCodec / MyLLM 保存创建时的注册快照，后续注册只影响新实例。

配置与客户端属于 MyLLM 实例；StreamState、MessageStream 和生成选项属于单次请求。
客户端延迟创建并按实例复用，响应流按请求关闭。实例退出前先完成或关闭响应流，再统一释放客户端，
详见[客户端生命周期](client-lifecycle.md)。流操作、块身份及完整工具发布规则见[流状态设计](stream-state.md)。

## 字段保留与轮次恢复

转换结果保留未知扩展；路径可以共存时保留原层级。发生类型冲突或存在原生签名、密文时，
结果附带 JSON 可序列化的恢复记录。重新输入未修改的结果可恢复原始中间态，
再转换或请求原协议时保留原生思考块和工具身份。

恢复记录随消息保存，不依赖客户端、进程缓存或 MyLLM 实例；修改对应转换分组会使旧快照失效。
消息合并或拆分时，Decoder 显式记录来源分组，未知字段不会按列表下标串到下一轮。
实际请求由 Transport 过滤扩展与恢复记录，避免它们进入 SDK 请求。
流式多轮对话应保存 `stream.result`，逐字拼接文本不能替代完整历史。
规则与边界见[字段保留说明](../audits/lossless.md)。

## 验证

- [恢复契约](../../tests/contract/test_replay_preservation.py)：五协议消息、工具、完整流的保存与往返，编辑失效、加密思考及工具 ID 恢复。
- [SDK 两轮回放](../../tests/integration/test_replay_transport.py)：真实 Anthropic SDK、本地 HTTP 响应，验证签名、密文、工具结果和实际请求过滤。
- [组件契约](../../tests/unit/test_native_codecs.py)与[依赖边界](../../tests/contract/test_codec_boundaries.py)：翻译、验证及 SDK 职责隔离。
- [独立协议注册](../../tests/integration/test_protocol_registration.py)：第六协议的默认值、注册快照及完整请求路径。
- [原路径保留](../../tests/contract/test_intermediate_preservation.py)与[协议符合性](../../tests/contract/test_protocol_conformance.py)：字段层级、转换优先级与协议结构。
- [并发验证](../audits/concurrency.md)：五协议组合、混合请求、失败隔离和资源关闭。

测试使用本地响应，不验证供应商真实密文的密码学有效性。服务端仍校验模型、上下文与签名。
开发与构建步骤见[文档目录](../README.md)，扩展协议见[接入指南](../guides/adding-protocol.md)。
