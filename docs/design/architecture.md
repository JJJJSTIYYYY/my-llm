# 项目架构与职责边界

2026-10-19 更新。Encoder / Decoder 负责已识别字段的翻译，Transport 负责请求验证、
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
    │   ├── encoder.py          # 已识别的原生字段 → 中间态
    │   ├── decoder.py          # 已识别的中间态与生成选项 → 原生请求
    │   └── transport.py        # 请求验证、过滤及 SDK 生命周期
    ├── shared/
    │   ├── conversion.py       # 中间态与消息 / 工具翻译辅助
    │   ├── options.py          # 配置路径、生成选项翻译与独立校验辅助
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
同一协议两个方向共用的标准字段投影器放在该协议的 `utils/` 目录（如 `utils/content.py`、`utils/metadata.py`），
避免让 Encoder / Decoder 相互导入。
Codec 统一组合三个无状态组件；实际 SDK 客户端由 MyLLM 实例持有。
注册时检查实现类及其继承链中的非空 slots，阻止组件通过数据槽保存请求状态。

| 组件 | 职责 |
| --- | --- |
| BaseEncoder | 识别格式；翻译已识别的消息、工具、流事件与用量字段。 |
| BaseDecoder | 翻译已识别的消息、工具、流事件、用量和生成选项；组织原生请求结构；决定收集结果形状。 |
| BaseTransport | 验证输入格式、工具关联、模型能力及生成参数；返回请求独立的已解析选项；过滤不支持字段并校验原生结构；管理客户端创建、关闭和请求执行。 |
| BaseCodec | 声明协议身份、ProtocolDefaults 和三个组件，统筹翻译、验证及请求流程。 |
| MyCodec | 持有注册快照，路由转换与请求；内置五种协议，允许启动时注册新协议；完整离线流使用本次调用内的累积器。 |
| Profile / resolve_profiles | 声明配置和覆盖规则，合并协议、供应商与模型配置，校验通用字段并建立独立快照。 |
| MyLLM | 持有配置快照与 SDK 客户端缓存，编排请求的翻译、执行、累积和输出。 |
| StreamState / BlockState | 按请求独立维护块身份、内容、签名、工具参数和终态；不持有 SDK 客户端。 |
| MessageStream | 包装本次请求的异步生成器，提供迭代、完成结果副本和响应关闭；独立 generate 的流还负责临时实例的关闭回调。 |

Encoder / Decoder 不承担请求参数验证、SDK 结构过滤或 SDK 执行；翻译时只读取明确支持的协议字段。
Decoder 的 request_parameters 与 assemble_request 只映射选项和组织原生请求结构。
Transport.validate_parameters 只解析一次共同选项与控制配置，并返回公开类型 PreparedOptions。
BaseCodec 显式将其传给 Decoder.request；final 基类流程先调用 request_parameters(profile, options)，
再将已有参数字典传给 assemble_request(..., request=request)，由协议填入消息和工具。
assemble_request 不接收 options，也不重复应用参数；统一执行顺序保证参数流程仅执行一次，不重新解析或提供隐式兜底。
Profile 用 `Parameter[T]` 将路径、缺失默认值和阶段保存在同一声明中，用 `Config[T]` 控制分支。
encode 参数读取接口返回到中间态，decode 参数将中间态或生成选项写入接口请求；路径统一以 `$.` 开头，
`$` 表示 `**parameters` 解包字典的根。字段 None 继承，wrapper 的 default_value=None 是实际值。
默认值仅用于缺失字段，既有字段（包括 None）保持原值；显式生成控制参数可覆盖默认值。
`name` / `comment` 仅为展示数据，功能由 Profile 字段标识决定。
继承 Profile 可直接声明新的 Parameter 或 Config，不需要配对字段或维护中央参数表。
参数应用与阶段判断共用基类流程，五种内置协议及新注册协议复用；协议默认值仍由各自 Codec 声明。
公开 messages / tools / chunks 入口继承基类固定流程；协议实现 `_messages` / `_tools` / `_chunks`。
Decoder._messages / _tools 返回原生结果列表，不附加未知字段或恢复记录。
私有翻译钩子只读借用输入。messages / tools 的公开出口捕获已翻译结果；
两个方向的 `_chunks` 在借入可变字段处捕获数据，返回独立事件，公开 chunks 入口不再复制新建的完整事件外壳。
Transport 再为独立 SDK 请求建立所有权，StreamState 捕获流状态，公开结果读取仍返回独立副本。
翻译中的格式解析错误和上游失败事件仍按[异常边界](../reference/api.md#异常边界)报告。

Transport 根据 SDK 请求方法签名取得根参数，再从公开 SDK 消息、工具和选项类型取得内部字段定义。
翻译层生成普通字典，Transport 定点过滤和校验后仍以字典通过 **request 调用 SDK。过滤仅作用于请求副本，
不修改业务保存的消息，也不递归删除工具参数、结果或 JSON Schema 中的应用键。
TypedDict 与 Pydantic 协议结构仅处理声明字段；开放 dict / Any 内部保留应用键值。
兼容服务的显式 extra_body、extra_headers 与思考回放策略仍由 Profile 配置。

### 数据所有权与缓存

`MyLLM.generate` 的准备阶段不发生异步等待：先只读规范化并验证输入，再由 Encoder
构造拥有独立可变数据的中间态，完成后才进入 SDK 执行。协议私有钩子只读借用输入，
消息和工具在公开出口捕获，流事件在实际借入可变字段处捕获。历史关联校验只读借用数据；合并内容或覆盖字段时，
只重建实际改写的容器。公开转换结果、请求选项及完成结果拥有需要独立修改的内容。

Transport 的 Schema 缓存只保存类型分析结果：字段子计划、Literal 判别信息和 Union 候选顺序。
只读验证使用 `projected`，重建需要过滤的协议结构并借用应用 payload；生成请求使用 `supported`，
为可变输出建立必要副本。两者共用预编译计划，不保存历史、Profile 或 SDK 客户端。
Union 的同分选择遵循声明顺序；缓存键也保留嵌套 Union 的顺序。
缓存键、闭包中保存的实际规则及完整运行示例，见 [Schema 过滤计划缓存](schema-filter-cache.md)。

块物化缓存属于请求内的 `BlockState`，随内容更新失效。成功结束后，`StreamState` 复用经过验证的
终态块与消息，调用编排直接把完成事件中的消息交给 Decoder；不重新物化或再次解析工具 JSON。
这些缓存没有改变 Encoder、Decoder、Transport 的职责，也没有把请求数据放入共享组件。

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

## 翻译字段边界

Encoder 只将明确支持的协议字段和已配置兼容字段翻译为中间态；Decoder 只构造目标协议
已识别的字段。未知字段在两个方向均不透传，已识别开放字典内部的键值随整个字段保留。
不保存原始对象、剩余字段或跨协议恢复记录；同协议也通过明确的字段映射处理原生内容。
原生签名、密文与工具身份属于显式支持的标准语义，不作为未知扩展搬运到其他协议。
流式多轮对话应保存 `stream.result`，逐字拼接文本不能替代完整历史。
规则与示例见[翻译字段边界](translation-boundaries.md)。

## 验证

- [组件契约](../../tests/unit/test_native_codecs.py)与[依赖边界](../../tests/contract/test_codec_boundaries.py)：翻译、验证及 SDK 职责隔离。
- [独立协议注册](../../tests/integration/test_protocol_registration.py)：第六协议的默认值、注册快照及完整请求路径。
- [协议符合性](../../tests/contract/test_protocol_conformance.py)：独立原生样例与目标协议结构。
- [并发验证](../audits/concurrency.md)：五协议组合、混合请求、失败隔离和资源关闭。
- [数据所有权](../../tests/unit/test_copy_ownership.py)、[过滤计划](../../tests/unit/test_schema_filter_plans.py)与[终态物化](../../tests/contract/test_stream_materialization.py)：深层隔离、缓存复用及请求间数据独立。

测试使用本地响应，不验证供应商真实密文的密码学有效性。服务端仍校验模型、上下文与签名。
开发与构建步骤见[文档目录](../README.md)，扩展协议见[接入指南](../guides/adding-protocol.md)。
