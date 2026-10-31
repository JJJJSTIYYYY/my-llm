# 项目架构与职责边界

2026-10-29 更新。一次 `generate(messages=..., stream=True, max_tokens=1024)` 将完整调用参数
打包为字典，连同已解析 Profile 进入 Codec。Encoder 先投影支持字段、验证输入协议并生成类型化中间协议；
Decoder 应用参数路径、默认值、覆盖规则及目标 SDK Schema，输出三个明确的目的地：

```python
{
    "client": {"api_key": "...", "base_url": "..."},
    "request": {"model": "...", "messages": [...], "stream": True},
    "local": {"stream": True, "client_key": None},
}
```

`local.stream` 决定返回 MessageStream 还是消费完整结果；上游仍统一使用流式 API。
客户端参数由 MyLLM 实例准备、按只读方式使用；请求字典和流状态属于本次调用。
Transport 只把最终字典交给官方 SDK，并管理客户端和本次响应的资源。

## 目录与组件

```text
src/my_llm/
├── __init__.py                 # 统一公开导出
├── _version.py                 # 构建与发行版本
├── py.typed                    # 类型标记
├── my_llm.py                   # MyLLM / generate、客户端缓存、流编排
├── cache/                      # 缓存实现与所有权隔离
│   ├── __init__.py             # 缓存接口导出
│   ├── compiled_cache.py       # CompiledCache、SDK 类型计划编译、过滤与验证
│   └── message_cache.py        # MessageCache，已翻译增量历史的更新与快照
├── basic/                      # 配置、模型与运行时共用的基础能力
│   ├── data.py                 # 数据捕获与 SDK 字段只读视图
│   ├── exceptions.py           # 统一异常
│   ├── semantic_loss.py        # 调用独立损失收集与实例策略
│   └── protocols.py            # 协议注册、工厂快照与可选 SDK 发现
├── middleware/                 # SDK 无关的固定中间协议
│   ├── models.py               # 消息、内容、工具及其 wire 投影
│   ├── stream.py               # 流操作、公开事件、条目身份与用量
│   └── protocol.py             # 完整容器与类型化生成输入校验
├── profile/                    # 配置分层、声明分类、校验及实例快照
├── runtime/                    # 响应流与每次调用的状态
│   ├── message_stream.py       # 公开异步流与关闭回调
│   └── state.py                # 本次请求的块累积、事件与终态
└── codec/
    ├── converter.py            # MyCodec，输入绑定与完整请求准备路由
    ├── base/
    │   ├── codec.py            # 管理实例组件和损失 adaptor，提供翻译入口
    │   ├── encoder.py          # 完整调用 → 已验证输入及中间协议
    │   ├── decoder.py          # 中间协议 → client / request / local
    │   └── transport.py        # SDK 执行契约及共用响应关闭辅助
    ├── shared/
    │   ├── conversion.py       # 消息、媒体、工具转换及独立历史关联检查入口
    │   ├── options.py          # 请求选项、参数路径及优先级
    │   ├── output.py           # 原生与中间输出共用的能力过滤
    │   └── stream.py           # 流操作、块身份及用量
    └── <protocol>/             # codec、transport、encoding、decoding 及按需共享模块
```

| 组件 | 职责 |
| --- | --- |
| BaseEncoder | 协议识别、输入 Schema 验证、完整调用编码、消息／工具／事件／用量翻译。 |
| BaseDecoder | 请求选项解析、参数映射、默认值与覆盖、工具关联及能力验证、目标 Schema 过滤和验证、最终字典及结果形状。 |
| BaseTransport | create_client(parameters)、upstream(client, request)、close_client(client)；不接收 Profile 或原始生成参数。 |
| BaseCodec | 声明身份及 ProtocolDefaults，组合 Encoder / Decoder / Transport；在调用损失 adaptor 前发布消息翻译缓存。 |
| MyCodec | 注册快照、输入协议绑定、prepare 路由；持有实例独立的 MessageCache。 |
| CompiledCache / MessageCache | 分别保存进程级类型规则和按实例键隔离的已翻译输入历史，集中管理缓存操作与快照。 |
| Profile / resolve_profiles | 分层合并、声明分类、通用字段验证、连接参数解析及独立配置快照。 |
| MyLLM | 固定实例配置、uuid4 消息缓存键、已准备连接字典、延迟客户端缓存、生成和输出编排。 |
| StreamState / MessageStream | 每次请求独立的响应状态、事件、最终结果 draft 和清理回调。 |

注册表保存工厂，codec 及其组件按实例按需创建；允许实例状态，不跨实例共享组件。
MyLLM 的 Profile 解析从本实例 MyCodec 获取协议默认声明，后续验证和请求复用同一个 codec。
`BaseCodec` 同时管理 `BaseSemanticLossAdaptor`，支持值无法表达时在完整 draft 上执行 raise / ignore / hook。
模型与边界详见[中间协议](protocol-middleware.md)。
Encoder / Decoder 不创建 SDK 客户端，也不相互导入。官方类型与方法签名可用于静态 Schema 分析；
实际 SDK 构造和调用只发生在 Transport。双向共用辅助按职责放在协议根目录，
例如 anthropic/content.py、ollama/schema.py；单个模块不额外套 utils 目录。
包初始化文件只导出接口；公共执行逻辑不按具体协议名分支。

目录按责任边界分组，不按方法数量拆分。中间协议只设三个实现模块；消息模型与 wire
解析相邻，流模型与事件投影相邻，完整输入校验与容器相邻。MyLLM、MyCodec、各协议
Encoder / Decoder 的完整调用流程仍集中在所属类中。初始化文件仅重导出；运行时内部
直接导入定义模块。`from my_llm import ...` 和 `from my_llm.middleware import ...`
继续提供同一组协议模型。MyLLM 执行编排位于顶层 my_llm.py；
基础能力在 basic，缓存实现在 cache，响应流状态在 runtime。迁移后的内部路径不保留转发文件。

## 请求与返回路径

```text
构造：Codec.defaults → provider → model → ResolvedProfile
      → Decoder.validate_profile → Decoder.client_parameters → 实例连接字典

请求：generate 完整输入字典 + ResolvedProfile
      → MyCodec.bind_input（已绑定时直接使用，否则唯一识别）
      → 输入 Encoder.request → 输入验证 → 消息与工具中间态
      → 上游 Decoder.prepare
          → validate_parameters（每次一次）
          → request_client / client_key（SDK 的请求级客户端变体）
          → request → 当前批工具关联验证 → 消息翻译
          → 启用缓存时：按批次哈希合并消息列表，取得独立快照
          → 参数路径及默认值 → assemble_request
          → native_request → Schema 过滤、能力及原生结构验证
          → {client, request, local}
      → BaseCodec.translate 发布消息翻译缓存，再调用语义损失 adaptor
      → MyLLM 延迟取得／复用客户端 → Transport.upstream → SDK(**request)

返回：SDK 事件 → 上游 Encoder → 本次 StreamState
      → stdio Decoder → 原生事件／完成消息 → 本次响应关闭
```

`ProviderProfile.protocol` 选择上游，`stdio_protocol` 选择业务输入输出。
未设置 stdio_protocol 时，仅整批输入唯一匹配才绑定；零匹配和多匹配报 MessageValidationError。
绑定与准备没有 await；准备完成后修改调用方历史、body、headers 不会改变待消费请求。
同协议和跨协议均经过 Encoder / Decoder；收集式调用消费同一个 MessageStream。

显式 `stdio_protocol="middleware"` 使用公开具体模型，不参与协议推断：请求略过输入 Encoder，
直接进入同一个上游 Decoder.prepare；返回由上游 Encoder / StreamState 生成
AssistantMessage / MessageEvent，略过 stdio Decoder。该 stdio 名称不注册协议组件。
模型和公开事件字段见[直接使用中间协议](../guides/middleware-io.md)。

`MyCodec.request(intermediate, profile, **parameters)` 保留为离线构造原生 SDK 请求的便利入口，
与完整 prepare 共用 Decoder.request。它返回裸请求字典，不创建客户端。
`Decoder.native_request` 用于已有原生字典的独立过滤验证，也不执行 SDK。

## 参数分类与优先级

Profile 用 `Parameter[T]` 声明路径、缺失默认值及 encode / decode 阶段，`Config[T]` 声明代码配置。
字段 `None` 继承，字段 `omit` 禁用；wrapper 的 `default_value=None` 是实际值。
路径统一从 SDK 字典根 `$` 定位。显示名称和注释不参与处理。

连接 Parameter 由 dataclass 字段的 `metadata={"scope": "client"}` 声明，内置 api_key 已声明该 scope。
解析器一次建立 `ResolvedProfile.request_parameters` 和连接快照；请求过滤器按分类遍历，
不再根据字段名排除认证参数。自定义请求 Parameter 默认使用 request scope。
连接 base_url、timeout、max_retries 的通用限制在解析时检查，SDK 专属转换由 Decoder 按实例完成。

优先级为显式生成控制值 → 已有原生值（包括 None、False、0）→ 缺失默认值。
headers/body 按现有分层规则合并；嵌套 omit 删除目标值，生成 max_tokens=omit 删除相应路径。
推理启用声明只在请求开启推理时应用，强度经过能力和映射检查。
保护的核心消息／模型字段不能被 extra_body 或参数声明覆盖。

协议的 assemble_request 只把原生消息和工具写入已有参数字典；
_request_parameters 追加协议独有控制，不重新解析生成选项。
Ollama 的逐请求 headers 绑定在 SDK 客户端上，由 request_client 创建独立参数变体；相同 headers 复用客户端。
其余协议把逐请求 headers 放在 request 中。

## 数据所有权与缓存

输入验证使用只读投影；Encoder 捕获已识别的中间值，不复制未知字段。
Decoder.request 调用只读私有翻译钩子，将借入的中间叶交给最终 Schema 边界一次捕获；
离线公开 messages / tools 捕获独立 payload；流事件在借入可变数据处捕获。
MessageStream.result 返回本调用最终 draft 本身，保证 output hook 的原位对象身份；get_last_usage 返回独立副本。

根 Schema 计划委派 message_path / tool_path 叶，由 filter_messages / filter_tools 过滤一次；
五种内置协议的消息过滤共用基类外层循环，各协议在 _filter_message 中处理一条消息。
结构校验与工具关联修复在同一次中间消息遍历中完成。HistoryNormalizer 去重调用与结果、
补空结果，并整条丢弃包含孤立结果的消息；丢弃时撤销本条已收集的关联与合并修改。
独立 HistoryValidator 保留严格校验用途。原始输入不被修改。
空结果在调用块的校验访问中预留，后续有效结果撤销该占位；无 ID 结果不推断关联。
重复结果内容在原有结果部件校验访问中合并。finish 仅执行常数时间检查，
原生翻译的既有循环惰性读取有效输出，不另行扫描工具表或重建消息列表。
已知 usage / metadata 操作在流入口允许进入防御性处理器，未知核心操作仍严格拒绝。
翻译结果、message_transform 改写和缓存历史都经过过滤循环及最终 SDK 校验。
Anthropic 的开放 input_schema 再委派为应用数据，只复制完整 JSON Schema 一次。
TypedDict / Pydantic 只处理声明的结构字段；开放 dict / Any 中的工具参数、结果和 Schema 约束完整保留。
Profile 显式声明的兼容叶字段按路径处理，未知字段不会通过缓存学习或透传。

cache/compiled_cache.py 集中管理 SDK 类型分析、计划编译、过滤与验证，以 `CompiledCache` 复用有界 LRU：
字段、TypeAdapter、SDK 方法参数类型、编译计划、路径声明各最多 512 项。
`SchemaKey`、计划图和缓存淘汰逻辑均位于该模块，Codec 通过接口调用。
缓存键及闭包只保存类型和编译规则，不保存请求数据、Profile、客户端或上次 Union 分支。
类型分析命中后，当前请求仍逐值选择分支、过滤和验证；输入和输出的验证分别保证两个协议边界。
详见 [Schema 过滤计划缓存](schema-filter-cache.md)。

`MyLLM(enable_message_cache=True)` 将开关和实例生成的 uuid4 键传至 `BaseCodec.translate`。
以上游 OpenAI Chat 为例：首次提交 `[system, user]`，下次提交 `[assistant, user]`，两批分别走原有翻译流程；
MessageCache 按原始输入哈希判断是否追加当前消息翻译列表，
返回包含四条消息的独立快照供 Decoder 组装本次请求。
增量可以包含 system、user、assistant 或完整工具调用与结果批次；只修复当前批的工具关联，
旧缓存中的对应关系不重复解析或修复。system 消息也保留在消息列表里，
Decoder 组装请求时将其放入 Anthropic 的 system、Gemini 的 system_instruction 等原生位置。
缓存不读取、提取或回填这些请求字段，也不需要 Decoder 声明缓存路径。
缓存只保存输入历史；模型输出需要调用方显式加入下一批增量，工具声明、生成控制和请求选项仍按次准备。
消息列表先在缓存中暂存，完整请求校验失败时不发布；请求准备完成后、语义损失 adaptor 调用前发布。
因此即使 adaptor、hook 或后续 Transport 调用失败，该批翻译结果也已保留。
hook 修改本次请求不会自动更新缓存，相关维护由使用者负责。
重发时仍翻译并处理语义损失；当前原始消息哈希与最后一次成功追加的批次相同，则复用原有原生历史，
不重复追加。哈希按消息内容稳定计算，SDK / middleware 输入均可使用，字典键顺序不影响比较；
工具声明、stream、headers 和生成选项不参与比较。只比较最后一批，因此 `A → B → A` 会追加最后的 A。
`aclose()` 同时释放本实例历史和批次哈希。

消息缓存使同一实例具有会话状态。`enable_message_cache=True` 时不支持同实例并发生成，
调用方必须先完成或关闭前一条流，再开始下一次调用；不能交错消费多个流。
此约束由调用方遵守，不通过内部排队串行化请求。关闭消息缓存时保留原有并发支持，
不同 MyLLM 实例可以并发使用。

客户端缓存属于 MyLLM；块物化与终态复用属于请求独立的 StreamState。
生命周期见[客户端生命周期](client-lifecycle.md)，流式顺序见[流状态设计](stream-state.md)。

## 验证

- [完整输入与准备生命周期](../../tests/unit/test_request_pipeline.py)：完整字段、三个目的地、连接仅解析一次、客户端复用及无重复结果复制。
- [组件契约](../../tests/unit/test_native_codecs.py)、[依赖边界](../../tests/contract/test_codec_boundaries.py)与[独立第六协议](../../tests/integration/test_protocol_registration.py)。
- [Schema 计划](../../tests/unit/test_schema_filter_plans.py)、[原生请求边界](../../tests/unit/test_native_request_boundary.py)及[数据所有权](../../tests/unit/test_copy_ownership.py)。
- [真实 SDK 并发与资源隔离](../../tests/integration/test_concurrency.py)及[客户端生命周期](../../tests/unit/test_client_lifecycle.py)。
- [相同负载性能基准](../guides/performance.md)：完整请求时间、独立内存探针、缓存和真实重叠并发。

输入输出字段的范围见[翻译边界](translation-boundaries.md)，扩展接口见[接入指南](../guides/adding-protocol.md)。
