# 项目架构与职责边界

2026-10-21 更新。一次 `generate(messages=..., stream=True, max_tokens=1024)` 将完整调用参数
打包为字典，连同已解析 Profile 进入 Codec。Encoder 验证输入协议并生成中间字典；
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
├── invoke.py                   # 完整输入打包、客户端缓存、输出流编排
├── registry.py                 # 协议注册和实例注册快照
├── profile/                    # 配置分层、声明分类、校验及实例快照
├── stream/                     # 请求独立的累积状态、结果及关闭回调
└── codec/
    ├── converter.py            # MyCodec，输入绑定与完整请求准备路由
    ├── base/
    │   ├── codec.py            # 组合三个无状态组件，提供离线转换入口
    │   ├── encoder.py          # 完整调用 → 已验证输入及中间字典
    │   ├── decoder.py          # 中间字典 → client / request / local
    │   └── transport.py        # 官方 SDK 构造、执行及响应资源管理
    ├── shared/
    │   ├── conversion.py       # 字段视图与转换辅助
    │   ├── options.py          # 请求选项、参数路径及优先级
    │   ├── stream.py           # 流操作、块身份及用量
    │   └── transport.py        # 共享 Schema 规则、历史验证及流资源辅助
    └── <protocol>/             # codec、transport、encoding、decoding、按需 utils
```

| 组件 | 职责 |
| --- | --- |
| BaseEncoder | 协议识别、输入 Schema 验证、完整调用编码、消息／工具／事件／用量翻译。 |
| BaseDecoder | 请求选项解析、参数映射、默认值与覆盖、工具关联及能力验证、目标 Schema 过滤和验证、最终字典及结果形状。 |
| BaseTransport | create_client(parameters)、upstream(client, request)、close_client(client)；不接收 Profile 或原始生成参数。 |
| BaseCodec | 声明身份及 ProtocolDefaults，组合 Encoder / Decoder / Transport，协调离线入口。 |
| MyCodec | 注册快照、输入协议绑定、prepare 路由；不保存调用数据。 |
| Profile / resolve_profiles | 分层合并、声明分类、通用字段验证、连接参数解析及独立配置快照。 |
| MyLLM | 固定实例配置、已准备连接字典、延迟客户端缓存、生成和输出编排。 |
| StreamState / MessageStream | 每次请求独立的响应状态、事件、结果副本和清理回调。 |

四个协议组件保持空 `__slots__`，不保存 Profile、客户端、历史或流状态。
Encoder / Decoder 不创建 SDK 客户端，也不相互导入。官方类型与方法签名可用于静态 Schema 分析；
实际 SDK 构造和调用只发生在 Transport。双向共用辅助放在本协议 utils 中。
包初始化文件只导出接口；公共执行逻辑不按具体协议名分支。

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
          → request → 参数路径及默认值 → assemble_request
          → native_request → 历史关联、Schema 过滤、能力及原生结构验证
          → {client, request, local}
      → MyLLM 延迟取得／复用客户端 → Transport.upstream → SDK(**request)

返回：SDK 事件 → 上游 Encoder → 本次 StreamState
      → stdio Decoder → 原生事件／完成消息 → 本次响应关闭
```

`ProviderProfile.protocol` 选择上游，`stdio_protocol` 选择业务输入输出。
未设置 stdio_protocol 时，仅整批输入唯一匹配才绑定；零匹配和多匹配报 MessageValidationError。
绑定与准备没有 await；准备完成后修改调用方历史、body、headers 不会改变待消费请求。
同协议和跨协议均经过 Encoder / Decoder；收集式调用消费同一个 MessageStream。

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
离线公开 messages / tools 仍返回独立副本。流事件在借入可变数据处捕获，结果读取仍返回独立副本。

根 Schema 计划委派 message_path / tool_path 叶，由 filter_messages / filter_tools 过滤一次；
Anthropic 的开放 input_schema 再委派为应用数据，只复制完整 JSON Schema 一次。
TypedDict / Pydantic 只处理声明的结构字段；开放 dict / Any 中的工具参数、结果和 Schema 约束完整保留。
Profile 显式声明的兼容叶字段按路径处理，未知字段不会通过缓存学习或透传。

复用 shared/transport.py 的有界 LRU：字段、TypeAdapter、SDK 方法参数类型、编译计划、路径声明
各最多 512 项。缓存键及闭包只保存类型和编译规则，不保存请求数据、Profile、客户端或上次 Union 分支。
类型分析命中后，当前请求仍逐值选择分支、过滤和验证；输入和输出的验证分别保证两个协议边界。
详见 [Schema 过滤计划缓存](schema-filter-cache.md)。

客户端缓存属于 MyLLM；块物化与终态复用属于请求独立的 StreamState。
生命周期见[客户端生命周期](client-lifecycle.md)，流式顺序见[流状态设计](stream-state.md)。

## 验证

- [完整输入与准备生命周期](../../tests/unit/test_request_pipeline.py)：完整字段、三个目的地、连接仅解析一次、客户端复用及无重复结果复制。
- [组件契约](../../tests/unit/test_native_codecs.py)、[依赖边界](../../tests/contract/test_codec_boundaries.py)与[独立第六协议](../../tests/integration/test_protocol_registration.py)。
- [Schema 计划](../../tests/unit/test_schema_filter_plans.py)、[原生请求边界](../../tests/unit/test_native_request_boundary.py)及[数据所有权](../../tests/unit/test_copy_ownership.py)。
- [真实 SDK 并发与资源隔离](../../tests/integration/test_concurrency.py)及[客户端生命周期](../../tests/unit/test_client_lifecycle.py)。
- [相同负载性能基准](../guides/performance.md)：完整请求时间、独立内存探针、缓存和真实重叠并发。

输入输出字段的范围见[翻译边界](translation-boundaries.md)，扩展接口见[接入指南](../guides/adding-protocol.md)。
