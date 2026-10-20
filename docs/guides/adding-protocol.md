# 接入新的协议

每个协议实现 BaseEncoder、BaseDecoder、BaseTransport，由 BaseCodec 组合并声明默认值。
Encoder / Decoder 只翻译已识别字段；请求验证、SDK 结构过滤与 SDK 调用归 Transport。
四个基类及 PreparedOptions、ProtocolDefaults、Parameter、Config 均可从 `my_llm` 导入。
独立协议直接实现这些基类；不需要继承内置五种协议，也不需要修改公共调用入口。

## 原生实现目录

```text
<protocol>/
├── __init__.py                 # 仅导出
├── codec.py                    # 协议身份、默认值与三个组件
├── transport.py                # 参数验证、过滤、SDK 生命周期
├── utils/                      # 按需提供的双向共用辅助
│   └── __init__.py
├── encoding/
│   ├── __init__.py
│   ├── encoder.py              # 原生 → 中间态
│   └── content.py              # 按需提供的翻译辅助
└── decoding/
    ├── __init__.py
    ├── decoder.py              # 中间态与生成选项 → 原生请求
    └── content.py
```

实现保持无状态，使用空 `__slots__`。Profile 按调用传入，不保存流片段、客户端或请求历史。
包初始化文件只导出接口；原生实现不导入其他协议、相反翻译方向或 Transport 的实现。

## Encoder 与 Decoder

两个基类均要求提供 `protocol` 属性，可用类属性实现。
messages / tools / chunks 是固定公开入口；原生实现提供对应的下划线钩子，
签名为 `(values: list, profile: ResolvedProfile | None = None, **parameters)`。

| 实现 | 必需接口 |
| --- | --- |
| BaseEncoder | `_messages`、`_tools`、`_chunks`、`usage`、`match_item`、`protocol`。 |
| BaseDecoder | `_messages`、`_tools`、`_chunks`、`usage`、`protocol`、`assemble_request`。 |

Encoder.matches 统一整批识别。Encoder 与 Decoder 的 `_messages` / `_tools` 返回普通 list，
每个实现显式读取与构造受支持的字段，不透传未知协议键，也不附加原生恢复记录。
工具参数、工具结果和 JSON Schema 等已识别开放字典整体按应用数据处理。
原生钩子内部复用本方向方法时调用 `_messages` 等钩子，避免重复包装。
Decoder.result 默认返回首条消息，Responses 等协议可覆盖为 items 列表。

Encoder 的固定入口借用输入字典或 SDK 字段视图，私有钩子只读取已识别字段并构造中间态。
Decoder 同样只读借用中间态；两个方向的 messages / tools 公开出口复制本次已翻译结果。
两个方向的 `_chunks` 必须直接返回独立事件：新建协议容器，只对借入的可变字段建立副本；
公开 chunks 入口直接转交结果，避免每个文本增量再次复制完整新建外壳。
例如工具参数对象、开放元数据、logprobs 的 bytes 列表仍须捕获；不可变字符串可以直接使用。
共享 metadata(values, path=()) 已捕获 values；直接构造其他操作或原生帧时由钩子保证所有权。
内部 select 仅选择字段，不复制嵌套 payload。
各公开边界均承担自己的隔离责任，不能理解为整条 generate 路径仅复制一次。
同协议转换也必须显式翻译字段，不能直接返回原生对象快照。
私有钩子不能原位修改调用方的输入块来补齐字段，应重建实际变化的字典或列表。
新增协议必须验证 chunks 的两个方向：修改输入不得影响已返回事件，修改事件不得影响输入，
并覆盖通过 parameters 传入的可变元数据。

Transport 的 `filter_messages` / `filter_tools` 接收只读借用的原生叶节点，必须返回拥有独立可变数据的结果。
根请求过滤会将 `message_path` / `tool_path` 对应叶交给这些钩子处理一次；覆盖钩子同样遵循该约束。

BaseDecoder.request 显式接收已验证的 options；final 基类流程统一调用 request_parameters(profile, options)
生成参数字典，再调用 assemble_request(messages, tools, profile, *, request: dict, **parameters)。
协议的 assemble_request 只将原生消息和工具写入已有 request 字典，不接收 options，也不重复调用 request_parameters。
参数应用的顺序和仅执行一次由基类保证，不提供缺失选项后重新解析的兼容路径。
PreparedOptions 是公开的请求独立选项类型，捕获本次调用的 token、headers、原生参数及思考控制。
原生请求始终使用普通字典。BaseDecoder.request_parameters 固定处理所有 phase="decode" 的 Parameter，
协议复用该流程，不再各自实现参数默认值遍历。通过类属性声明请求布局：
body_container 默认 None、headers_field 默认 "$.extra_headers"、body_overlay 默认 None、omit_empty_body 默认 False。
例如 Gemini 将 body_container 设为 "config"；SDK 具有额外请求体覆盖机制时用 body_overlay 声明对应容器。
协议专有的请求控制通过可选 `_request_parameters(request, profile, options)` 钩子补充，
该钩子返回请求字典，不重新应用路径或解析共同选项。Config 转换结果优先于缺失时默认值。
已存在参数即使为 None 也保留；default_value=omit 不补缺失值，整个字段=omit 禁用自动处理。
显式 generate(max_tokens=omit) 删除该请求的目标键。推理启用值仅在本次请求开启推理时应用。

BaseEncoder.parameter_value(value, profile, field_id, default_path, default=None, *, use_default=True)
统一读取 phase="encode" 的参数路径及缺失默认值。协议钩子在确定事件语义后调用它，
再构造对应中间态字段；不遍历所有返回字段，也不根据 Parameter.name 生成中间态键。
use_default=False 可用于事件判别后的分支读取，避免在不对应的事件中制造缺失默认值。
例如独立协议确认当前事件是文本增量后，读取 content；已有 None 保留，只有缺失时应用默认值：

```python
def read_text_delta(encoder, event, profile):
    return encoder.parameter_value(event, profile, "content", "$.text")
```

Decoder 输出标准事件时使用其原生字段，不使用 encode 参数重排输出。
这两个方向不实现请求参数校验、模型能力验证、SDK 字段过滤或请求执行。
工具 JSON、媒体及原生 Schema 的解析仍属于翻译；上游失败事件仍需转换为相应流错误。

## Transport

[BaseTransport](../../src/my_llm/codec/base/transport.py) 提供固定的 request 流程：
检查工具关联与能力 → filter_messages / filter_tools → SDK 消息与工具类型校验 → validate_request。
过滤读取公开 SDK 类型，不把任意工具参数、结果字典或 JSON Schema 当作协议字段清理。
Schema 结构预编译为共享的类型计划；只读输入校验使用 `projected`，需要独立可变结果的过滤使用
`supported`。缓存不保存请求值。固定流程只重建会被替换的请求容器，消息和工具由过滤器隔离；
`validate_request` 若需修改其他嵌套选项，也应只重建其实际改写的容器。

| 接口 | 职责 |
| --- | --- |
| message_type / tool_type | 必需。公开 SDK 的消息与工具类型，用于结构字段过滤。 |
| message_path / tool_path | 必需。声明原生请求中消息与工具的字段路径。 |
| request_schema | 必需。声明官方 SDK 方法的根关键字及相应类型；使用 parameter_types(SDK方法) 获取。 |
| validate_parameters(profile, **parameters) | 验证并返回 PreparedOptions；协议在父类结果上检查参数路径、控制值和能力限制。 |
| validate_request(request, profile, **parameters) | 必需。验证原生选项并返回参数字典，不能将整份请求替换为 SDK 模型。 |
| create_client(profile, **parameters) | 必需。同步创建官方 SDK 客户端。 |
| upstream(client, request, profile, **parameters) | 必需。将过滤后的字典通过 **request 解包给 SDK，返回清理本次响应资源的异步上下文管理器。 |
| validate_input | 默认按公开 SDK 类型验证声明的输入协议，保留原始输入。 |
| validate_profile / usage_modes | 默认检查用量模式，可增加连接和协议限制；新增模式只需在本 Transport 的 usage_modes 中声明。 |
| filter_messages / filter_tools | 默认按 SDK 类型过滤；可按 Profile 放行兼容字段或检查 Schema 表达能力，不改写消息内容。 |
| allows_external_calls(previous_response_id) | 默认 False；服务端历史协议可覆盖。 |
| close_client / client_key | 默认关闭 SDK 客户端、使用单个复用键；按 SDK 生命周期与请求头绑定规则覆盖。 |

不覆盖 BaseTransport.request；tools=None 表示省略，tools=[] 表示显式空列表。
BaseCodec.request 接收中间态，由 Transport 一次校验、解析统一生成控制参数，
再将返回的 options 显式传给 Decoder.request 翻译，
最后将原生请求交给 Transport.request 过滤和校验。MyCodec.request 只路由。
BaseCodec.validate_profile 委托 Transport。Profile 从注册的 Codec.defaults 复制默认值，
负责合并配置、校验通用字段并建立快照，不创建 Codec 或 SDK 对象。
usage_mode 是开放字符串；Profile 检查非空字符串格式，Transport.usage_modes 决定接受哪些具体值。

## Codec 组合

BaseCodec 必须提供 protocol_or_api_style、defaults、encoder、decoder、transport。
defaults 使用 frozen dataclass ProtocolDefaults。两个行为开关
replay_reasoning_content_when_tool_calls、supports_reasoning_effort 通过 Config 明确声明；
max_tokens、content、reasoning_content、reasoning_enable、reasoning_effort 使用 Parameter 声明，不用的字段保留默认 omit。
max_tokens / reasoning_enable / reasoning_effort 属于 decode，content / reasoning_content 属于 encode。
路径从根字典定位，使用 $. 前缀，$ 表示 **parameters 解包字典的根；事件类型不是参数路径。
wrapper 的 name / comment 仅为展示信息；字段标识承载语义。
Profile 字段 None 表示继承，wrapper.default_value=None 为实际值，仅在目标字段缺失时应用。

ProtocolDefaults、ProviderProfile、ModelProfile 可继承扩展：新增参数声明
Parameter[T] | Omit | None，分支配置声明 Config[T] | Omit | None。
解析器自动发现继承字段并检查声明，不需要路径与默认值配对，也不需要中央参数表。
自定义 Config 的含义由协议组件实现；所有 decode 参数的路径和默认值由 BaseDecoder 自动应用。
自定义 encode 参数由协议钩子调用 BaseEncoder.parameter_value 后映射为明确的中间态语义。
SDK 根参数必须符合 SDK 调用签名，非标准字段使用 SDK 正式提供的 extra_body 等扩展入口。

完成自己的 Encoder、Decoder、Transport 后，组合和注册如下；将 your_package 替换为实际包名：

```python
from my_llm import (
    BaseCodec,
    Config,
    ModelProfile,
    MyCodec,
    MyLLM,
    Parameter,
    ProtocolDefaults,
    ProviderProfile,
    omit,
)
from your_package.decoding.decoder import ExampleDecoder
from your_package.encoding.encoder import ExampleEncoder
from your_package.transport import ExampleTransport


class ExampleCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "example"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=Config(name="回放思考", default_value=False),
        supports_reasoning_effort=Config(name="思考强度", default_value=False),
        max_tokens=Parameter(
            name="输出上限",
            path="$.max_tokens",
            default_value=omit,
            phase="decode",
        ),
    )
    encoder = ExampleEncoder()
    decoder = ExampleDecoder()
    transport = ExampleTransport()


# 在应用启动时执行一次，早于 Profile 解析及 MyCodec / MyLLM 实例创建。
MyCodec.register(ExampleCodec())
codec = MyCodec()
llm = MyLLM(
    provider=ProviderProfile(protocol="example"),
    model=ModelProfile(model="example-model"),
    stdio_protocol="example",
)
```

注册会检查 Codec 与三个组件的基类类型、ProtocolDefaults 类型，以及 Codec / Encoder / Decoder
的协议名一致性。四个实现都须声明空 `__slots__`，不能有实例 `__dict__`，
继承链上的非空数据槽也会被拒绝；
协议名须为非空字符串，重名注册抛出 ConfigurationError，不覆盖已有实现。
MyCodec / MyLLM 实例保留创建时的注册快照；注册新协议后需创建新实例才能使用。

## 中间态与字段边界

消息使用 role 与 content 块；工具调用与结果使用 tool_use / tool_result。
它是内部开放字典，不是新的应用消息类型。

- 中间态只包含已识别字段；未声明的非标准字段直接跳过，不读取、不校验，也不逐项扫描检查未知键。
- protocol 字符串记录必要的来源语义，Responses 的 coalesce 标志控制已识别内容的合并；不保存原始对象或事件。
- functions 表示规范化工具声明，参数、结果及工具 JSON Schema 的开放键值完整保留。
- conversion.encode_functions / decode_functions 管理工具声明；stream 管理原生事件操作与用量计数。
- 响应元数据通过 shared.stream.metadata(values, path=()) 声明：Encoder 显式投影已识别字段并归一化，
  StreamState 仅沿 path 浅合并 values，不维护任何协议专属字段清单。嵌套更新与完整替换由 Encoder 选择位置。
- options 提供参数路径操作与生成选项；BaseEncoder / BaseDecoder 固定参数读取和请求应用流程，
  transport 提供 SDK 字段过滤、工具关联验证及响应资源关闭。
跨协议来回翻译不保证无损。不能把其他协议的密文解释为本协议的签名，也不通过隐藏记录恢复跨协议无法表达的信息。
标准签名与密文必须在对应协议实现中明确处理。详见[翻译字段边界](../design/translation-boundaries.md)。

## 注册与验证

1. 实现四个基类并通过 `MyCodec.register(codec)` 注册。Protocol 是 str 别名，
   不需要修改公共入口、协议枚举、默认值表或公共编排。
2. 独立编写原生样例，覆盖消息、工具、完整流、两个翻译方向未知字段不泄漏及开放字典内部键值保留。
3. 使用官方 SDK 和本地 HTTP transport 检查实际请求：Profile 兼容能力生效，标准签名按原协议处理，工具参数与 JSON Schema 不被误删。
4. 验证正常、取消、失败、提前退出、并发隔离和客户端关闭。

组件与依赖约束见[契约测试](../../tests/unit/test_native_codecs.py)和[边界测试](../../tests/contract/test_codec_boundaries.py)。
独立第六协议的注册、默认值、请求与流集成示例见[注册测试](../../tests/integration/test_protocol_registration.py)。
构建及完整验证命令见[开发规范](development.md)。
