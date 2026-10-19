# 接入新的协议

每个协议实现 BaseEncoder、BaseDecoder、BaseTransport，由 BaseCodec 组合并声明默认值。
Encoder / Decoder 只翻译已识别字段；请求验证、SDK 结构过滤与 SDK 调用归 Transport。
四个基类及 PreparedOptions、ProtocolDefaults 均可从 `my_llm` 导入。

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
| BaseDecoder | `_messages`、`_tools`、`_chunks`、`usage`、`protocol`、`request_parameters`、`assemble_request`。 |

Encoder.matches 统一整批识别。Encoder 与 Decoder 的 `_messages` / `_tools` 返回普通 list，
每个实现显式读取与构造受支持的字段，不透传未知协议键，也不附加原生恢复记录。
工具参数、工具结果和 JSON Schema 等已识别开放字典整体按应用数据处理。
原生钩子内部复用本方向方法时调用 `_messages` 等钩子，避免重复包装。
Decoder.result 默认返回首条消息，Responses 等协议可覆盖为 items 列表。

Encoder 的固定入口已捕获一次输入，私有钩子接收隔离后的标准字典，不再重复复制整批数据。
Decoder 只读借用中间态，在实际生成可变公开内容时隔离对应 payload；
同协议转换也必须显式翻译字段，不能直接返回原生对象快照。
不要通过修改输入块补齐字段，应重建实际变化的字典或列表。

Decoder.request_parameters(profile, options) 将已验证的 PreparedOptions 映射为原生字段，
assemble_request(messages, tools, profile, *, options, **parameters) 将原生消息、工具和选项放入原生请求结构。
Decoder.request 同样必须显式接收 options；这些入口不接受缺失选项后重新解析的兼容路径。
PreparedOptions 是公开的请求独立选项类型，字段与嵌套内容按只读借用；
生成可变请求时复制所需字段。bindings 保存按原顺序展开占位符的绑定，并保留 omit 删除标记；
不再分别保存 raw / expanded 两套副本。五协议统一使用 codec.shared.options.merge_bindings 递归合并绑定，
在合并对应层时消费 omit，保留未修改的兄弟字段；列表和标量整体替换。
Profile 只为 reasoning_enable_parameter / reasoning_effort_parameter 延迟消费删除标记，
其余配置仍按既有按键覆盖规则合并。删除标记不发送给 SDK；生成参数优先级和必需字段校验仍由协议实现负责。
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
| validate_parameters(profile, **parameters) | 验证并返回 PreparedOptions；协议在父类结果上检查原生参数绑定和能力限制，并返回准备好的选项。 |
| validate_request(request, profile, **parameters) | 必需。验证原生选项，按 SDK 要求构造公开请求类型。 |
| create_client(profile, **parameters) | 必需。同步创建官方 SDK 客户端。 |
| upstream(client, request, profile, **parameters) | 必需。执行请求，返回清理本次响应资源的异步上下文管理器。 |
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
defaults 使用 frozen dataclass ProtocolDefaults，七个字段都必须显式声明：
两个布尔开关、三个字段路径、两个推理绑定；不用的路径或绑定设置为 omit。
嵌套字典仍按只读借用，Profile 解析会复制这些声明。

完成自己的 Encoder、Decoder、Transport 后，组合和注册如下；将 your_package 替换为实际包名：

```python
from my_llm import (
    BaseCodec, ModelProfile, MyCodec, MyLLM,
    ProtocolDefaults, ProviderProfile, omit,
)
from your_package.decoding.decoder import ExampleDecoder
from your_package.encoding.encoder import ExampleEncoder
from your_package.transport import ExampleTransport

class ExampleCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "example"
    defaults = ProtocolDefaults(
        replay_reasoning_content_when_tool_calls=False,
        supports_reasoning_effort=False,
        max_tokens_path="$.max_tokens",
        content_path=omit,
        reasoning_content_path=omit,
        reasoning_enable_parameter=omit,
        reasoning_effort_parameter=omit,
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
的协议名一致性。四个实现都须声明空 `__slots__`，不能有实例 `__dict__`；
协议名须为非空字符串，重名注册抛出 ConfigurationError，不覆盖已有实现。
MyCodec / MyLLM 实例保留创建时的注册快照；注册新协议后需创建新实例才能使用。

## 中间态与字段边界

消息使用 role 与 content 块；工具调用与结果使用 tool_use / tool_result。
它是内部开放字典，不是新的应用消息类型。

- 中间态只包含已识别字段；未知输入键不复制到中间态，未知中间态键不合并进输出。
- protocol 字符串记录必要的来源语义，Responses 的 coalesce 标志控制已识别内容的合并；不保存原始对象或事件。
- functions 表示规范化工具声明，参数、结果及工具 JSON Schema 的开放键值完整保留。
- conversion.encode_functions / decode_functions 管理工具声明；stream 管理原生事件操作与用量计数。
- options 管理配置字段路径与生成选项；transport 提供 SDK 字段过滤、工具关联验证及响应资源关闭。
不能把其他协议的密文解释为本协议的签名，也不通过隐藏记录恢复跨协议无法表达的信息。
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
