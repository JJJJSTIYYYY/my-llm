# 接入新的协议

每个协议实现 BaseEncoder、BaseDecoder、BaseTransport，由 BaseCodec 组合并声明 ProtocolDefaults。
Encoder 接收完整调用字典，验证和编码输入；Decoder 准备客户端参数、请求参数和本地控制项；
Transport 只构造／调用 SDK、关闭客户端及本次响应。四个基类、PreparedOptions、Parameter 和 Config 均从 my_llm 导入。

## 目录与实例生命周期

协议包包含 codec.py、transport.py、encoding/encoder.py、decoding/decoder.py；内容辅助放在各方向 content.py，
双向共用规则按职责直接放在协议根目录。例如 Ollama 的公开 base64 Image 验证和 SDK 方法类型分析放在 schema.py，不再为单个模块建立 utils/。
注册工厂每次创建独立 codec 与组件，允许实例状态；不能共享 adaptor 或可变组件。初始化文件只导出接口。
不导入其他协议实现，也不让 Encoder / Decoder 相互导入或依赖 Transport。
官方 SDK 类型和方法签名可用于静态 Schema 分析；SDK 客户端创建和调用只能出现在 Transport。

## Encoder 与 Decoder 契约

| 实现 | 必需接口 |
| --- | --- |
| BaseEncoder | protocol、message_type / tool_type、match_item、_messages / _tools / _chunks、usage。 |
| BaseDecoder | protocol、request_schema、message_type / tool_type、message_path / tool_path、_messages / _tools / _chunks、usage、assemble_request。 |
| BaseTransport | create_client(parameters: dict)、upstream(client, request: dict)。 |

属性可用类属性实现。messages / tools / chunks 是固定公开入口，私有钩子的签名为
(values: list, profile: ResolvedProfile | None = None, **parameters)。
Encoder.matches 负责整批识别。validate_input 是固定公开入口，先通过 _validation_messages / _validation_tools 丢弃未支持原生字段，再使用公开 SDK 类型只读验证；自定义输入语义通过这些投影钩子和 validate_values 实现。request 复用已归一化视图执行同一组验证钩子，不重复进入公开输入边界。
Encoder.request(inputs, profile) 接收完整 generate 输入并输出包含 profile 和类型化 middleware 的调用信封。

两个方向的私有转换钩子只读借用输入，只处理已识别字段，不保留未知字段或原始快照。
Decoder 的消息循环通过 `codec.shared.input.message_feedback(messages, profile)` 获取
`(index, message, feedback)`，在原有内容循环内调用 `input_content(block, profile, feedback=feedback)`，
返回 False 时跳过该块。工具结果中的图片也在翻译子内容时调用 `input_content`。
`codec.shared.capabilities` 统一判断内容能力并记录不支持的原因；原生值可传入内容种类与来源路径复用检查。
`message_feedback` 在一次分组中完成消息合并与回传判断；需要保留原生 item 身份时传 `coalesce=False`。
回传策略排除的思考不记录损失；能力损失与协议原生限制一同交给完整 draft 的 adaptor 处理。
公开 messages / tools 捕获独立结果；内部 Decoder.request 直接调用私有钩子，
由最终 native_request 捕获 SDK 需要的可变内容，避免先复制公开转换结果再复制请求。
工具参数、结果和 JSON Schema 的开放字典应完整保留内部应用键。
两个方向的 _chunks 直接构造独立事件，只复制借入的可变叶；不可变文本不需要再次复制。
修改调用方输入不能改变返回结果，修改结果也不能改变调用方输入。
Decoder.result 默认返回首条消息；Responses 等协议可覆盖为列表。

## Decoder 的固定请求流程

Decoder.prepare(intermediate, client) 返回 {client, request, local} 三部分字典：
client 是 SDK 构造参数，request 是 SDK 调用参数，local 包含 stream 和 client_key。
完整准备路径每次只调用一次 validate_parameters 返回 PreparedOptions，再交给 request。
离线 Decoder.request 可自行准备一次 options；已提供 options 时直接复用。

request 依次翻译消息／工具、request_parameters、assemble_request、native_request。
`_messages` 接收已验证的 `Iterable[dict]`，在原有循环中读取即可；不要预先转成列表或重复消费。
修复视图会在该次迭代中省略已丢弃消息，并呈现必要的空工具结果。
assemble_request(messages, tools, profile, *, request: dict, **parameters) 只写入消息和工具，
不重新调用 request_parameters；协议控制通过 _request_parameters(request, profile, options) 补充。
布局通过 body_container、headers_field、body_overlay、omit_empty_body 声明。
默认值只填补缺失参数，已有 None、False、0 保留，显式生成控制覆盖已有值；omit 语义见配置指南。

| Decoder 接口 | 职责 |
| --- | --- |
| request_schema | SDK 根参数及类型；通常用 parameter_types(SDK方法) 读取一次。 |
| message_type / tool_type | SDK 结构 Schema，与 Encoder 的输入 Schema 分别声明。 |
| message_path / tool_path | 原生请求中消息与工具的路径，支持嵌套 config.tools。 |
| validate_parameters | 共同选项每次准备一次，协议检查能力、保护字段和路径。 |
| native_request | 固定流程：历史关联 → 根过滤并委派消息／工具 → 叶过滤与验证 → validate_request → 显式兼容叶。 |
| filter_messages / filter_tools | 只读借入原生叶，返回可独立修改的 SDK 值；根计划不会重复过滤同一叶。 |
| _filter_message(message, policy, *, path, decode, feedback, **parameters) | 在基类的共同消息循环中处理一条消息；通过 InputPolicy 检查能力并调用 decode 建立 SDK 字段副本，返回 None 表示省略。 |
| _message_inputs(messages, policy) | 默认逐条产生 (index, message, feedback)；需要跨原生 item 判断推理反馈时可覆盖分组逻辑。 |
| message_delegated_paths | 消息过滤中需要由协议钩子捕获的叶路径，例如 Ollama 的 images。 |
| validate_request | 可覆盖的原生选项和组合约束检查，返回字典。 |
| validate_profile / usage_modes | 构造期检查 SDK 路径与协议连接配置；声明支持的用量模式，不兼容模式在 Decoder 准备请求时记录损失。 |
| client_parameters(profile) | 构造期把已解析连接配置映射到 SDK 构造 kwargs，不创建 SDK 客户端。 |
| request_client(client, options) / client_key(client) | 默认复用实例字典及 None 键；SDK 把请求选项绑定客户端时生成请求独立变体。 |
| allows_external_calls | 默认 False；服务端历史协议按需覆盖。 |

不覆盖 final 的 request / prepare / native_request。tools=None 省略，tools=[] 显式发送空列表。
BaseCodec.validate_profile 委托 Decoder；MyLLM 在构造时调用，并准备连接字典。
MyCodec.request 是离线原生请求入口，共用 Decoder 请求流程；完整调用使用 MyCodec.prepare。
中间消息的结构检查与工具关联收集共用一次遍历，关联错误在工具定义和控制项检查后报告。
五种内置协议共用 BaseDecoder.filter_messages 的外层循环，不根据翻译标记跳过能力处理。
InputPolicy 在本次过滤开始时读取固定的模型能力和推理反馈配置，逐条检查仍保留损失来源与路径。
message_transform、缓存和扩展 Decoder 产生的消息都经过该循环；返回后仍做最终 SDK Schema 校验，
也会检查 filter_messages 自定义覆盖所作的修改。
usage_mode 是开放字符串，接受的具体值由 Decoder.usage_modes 声明。
不支持的非空模式不在构造期拒绝，而在准备请求时记录语义损失；空字符串或错误类型仍是配置错误。

BaseEncoder.parameter_value(value, profile, field_id, default_path, default=None, *, use_default=True)
统一读取 encode 参数。协议钩子先识别事件语义，再读取指定字段；不枚举未知返回字段。
例如：

```python
def read_text_delta(encoder, event, profile):
    return encoder.parameter_value(event, profile, "content", "$.text")
```

Schema 类型计划复用 cache/compiled_cache.py 的 CompiledCache 有界 LRU，只保存类型及编译规则。
输入验证用只读 projected；Decoder 的原生 SDK 过滤使用计划 decode，隔离数据并记录无法匹配的值。
其他需要独立可变数据的过滤使用 supported；每份请求仍单独验证。
JSON Schema 的应用键不能作为协议字段删除。若先委派开放叶，协议钩子必须随后建立独立副本。

## Transport 契约

create_client(parameters: dict) 同步用已准备 kwargs 构造 SDK；默认 close_client 为 await client.close()。
upstream(client, request: dict) 返回异步上下文管理器，把 **request 交给官方 SDK，
完整消费、提前关闭、异常和取消时都释放本次响应，但不关闭借用的客户端。
Transport 不接收 Profile、生成控制参数，也不解析路径、过滤或校验请求。
MyLLM 根据 Decoder 生成的 client_key 管理实例缓存和最终关闭。
具体资源规则见[生命周期设计](../design/client-lifecycle.md)。

## Codec 组合

BaseCodec 必须提供 protocol_or_api_style、defaults、encoder、decoder、transport。
defaults 使用 frozen dataclass ProtocolDefaults。feedback_reasoning_content 通过 Config 明确声明；
max_tokens、content、reasoning_content、reasoning_enable、reasoning_effort 使用 Parameter 声明，不用的字段保留默认 omit。
max_tokens / reasoning_enable / reasoning_effort source 固定为 input，content / reasoning_content source 固定为 output。
路径从根字典定位，使用 $. 前缀，$ 表示 **parameters 解包字典的根；事件类型不是参数路径。
wrapper 的 name 与 Parameter.source 由 Profile 固定；comment 可用于展示说明。
Profile 字段 None 表示继承，wrapper.default_value=None 为实际值，仅在目标字段缺失时应用。

ProtocolDefaults、ProviderProfile、ModelProfile 可继承扩展：新增参数通过字段 metadata 硬编码 source="input" 或 source="output"，声明
Parameter[T] | Omit | None，分支配置声明 Config[T] | Omit | None。
解析器自动发现继承字段并检查声明，不需要路径与默认值配对，也不需要中央参数表。
自定义 Config 的含义由协议组件实现；所有请求 scope 的 input 参数路径和默认值由 BaseDecoder 自动应用。
自定义 output 参数只能映射为现有中间态语义，不能扩大固定支持集合。
支持字段无法表达时调用 my_llm.basic.semantic_loss.report_loss(value, reason)，继续完成 draft；
BaseDecoder 聚合损失，BaseCodec 统一执行策略，不在 encoder 报 SemanticLossError。
配置声明不支持的能力与字段也走该损失通道；Encoder 的非法输入结构使用 MessageValidationError。
`supports_stream` 是项目调用能力，禁用时仍直接拒绝生成。
SDK 根参数必须符合 SDK 调用签名，非标准字段使用 SDK 正式提供的 extra_body 等扩展入口。

完成自己的 Encoder、Decoder、Transport 后，组合和注册如下；将 your_package 替换为实际包名：

```python
from my_llm import (
    Attribute,
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
from copy import deepcopy

from your_package.decoding.decoder import ExampleDecoder
from your_package.encoding.encoder import ExampleEncoder
from your_package.transport import ExampleTransport


class ExampleCodec(BaseCodec):
    __slots__ = ("encoder", "decoder", "transport", "_defaults")
    protocol_or_api_style = "example"
    _DEFAULTS = ProtocolDefaults(
        feedback_reasoning_content=Config(default_value="none"),
        max_tokens=Parameter(
            path="$.max_tokens",
            default_value=omit,
        ),
    )
    def __init__(self):
        super().__init__()
        self._defaults = deepcopy(self._DEFAULTS)
        self.encoder = ExampleEncoder()
        self.decoder = ExampleDecoder()
        self.transport = ExampleTransport()

    @property
    def defaults(self):
        return self._defaults


# 在应用启动时执行一次，早于 Profile 解析及 MyCodec / MyLLM 实例创建。
MyCodec.register(ExampleCodec)
codec = MyCodec()
llm = MyLLM(
    provider=ProviderProfile(protocol=Config(default_value="example")),
    model=ModelProfile(model=Attribute(default_value="example-model")),
    stdio_protocol="example",
)
```

注册会检查 Codec 与三个组件的基类类型、ProtocolDefaults 类型，以及 Codec / Encoder / Decoder
的协议名一致性。工厂连续两次创建的 codec / encoder / decoder / transport 不得相同；
协议名须为非空字符串，重名注册抛出 ConfigurationError，不覆盖已有实现。
MyCodec / MyLLM 实例保留创建时的注册快照；注册新协议后需创建新实例才能使用。

## 中间态与字段边界

消息使用 role 与 content 块；工具调用与结果使用 tool_use / tool_result。
公开中间态是具体 ProtocolMiddleWare 及类型化模型；临时字典仅服务原生解析和组装。应用历史仍使用原生协议值。

- 中间态只包含已识别字段；未声明的非标准字段直接跳过，不读取、不校验，也不逐项扫描检查未知键。
- protocol 字符串记录必要的来源语义，Responses 的 coalesce 标志控制已识别内容的合并；不保存原始对象或事件。
- functions 表示规范化工具声明，参数、结果及工具 JSON Schema 的开放键值完整保留。
- conversion.encode_functions / decode_functions 管理工具声明；stream 管理原生事件操作与用量计数。
- 流控制数据只保留中间协议明确声明的 id、模型、终止与回放分组身份；不能通过 metadata 钩子扩展用户 metadata 支持。
- options 提供参数路径操作与生成选项；BaseEncoder / BaseDecoder 固定参数读取和请求应用流程，
  cache/compiled_cache 提供 Codec 共用的 Schema 规则；shared/conversion 验证历史关联；base/transport 集中保存执行契约与共用响应关闭辅助。
跨协议来回翻译不保证无损。不能把其他协议的密文解释为本协议的签名，也不通过隐藏记录恢复跨协议无法表达的信息。
标准签名与密文必须在对应协议实现中明确处理。详见[翻译字段边界](../design/translation-boundaries.md)。

## 注册与验证

1. 实现四个基类并通过 `MyCodec.register(CodecFactory)` 注册。Protocol 是 str 别名，
   不需要修改公共入口、协议枚举、默认值表或公共编排。
2. 独立编写原生样例，覆盖消息、工具、完整流、两个翻译方向未知字段不泄漏及开放字典内部键值保留。
3. 使用官方 SDK 和本地 HTTP transport 检查实际请求：Profile 兼容能力生效，标准签名按原协议处理，工具参数与 JSON Schema 不被误删。
4. 验证正常、取消、失败、提前退出、并发隔离和客户端关闭。

组件与依赖约束见[契约测试](../../tests/unit/test_native_codecs.py)和[边界测试](../../tests/contract/test_codec_boundaries.py)。
独立第六协议的注册、默认值、请求与流集成示例见[注册测试](../../tests/integration/test_protocol_registration.py)。
构建及完整验证命令见[开发规范](development.md)。
