# 接入新的协议

转换始终经过中间态：`源原生输入 → 源 Encoder → 中间态 → 目标 Decoder → 目标原生输出`。
实现只理解自己的原生格式与中间态，不读取另一个协议的 Encoder / Decoder，
也不在 MyCodec、MyLLM 或 shared 中增加厂商判断。

## 文件边界

每个协议保留一个 codec.py，以及 encoding / decoding 两个方向。
主类分别放在 encoder.py 和 decoder.py；消息、工具、事件和用量的方法可在同一类中阅读。
本方向复用的内容块、字段、工具调用及 Schema 辅助放在 content.py；
SDK 的打开和关闭独立放在协议根目录的 transport.py，与 encoding / decoding 并列，
由 Codec 组合其 Transport 实例。没有对应辅助逻辑时不创建空模块。
__init__.py 只导出接口。

## 必需的抽象接口

[BaseEncoder](../../src/my_llm/codec/base/encoder.py) 的输入是本协议原生值，输出是中间态。

| 方法 | 职责 |
| --- | --- |
| match_item(item: dict) → bool | 判断单条原生消息或工具的形状。整批遍历、空输入和畸形容器处理由继承的 matches(values) 完成。 |
| messages(values, profile=None, **parameters) → list[dict] | 消息及内容块转中间态，保留原生来源与未知字段。 |
| tools(values, profile=None, **parameters) → list[dict] | 工具声明转中间态。 |
| chunks(values, profile=None, **parameters) → list[dict] | 原生事件转中间态操作；不累积消息、文本或工具参数。 |
| usage(value: dict \| None) → dict \| None | 原生计数转统一用量，同时保留原始计数；None 表示尚无用量。 |

[BaseDecoder](../../src/my_llm/codec/base/decoder.py) 的输入是中间态，输出是本协议原生值。

| 方法或属性 | 职责 |
| --- | --- |
| messages / tools / chunks / usage | 签名与 Encoder 对称，方向相反。只根据中间态和本协议的表达能力生成目标值。 |
| request_parameters(profile, **parameters) → dict | 生成参数映射、受保护字段、推理绑定和 SDK 可接受的选项。 |
| assemble_request(messages, tools, profile, **parameters) → dict | 接收已经解码的原生消息和工具，调用 request_parameters，放置 system / history / tools 等字段并校验原生服务端历史参数。 |
| profile_defaults() → dict | 返回新的默认配置字典；不绑定 provider 或 model。 |
| usage_modes: tuple[str, ...] | 声明支持的用量策略。 |

不覆盖 BaseEncoder.matches 或 BaseDecoder.request；这两个入口用 `typing.final` 标明固定流程。
request 的入参始终为中间态，执行顺序为能力检查、messages、tools、assemble_request。
tools=None 表示未提供，tools=[] 表示显式空列表，两者保持区别。
MyCodec.request 只选择目标 Decoder，不再预先解码消息或工具。

Decoder 还有三个按需覆盖的默认钩子：validate_profile 调用通用用量策略校验，
可扩展原生限制；result 默认返回首条完成消息，可改为原生列表形状；
allows_external_calls 默认 False，支持服务端历史时才覆盖。

[BaseTransport](../../src/my_llm/codec/base/transport.py) 独立负责官方 SDK 的生命周期与原生请求执行，
不读取中间态，也不引用 Encoder / Decoder：

| 方法 | 职责 |
| --- | --- |
| create_client(profile, **parameters) | 必需。同步创建官方 SDK 客户端，MyLLM 负责保存与复用。 |
| upstream(client, request, profile, **parameters) | 必需。借用客户端，执行已组装的原生请求，返回负责关闭本次响应流和响应体的异步上下文管理器。 |
| close_client(client) | 默认 await client.close()；需要分别关闭同步与异步 SDK 部分时覆盖。 |
| client_key(profile, **parameters) | 默认 None；SDK 将请求头等选项绑定到客户端时返回不可变缓存键。 |
| validate_profile(profile) | 默认无操作；校验 SDK 连接、重试等限制，不创建客户端。 |

客户端关闭由 MyLLM 调用 Transport.close_client；不要把响应、工具参数或序号放进客户端缓存键。
BaseCodec.validate_profile 协调 Decoder 的转换策略校验与 Transport 的连接校验。
详见[客户端生命周期](../design/client-lifecycle.md)。

[BaseCodec](../../src/my_llm/codec/base/codec.py) 要求提供 protocol_or_api_style、encoder、decoder 和 transport。
可以用类属性满足这四个抽象属性，与内置实现一致：

```python
from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.openai.chat.decoding.decoder import ChatCompletionsDecoder
from my_llm.codec.openai.chat.encoding.encoder import ChatCompletionsEncoder
from my_llm.codec.openai.chat.transport import ChatCompletionsTransport

class ChatCompletionsCodec(BaseCodec):
    __slots__ = ()
    protocol_or_api_style = "openai_chat"
    encoder = ChatCompletionsEncoder()
    decoder = ChatCompletionsDecoder()
    transport = ChatCompletionsTransport()
```

新协议用自己的实现替换示例中的身份与三个实现类即可。缺少必需接口的类不能实例化。
Encoder / Decoder / Transport / Codec 均使用空 __slots__，只声明方法、不可变策略和无状态实现；
不保存 profile、识别结果、响应 ID、流序号或片段。profile 按调用传入，None 表示标准协议；
所有调用均不得修改输入字典或解析后的配置。请求状态由 StreamState 持有。

## 中间态约定

中间态是 codec 内部开放字典，不是新增的应用消息协议。应复用现有机制，
避免另造一套协议间交换格式：

- 消息使用 role 和 content 块；文本为 `{"type": "text", "text": "..."}`，工具调用与结果使用 tool_use / tool_result 块。
- [conversion.py](../../src/my_llm/codec/shared/conversion.py) 的 dictionaries 负责值展开与复制，remember / original 负责来源快照和同协议回放，residual 负责原路径的未消费字段。
- [tools.py](../../src/my_llm/codec/shared/tools.py) 的 encode_functions / decode_functions 管理统一工具声明和不透明工具的拒绝；保留键从 [keys.py](../../src/my_llm/codec/shared/keys.py) 导入。
- [chunks.py](../../src/my_llm/codec/shared/chunks.py) 的 batch 包装原生事件与操作，BlockRef 标识来源块；块、快照、终态和请求累积规则见[流状态设计](../design/stream-state.md)。
- [usage.py](../../src/my_llm/codec/shared/usage.py) 的 record / counts 管理统一计数组，不凭空补充供应商未报告的计数。

同协议回放保留原生扩展、签名和不透明值；跨协议只输出目标能够表示的字段。
源编码不根据目标格式丢字段，目标解码不借用其他协议的转换逻辑。
用同方向的 self.messages / self.usage 或 content 辅助复用逻辑。

## 注册与验证

1. 实现两个转换方向与 Transport，由 Codec 组合，在 [MyCodec 注册表](../../src/my_llm/codec/converter.py) 加入实例；匹配优先级由注册顺序决定。
2. 在 [configuration.py](../../src/my_llm/profile/configuration.py) 的 Protocol / PROTOCOLS 中加入公开身份，同步包导出和协议文档。MyLLM、公共流状态和 shared 无需增加原生协议分支。
3. 独立编写原生消息、工具和流样例，不用待测转换器生成预期。新增协议与全部已有协议双向验证消息、工具、完整流和请求，并检查同协议扩展保留、未知值拒绝、用量和终态。
4. 通过官方 SDK 和本地 HTTP transport 验证请求、正常完成、异常、取消、提前退出及并发隔离。无需调用真实模型服务。

基类完整性和请求流水线测试见 [test_native_codecs.py](../../tests/unit/test_native_codecs.py)，
方向与协议隔离测试见 [test_codec_boundaries.py](../../tests/contract/test_codec_boundaries.py)。
完整检查与发行包验证见[本地开发规范](development.md)和[构建指南](building.md)。

旧的 messages/、tools/、stream/、request/ 内部模块已合并，没有保留旧路径的转发。
自定义实现需将 matches 改为 match_item，将原先接收原生历史的 request 改为 assemble_request，
并提供 request_parameters；调用 Decoder.request 的代码应改传中间态。
SDK 客户端职责已从 BaseDecoder 迁至 BaseTransport。原先 Decoder 上的 create_client、
upstream、close_client、client_key 迁入 Transport，SDK 连接限制也在 Transport.validate_profile 校验；
由 Codec.transport 组合。旧 decoding/transport.py 路径已移除，无兼容转发。
