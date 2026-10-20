# 翻译字段边界

项目接收已支持的标准协议，通过 Profile 明确配置兼容服务差异。翻译仅处理已识别的字段，
不把原生字典中的全部信息搬运到中间态，也不以隐藏记录提供跨协议恢复能力。
Encoder / Decoder 直接读取并构造已支持的字段，不扫描、收集或记录未识别字段。

## 输入与输出

- Encoder 将已识别的消息、工具和事件字段翻译为中间态。未知字段和未知原生事件不进入中间态。
- Decoder 从中间态构造目标协议已识别的字段，不把中间态中的未知键合入输出。
- 同协议转换也执行这些规则，不以原始对象快照绕过翻译。
- Decoder 在 SDK 请求边界统一过滤、验证消息与工具结构，并执行 Profile 明确声明的兼容策略。

SDK 的类型声明用于 Decoder 的结构过滤和验证，不是翻译器自动支持该类型全部字段的证明。
SDK 模型的 `extra="allow"` 也不构成未知字段的通用透传入口。

## 开放字典

已识别字段内部如果本来就是开放应用数据，其内部键值随整个字段保留，例如工具参数、
工具结果对象、工具 JSON Schema 和协议明确支持的 metadata。

```text
已识别的工具 parameters
  └── properties.vendor_key       保留：工具 JSON Schema 内部的应用字段

未知消息字段 vendor_options
  └── properties.vendor_key       忽略：外层字段没有被翻译器识别
```

数据在运行时同为 dict，不意味着字段语义相同。TypedDict 或 Pydantic 模型描述的协议结构
只处理已声明、已识别的字段；开放应用字典才允许内部任意键。

## 兼容配置

Profile 提供明确的兼容入口：参数路径和缺失默认值、extra_body、extra_headers、内容与思考增量路径、
思考内容回放及 token 参数路径等。配置每次按当前请求生效，不由首条输入的形状决定。

这些入口各有作用范围。例如 `content` / `reasoning_content` 的 encode 参数读取配置的返回字段，
不会把任意历史消息字段加入放行范围；Chat / Responses 的 extra_body 通过 SDK 的扩展请求体
机制发送，内部键不套用消息字段过滤。其他协议仍遵循各自 SDK 支持的参数接口。
新的兼容参数通过继承 Profile 声明 `Parameter[T]`，将路径、缺失默认值和阶段保存在一起；
新的控制能力通过 `Config[T]` 声明并由协议实现消费，不能依靠未知字段透传。

具体字段和落点见[配置指南](../reference/configuration.md#sdk-参数与协议字段)。

## 签名、密文与工具身份

Anthropic thinking.signature、redacted_thinking、Gemini thought_signature、Responses
encrypted_content 和工具关联 ID 都属于有明确语义的标准字段，由对应协议实现处理。
同协议路径保留实现支持的原生值；外来思考按目标协议可表达的形式翻译，不伪造签名或解密密文。

不在其他协议输出中附加原始对象、剩余字段或恢复快照。A → B → A 只处理每一步明确支持的语义，
不能用未修改结果的隐藏记录恢复 B 无法表达的原生信息。

## 缓存

Schema 缓存保存 SDK 类型结构和过滤函数，不保存输入字段样本或请求值。某个可选字段在第二次
请求才出现，不影响类型计划；能否进入翻译结果取决于协议实现及 Profile 的明确支持。

已识别开放字典内部的新键随当前值处理，不需要为新键重新编译。缓存不会为未识别字段新增
放行规则，也不负责恢复任何转换前的数据。详见[Schema 过滤计划缓存](schema-filter-cache.md)。
