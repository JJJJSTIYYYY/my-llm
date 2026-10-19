# 配置指南

`ProviderProfile` 描述连接和上游 API；`ModelProfile` 描述模型与覆盖项。
MyLLM 构造时按协议默认值 → provider → model 合并，保存两份 profile 副本和一份
`resolved_profile: ResolvedProfile`。generate 不接受动态 provider/model 或连接参数。
协议默认值由各 Codec.defaults 的 ProtocolDefaults 声明，解析从注册表复制，
不创建 Codec 或 SDK 客户端。新协议须先通过 MyCodec.register(codec) 注册，再解析配置或创建 MyLLM。

```python
from my_llm import ModelProfile, MyLLM, ProviderProfile

llm = MyLLM(
    provider=ProviderProfile(
        protocol="openai_responses", timeout=30, max_retries=2,
        usage_mode="native",
    ),
    model=ModelProfile(model="your-model", max_tokens=2048),
    stdio_protocol="openai_chat",
)
```

## 继承和省略

None 继承上层值；`openai.omit`（也可从 my_llm 导入）禁用继承值。
字典按键覆盖，嵌套值整体替换；字典中的 omit 删除该键。

`reasoning_enable_parameter` / `reasoning_effort_parameter` 是待应用的参数绑定模板，
其中的 omit 会保留到请求准备阶段。五协议均先应用启用绑定、再应用强度绑定：
绑定字典递归按键合并，omit 删除对应深度的旧键，未修改的兄弟键保留；列表和标量整体替换。
后续绑定可以显式重新设置已删除的键。这个规则同样适用于绑定内的 extra_body，
最终 SDK 请求不会携带这些删除标记；协议核心字段、必需参数和生成参数优先级仍受各协议约束。
Profiles 是 frozen dataclass；使用 `dataclasses.replace()` 创建新配置，再创建 MyLLM。
嵌套可变值在 MyLLM 构造时隔离，每次生成也隔离消息、工具及生成选项。

```python
from my_llm import ModelProfile, ProviderProfile, omit, resolve_profiles

provider = ProviderProfile(extra_body={"a": 1, "b": 2})
model = ModelProfile(model="demo", extra_body={"a": 3, "b": omit})
assert resolve_profiles(provider, model).get("extra_body") == {"a": 3}
```

## Provider 字段

| 字段 | 默认值 / 说明 |
| --- | --- |
| `protocol` | `openai_chat`；接受已注册协议名，内置另有 openai_responses / anthropic / gemini / ollama |
| `base_url` | None，官方 SDK 默认地址 |
| `api_key` | None，使用 SDK 环境变量等默认行为；不出现在 profile repr |
| `timeout` | None，SDK 默认；显式值必须为正秒数 |
| `max_retries` | None，SDK 默认；显式值必须是非负整数 |
| `supports_stream` | 解析后 True；False/omit 不支持，因为两种返回模式均用上游流 |
| `usage_mode` | `none`；Responses/原生协议使用 `native`，Chat 使用 `stream_options`；扩展模式由协议 Transport 声明 |
| `models` | None，可选 ModelProfile 列表；生成仍须显式选择模型 |

只有一个 `protocol` 字段，没有单独的 API 类型选择。
Protocol 是 str 别名；未注册的协议名在配置解析时拒绝。
usage_mode 必须为非空字符串，具体值由 Transport.usage_modes 校验，新增协议可自行声明模式。
`stdio_protocol` 是 MyLLM 的输入输出协议，不属于 provider。
SDK 客户端由 MyLLM 实例延迟创建、复用并统一关闭；响应流按请求关闭，不接受自定义 client。
Ollama SDK 没有原生 max_retries 选项；None / omit / 0 保留原生行为，正数在构造时明确拒绝。

## Model 字段

| 字段 | 默认值 / 说明 |
| --- | --- |
| `model` | 必填，非空模型 ID |
| `max_context_window` | 256000，仅描述性整数元数据，不截断输入或限制输出 |
| `max_tokens` | None，正整数输出上限；可由 generate 覆盖，omit 取消 |
| `supports_reasoning` | 解析后 True |
| `supports_function_calling` | 解析后 False，发送非空 tools 前须开启 |
| `supports_message_name` | 解析后 True；False/omit 在 Chat 请求移除 name |

`ModelProfile.supported_roles` 是固定类属性，不是构造参数。
Provider 的 `add_models(ModelProfile | list[ModelProfile])` 按 ID 更新并保留顺序；
`list_models()` 返回列表副本；`delete_model(list[str])` 忽略不存在的 ID。
模型字符串不会触发 catalogue 查找。

## Provider / Model 共享覆盖项

`ProviderProfile` 和 `ModelProfile` 继承公开的 `OverridableProfile`，以下字段可在两层设置。

| 字段 | 说明 |
| --- | --- |
| `extra_headers` | 请求 headers 按键合并 |
| `extra_body` | 协议原生生成参数按键合并 |
| `supports_reasoning_effort` | Responses 默认 True，其他默认 False |
| `reasoning_enable_parameter` | 开启思考时注入的 SDK 参数字典 |
| `reasoning_effort_parameter` | 支持强度时额外注入的 SDK 参数字典 |
| `reasoning_effort_mapping` | none/low/medium/high/xhigh/max 映射；None/omit 禁用某级 |
| `max_tokens_path` | `max_tokens` 在 SDK 请求参数中的写入路径；None 继承协议默认值，omit 禁用自动写入 |
| `content_path` | OpenAI 文本增量路径或 Responses 事件类型 |
| `reasoning_content_path` | 兼容 Chat 服务的可见思考扩展路径或 Responses 事件类型 |
| `replay_reasoning_content_when_tool_calls` | Chat 默认 False，Ollama 默认 True；兼容 Chat 服务显式启用后回放可见思考，仅思考开启且当条有工具调用时补空值 |

Chat 默认文本路径 `$.choices[0].delta.content`，思考路径
`$.choices[0].delta.reasoning_content`；该字段属于兼容服务扩展，并非官方 OpenAI Chat 字段。Responses 默认事件为
`response.output_text.delta` / `response.reasoning_summary_text.delta`。
`$.` 路径支持属性/字典键与数值索引；omit 禁用路径。
原生协议使用原生字段（例如 Anthropic thinking、Gemini thought、Ollama thinking）。

`max_tokens_path` 从传给 SDK 的参数字典根部开始，支持与 content_path 相同的
点分键名和数值索引，`$.` 前缀可省略。例如 `$.max_tokens` 直接传 SDK 参数，
`$.extra_body.max_tokens` 或 `$.extra_body.max_completion_tokens` 写入 extra_body。
协议默认路径分别为 Chat 的 `$.max_completion_tokens`、Responses 的 `$.max_output_tokens`、
Anthropic 的 `$.max_tokens`、Gemini 的 `$.config.max_output_tokens` 和 Ollama 的 `$.options.num_predict`。
自定义路径替代默认写入位置，不会额外添加默认 token 参数。目标路径的值优先于
extra_body 和 reasoning 绑定中的同名值，其他字段保留；路径不能覆盖协议核心字段。
generate 的 max_tokens 仍覆盖模型值；omit 禁用本次自动写入，不删除手动配置的原生参数。
目标路径和键名须符合对应 SDK 的接口要求；Anthropic 仍要求提供 max_tokens。

```python
provider = ProviderProfile(max_tokens_path="$.extra_body.max_tokens")
model = ModelProfile(
    model="your-model",
    max_tokens=4096,
    max_tokens_path="$.extra_body.max_completion_tokens",
)
```

replay=False/omit 在 Chat/Ollama 请求中去掉可见思考字段；True 保留已有内容。
官方 Chat 默认关闭该兼容字段回放，避免生成官方未定义的 reasoning_content；Ollama thinking 是原生字段，默认保留。
没有思考内容时，普通 assistant 消息保持不带思考字段；仅在思考开启且有工具调用时
分别补 `reasoning_content=""` / `thinking=""`。签名回放保持协议原生块。

## SDK 参数与参数内部的协议字段

项目按字段所在的位置、协议实现明确识别的字段以及 SDK 类型处理数据。
标准协议的已识别字段无需逐项配置；非标准字段只通过 Profile 已提供的兼容能力处理。
Profile 不是任意字段的通用放行表，SDK 接受某个扩展也不代表翻译层会识别它。
SDK 参数与协议字段也不是互斥的概念：例如 messages 是 SDK 调用参数，同时对应 HTTP 请求体的字段，
而 messages 中各项的 role / content 是消息内部字段。

以 Chat 的 SDK 调用参数为例：

```python
sdk_arguments = {
    "model": "your-model",
    "messages": [{"role": "user", "content": "hello"}],
    "extra_body": {"vendor_options": {"enabled": True}},
    "extra_headers": {"X-Client": "example"},
}
```

这里 model / messages / extra_body / extra_headers 是传给 SDK 方法的参数。
messages 内部的 role / content 按消息类型处理；extra_body 内部是额外 JSON 请求体，
extra_headers 内部是 HTTP headers，都不套用消息字段的过滤规则。

| 层级 | 项目如何确定边界 | 处理位置 |
| --- | --- | --- |
| SDK 客户端参数 | Profile 连接字段及各协议的客户端构造规则 | Transport.create_client；例如 api_key、base_url、timeout。 |
| SDK 请求参数 | Decoder 构造的请求字典；协议 Transport 的允许参数、保护字段和 SDK 调用接口 | validate_parameters / validate_request 及 SDK 方法；例如 model、messages、max_tokens。 |
| 消息、工具内部字段 | Transport 显式声明的 message_path / tool_path，以及 message_type / tool_type | BaseTransport 定位对应值，再按缓存的 SDK 类型计划过滤和验证。 |
| 其他结构参数内部字段 | 对应协议显式绑定的 SDK 类型 | 例如 Anthropic thinking、Responses reasoning、Ollama options 的类型处理。 |
| 开放应用数据 | 字段声明为 dict / Any，或协议明确保留的应用字典 | 保留内部键值，例如工具参数、工具结果和 JSON Schema；不按 role / type 等键名猜测其协议。 |

消息和工具的路径由各协议实现明确给出：

| 协议 | 消息路径 | 工具路径 |
| --- | --- | --- |
| Chat | messages | tools |
| Responses | input | tools |
| Anthropic | messages | tools |
| Gemini | contents | config.tools |
| Ollama | messages | tools |

例如 Chat 指定 message_path=("messages",)，message_type=ChatCompletionMessageParam，
过滤器据此分析每条消息，而不是对整个 SDK 请求字典套用消息类型。
Gemini 先定位 config.tools，才按 Tool 类型处理其中各项；config 的其他生成参数另行验证。
这些根类型由协议实现者根据 SDK 接口选择，缓存不自动搜索 SDK 或证明选型完整。

生成参数的允许范围也由各协议确定：Anthropic 读取 AsyncMessages.create 的参数签名，
Ollama 读取 AsyncClient.chat 的参数签名，Gemini 读取 GenerateContentConfig.model_fields，
再排除库管理的参数。Chat / Responses 对推理绑定使用显式允许键，并检查保护字段；
实际 SDK 调用参数仍须符合各自 SDK 方法的接口。

Profile.extra_body 在不同协议中的落点不同：Chat / Responses 写入 SDK 的 extra_body 参数，
由 SDK 合入 JSON 请求体；Anthropic / Ollama 作为原生生成参数，Gemini 作为 config 字段。
因此，Chat / Responses 的 extra_body 可以携带额外 JSON 键；其他协议仍检查当前 SDK 支持的生成字段。
这些配置都不能覆盖库管理的 messages / input / contents / tools 等核心结构。

翻译层只构造已识别的协议字段，不复制未知字段或原生对象作为恢复记录。
Transport 再按声明的 SDK 类型过滤并校验请求；TypedDict 和 Pydantic 协议结构只保留声明字段，
即使 SDK 模型配置 extra="allow"，也不将额外字段作为协议翻译的扩展入口。
已识别字段若声明为开放 dict / Any，其内部应用键值完整保留。
Schema 缓存只保存这些类型规则，不保存 Profile 或请求字段样本；兼容策略每次读取当前配置和当前值。
详细对象与执行过程见[Schema 过滤计划缓存](../design/schema-filter-cache.md)。

## 协议参数

| protocol | 输出 token 限制 | 思考 | tools |
| --- | --- | --- | --- |
| openai_chat | max_completion_tokens | 开启时默认 reasoning_effort=medium；支持强度时按映射设置 | function wrapper |
| openai_responses | max_output_tokens | reasoning.summary=detailed，effort 经映射 | 扁平 function |
| anthropic | max_tokens，必须设置 | thinking=enabled/adaptive，预算必须小于 max_tokens | name/input_schema |
| gemini | config.max_output_tokens | config.thinking_config，禁用 SDK 自动工具执行 | function_declarations |
| ollama | options.num_predict | think=True 或模型原生字符串 | function wrapper，arguments 为对象 |

Anthropic 默认 thinking budget_tokens=1024，启用时要求 max_tokens>1024。
Gemini 默认 include_thoughts=True 与 thinking_budget=-1；支持强度时使用 thinking_level。
显式开启 Ollama 推理时默认生成 think=True；未指定 reasoning / 非 none effort 时不自动增加 think。
强度映射由具体模型决定，例如 low/medium/high，不限定为全局枚举。
原生参数通过 extra_body 配置，使用各 SDK 原生字段，不设计通用 thinking 对象。
Anthropic 的 tool_choice / metadata 和 Gemini 的 tool_config / safety_settings 会保留到 SDK 请求。
Anthropic / Gemini / Ollama 接收当前安装 SDK 支持的生成字段；不支持的字段会抛出
ConfigurationError，不再静默丢弃。Gemini 的 automatic_function_calling 与 http_options
由库管理，不能通过 extra_body 覆盖；HTTP headers 使用 extra_headers。

Chat 默认使用官方 reasoning_effort 参数；兼容服务的 thinking / enable_thinking 扩展需要显式配置。
覆盖 reasoning_enable_parameter 时仍逐键继承默认值；如只使用扩展，需在该字典内设置
reasoning_effort=omit，再添加 extra_body 绑定。

```python
from my_llm import ModelProfile, MyLLM, ProviderProfile

anthropic = MyLLM(
    provider=ProviderProfile(protocol="anthropic"),
    model=ModelProfile(model="your-model", max_tokens=4096),
    stdio_protocol="anthropic",
)
gemini = MyLLM(
    provider=ProviderProfile(protocol="gemini"),
    model=ModelProfile(model="your-model", supports_function_calling=True),
    stdio_protocol="gemini",
)
```

生成的推理参数优先于 extra_body 同名键。extra_body 和 reasoning 绑定不能覆盖
model / stream / messages / input / tools 等核心协议字段。
SDK 会验证目标协议具体参数和字段；兼容配置须符合当前 Transport 与 SDK 的调用接口。

## 跨协议请求的处理边界

Encoder 只将明确识别的标准字段和已配置兼容字段翻译为中间态；Decoder 只构造目标协议
已识别的字段。两个方向都不透传未知协议字段，也不保存原始对象、剩余字段或跨协议恢复记录。
工具参数、结果对象和 JSON Schema 等已识别的开放字典完整保留内部应用键值。
BaseTransport 在 SDK 请求边界统一过滤并校验原生结构，不修改调用者传入的历史。
完整规则见[翻译字段边界](../design/translation-boundaries.md)。

外来思考的可见内容在 Chat / Responses / Anthropic / Gemini 中按普通文本表达，在 Ollama 中可使用
原生 thinking；原生签名、redacted_thinking、encrypted_content 按对应协议的明确字段处理，
不伪造签名、不解密 reasoning，也不在其他协议的隐藏记录中保存它们供后续恢复。Gemini 的外来工具历史
使用其[官方迁移标记](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures#faqs)
`skip_thought_signature_validator`；原生 Gemini 历史的签名和并行调用位置不改变。
请求能力、生成选项、SDK 表达限制及思考回放开关均由 Transport 校验；Encoder / Decoder 只翻译已识别的数据。

Chat / Ollama 不能把 URL 文档作为原生文件读取时保留文本引用；没有隐式下载或文件系统读写。
必要的不透明内容、不能安全表达的媒体或 SDK 会静默丢弃的工具 Schema 约束，
会在请求前抛出 UnsupportedCapabilityError，避免发送已知不合法或已被改变约束的请求。
该策略检查格式与已知表达能力，不替代目标模型能力、服务端上下文、鉴权与配额校验。

兼容 Chat 服务若需要 reasoning_content 回放，应显式配置：

```python
from my_llm import ProviderProfile, omit

compatible = ProviderProfile(
    protocol="openai_chat",
    replay_reasoning_content_when_tool_calls=True,
    reasoning_enable_parameter={
        "reasoning_effort": omit,
        "extra_body": {"thinking": {"type": "enabled"}},
    },
)
```

Ollama 当前 SDK 不提供重试次数配置；max_retries 仅接受 None / omit / 0，正数在构造时明确拒绝。
