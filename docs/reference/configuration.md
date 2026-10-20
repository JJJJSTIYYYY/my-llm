# 配置指南

`ProviderProfile` 描述连接和上游 API，`ModelProfile` 描述模型及覆盖项。
MyLLM 构造时按 `Codec.defaults → provider → model` 解析为独立的 `ResolvedProfile`。
协议默认值由各协议的 `ProtocolDefaults` 声明；新协议先注册，再创建 MyLLM。
generate 不接受动态 provider/model 或连接参数。

```python
from my_llm import Config, ModelProfile, MyLLM, Parameter, ProviderProfile

llm = MyLLM(
    provider=ProviderProfile(
        protocol="openai_responses",
        timeout=Config(name="超时秒数", default_value=30),
        max_retries=Config(name="重试次数", default_value=2),
        usage_mode=Config(name="用量模式", default_value="native"),
    ),
    model=ModelProfile(
        model="your-model",
        max_tokens=Parameter(
            name="输出上限",
            path="$.max_output_tokens",
            default_value=2048,
            phase="decode",
        ),
    ),
    stdio_protocol="openai_chat",
)
```

## 参数与行为配置

以下类型均可从 `my_llm` 导入，`Parameter` 与 `Config` 都是 frozen、kw_only dataclass：

| 类型 | 字段 | 职责 |
| --- | --- | --- |
| `Parameter[T]` | name、path、default_value、phase、comment=None | 将业务参数的位置、缺失默认值和翻译阶段绑定在一起 |
| `Config[T]` | name、default_value、comment=None | 控制 Encoder / Decoder / Transport 的代码行为或 SDK 连接 |
| `Omit` / `omit` | 显式省略标记 | 不依赖任何 SDK 的 Omit，复制保持标记身份 |

`name` 与 `comment` 只用于前端展示。业务由 Profile 的字段标识决定，例如 `max_tokens`，
不会根据展示名称、注释或路径末段猜测参数含义。两种 wrapper 的 default_value 都支持 None 和 omit。

`extra_body` / `extra_headers` 保留 Config，控制请求字段组的组装与合并：extra_body 在 Anthropic / Ollama 对应根字段组，Gemini 对应 config 组，Chat / Responses 对应 SDK overlay 组；headers 还参与 Ollama 客户端配置。
单个需要指定 path / phase 翻译的业务参数使用 Parameter。

| phase | 翻译方向 | 例子 |
| --- | --- | --- |
| `encode` | 从接口返回的指定位置读取值，编码到中间态 | content、reasoning_content |
| `decode` | 将中间态或生成选项翻译为接口请求，在指定位置应用参数 | max_tokens、reasoning_enable、reasoning_effort |

encode 参数不会注入请求；decode 参数不会被用来读取响应。
`Parameter` 声明字段位置，不会自动实现新的消息语义。自定义 Encoder 通过基类的
`parameter_value()` 读取声明参数，再将它映射到明确支持的中间态语义。

## 继承、默认值和省略

Profile 的参数与行为字段都使用 `Parameter[T] | Omit | None` 或 `Config[T] | Omit | None`：

| 配置 | 含义 |
| --- | --- |
| 字段未传或为 None | 继承上一层完整声明 |
| 字段为 omit | 禁用该声明；参数不再自动读取或写入，手动提供的原生请求字段保留 |
| Parameter(default_value=None, ...) | 仅在目标字段缺失时使用实际值 None |
| Parameter(default_value=omit, ...) | 字段缺失时不补值；已经存在的值仍保留 |
| Config(default_value=None, ...) | 将行为配置值设为实际的 None |

**目标字段存在时，即使为 None，也保留原值。** default_value 只在字段缺失时使用。
显式 generate 控制参数和由行为配置生成的控制值优先于参数默认值。
`generate(max_tokens=omit)` 删除该请求的 token 目标键；这与禁用 Profile 的参数声明不同。

Parameter 整体覆盖，路径、阶段和默认值作为一份声明更新，不与上一层分别拼接。
Config 字典按键合并，嵌套值整体替换，字典中的 omit 删除该键。
Profiles 是 frozen dataclass；使用 `dataclasses.replace()` 创建新配置，再创建 MyLLM。
MyLLM 构造时隔离配置的嵌套可变数据；请求只在需要独立所有权的边界复制数据。

```python
from my_llm import Config, ModelProfile, Parameter, ProviderProfile, omit, resolve_profiles

provider = ProviderProfile(
    extra_body=Config(name="附加请求体", default_value={"a": 1, "b": 2}),
    max_tokens=Parameter(
        name="输出上限",
        path="$.max_completion_tokens",
        default_value=4096,
        phase="decode",
    ),
)
model = ModelProfile(
    model="demo",
    extra_body=Config(name="模型请求体", default_value={"a": 3, "b": omit}),
    max_tokens=None,  # 继承 provider 的整份 Parameter
)
resolved = resolve_profiles(provider, model)
assert resolved.get("extra_body") == {"a": 3}
assert resolved.get("max_tokens") == 4096
assert resolved.parameter("max_tokens", phase="decode").path == "$.max_completion_tokens"
```

`ResolvedProfile.get(field)` 返回声明的 default_value，保留实际 None；缺失或 omit 使用调用时的默认值。
`parameter(field, phase=...)` 返回对应 Parameter，字段被禁用或阶段不符时返回 None。
不再维护与值分离的 parameter_paths，也不提供 ParameterPath / ParameterDefaultV 别名。

## 统一参数路径

路径维持 `$.` 前缀、点分键和非负数值索引的语义。
`$` 表示 `**parameters` 解包字典的根目录；例如 `$.reasoning.effort` 指根参数 reasoning 内的 effort，
`$.extra_body.vendor_options.enabled` 指 SDK 正式 extra_body 入口内的扩展字段。
encode 参数从接口返回对象的根部按同样的路径语法读取。
路径不接受省略 `$.` 的写法，也不使用事件类型字符串充当路径。
事件类型的识别由协议 Encoder / Decoder 负责。

| 协议 | max_tokens（decode） | content（encode） | reasoning_content（encode） |
| --- | --- | --- | --- |
| openai_chat | `$.max_completion_tokens` | `$.choices[0].delta.content` | `$.choices[0].delta.reasoning_content` |
| openai_responses | `$.max_output_tokens` | `$.delta` | `$.delta` |
| anthropic | `$.max_tokens` | `$.delta.text` | `$.delta.thinking` |
| gemini | `$.config.max_output_tokens` | `$.candidates[0].content.parts[0].text` | `$.candidates[0].content.parts[0].text` |
| ollama | `$.options.num_predict` | `$.message.content` | `$.message.thinking` |

Chat 的 reasoning_content 是兼容服务扩展；Responses、Gemini 仍按标准事件或 part 类型区分文本和思考。
修改路径不会改变协议事件类型的含义，也不会更改 Decoder 输出的标准字段位置。

## Provider 与 Model 字段

| 字段 | 归属 / 说明 |
| --- | --- |
| `protocol` | Provider；默认 openai_chat，接受已注册协议名 |
| `base_url` / `api_key` | Provider Config；继承 SDK 默认值，api_key 不出现在 repr |
| `timeout` / `max_retries` | Provider Config；正秒数 / 非负整数，默认使用 SDK 配置 |
| `supports_stream` | Provider Config；默认 True，False/omit 不支持，因为两种返回模式均使用上游流 |
| `usage_mode` | Provider Config；默认 none；可选值由协议 Transport.usage_modes 声明 |
| `models` | Provider；可选 ModelProfile 列表，生成仍须显式选择模型 |
| `model` | Model；必填非空模型 ID |
| `max_context_window` | Model；默认 256000，仅描述性整数元数据，不截断输入或限制输出 |
| `supports_reasoning` | Model Config；默认 True |
| `supports_function_calling` | Model Config；默认 False，发送非空 tools 前开启 |
| `supports_message_name` | Model Config；默认 True，False/omit 在 Chat 请求移除 name |

`ModelProfile.supported_roles` 是固定类属性。
Provider 的 `add_models(ModelProfile | list[ModelProfile])` 按 ID 更新并保留顺序；
`list_models()` 返回列表副本，`delete_model(list[str])` 忽略不存在的 ID。
模型字符串不会触发 catalogue 查找。

`stdio_protocol` 属于 MyLLM，决定输入输出协议。客户端由 MyLLM 延迟创建、复用和关闭，响应流按请求关闭。
Ollama SDK 没有原生 max_retries 选项，仅接受默认值 None / omit / 0，正数在构造时拒绝。

## Provider / Model 共享覆盖项

两者继承公开的 `OverridableProfile`，共享如下声明：

| 字段 | 类型 / 说明 |
| --- | --- |
| `extra_headers` / `extra_body` | Config；请求头及协议原生生成参数，按键合并 |
| `max_tokens` | decode Parameter；正整数输出上限，标准缺失默认值 omit |
| `content` | encode Parameter；文本位置及缺失时默认值，标准缺失默认值 omit |
| `reasoning_content` | encode Parameter；思考内容位置及缺失时默认值，标准缺失默认值 omit |
| `reasoning_enable` | decode Parameter；开启思考的参数位置及缺失时默认值 |
| `reasoning_effort` | decode Parameter；强度位置及缺失时默认值 |
| `supports_reasoning_effort` | Config；Responses 默认 True，其他默认 False |
| `reasoning_effort_mapping` | Config；none/low/medium/high/xhigh/max 的映射，映射项 None/omit 禁用对应级别 |
| `replay_reasoning_content_when_tool_calls` | Config；Chat 默认 False，其他默认 True；控制可见思考回放 |

generate 的显式 max_tokens 覆盖 Profile 默认，omit 删除该请求目标键。
max_tokens 声明的 default_value=omit 时不制造 token 参数；Anthropic 仍要求提供有效 max_tokens。
路径不能覆盖 model、messages、tools 等库管理结构。

开启 reasoning 时才应用 reasoning_enable；支持强度时，映射结果写入 reasoning_effort 的路径，覆盖该参数的默认值。
各协议默认值由 Codec 声明：Chat 使用 medium，Responses 使用 effort=medium、summary=detailed；
Anthropic 使用 enabled 与 budget_tokens=1024，Gemini 使用 include_thoughts=True、thinking_budget=-1，Ollama 使用 True。
Anthropic 的预算必须小于 max_tokens。模型支持的强度由具体协议和配置决定。

## 兼容配置迁移示例

原先分开的路径与默认值改为同一 Parameter；行为开关改用 Config 包装。
下面的兼容 Chat 服务将启用思考的参数发往 extra_body.thinking，
从响应 delta.vendor_reasoning 读取思考内容，并将输出上限发往旧式 max_tokens：

```python
from my_llm import Config, Parameter, ProviderProfile, omit

compatible = ProviderProfile(
    protocol="openai_chat",
    replay_reasoning_content_when_tool_calls=Config(name="工具回合回放思考", default_value=True),
    reasoning_enable=Parameter(
        name="启用思考",
        path="$.extra_body.thinking",
        default_value={"type": "enabled"},
        phase="decode",
    ),
    reasoning_content=Parameter(
        name="思考内容",
        path="$.choices[0].delta.vendor_reasoning",
        default_value=omit,
        phase="encode",
    ),
    max_tokens=Parameter(
        name="输出上限",
        path="$.max_tokens",
        default_value=4096,
        phase="decode",
    ),
)
```

展示名称可以调整，参数行为仍由 reasoning_enable、reasoning_content 和 max_tokens 字段标识决定。
覆盖路径时显式提供该参数的 default_value 和 phase，避免位置与默认值从不同层隐式拼接。

## 继承扩展 Profile

ProviderProfile、ModelProfile、ProtocolDefaults 均支持 dataclass 继承扩展。
新业务参数直接声明 `Parameter[T] | Omit | None`，新分支控制声明 `Config[T] | Omit | None`。
解析器自动发现继承字段并检查类型、路径和阶段，不需要路径字段配对或中央参数清单。

```python
from dataclasses import dataclass
from my_llm import Config, Omit, Parameter, ProviderProfile


@dataclass(frozen=True, kw_only=True)
class CompatibleProvider(ProviderProfile):
    vendor_budget: Parameter[int] | Omit | None = Parameter(
        name="供应商预算",
        path="$.extra_body.vendor_budget",
        default_value=256,
        phase="decode",
        comment="仅在请求未提供该字段时使用",
    )
    vendor_mode: Config[bool] | Omit | None = Config(name="供应商模式", default_value=True)


provider = CompatibleProvider()  # 使用扩展类声明的预算配置
```

BaseDecoder 的统一请求流程应用所有 decode 参数，以上 vendor_budget 无需在协议中另写默认值注入。
encode 参数由 BaseEncoder.parameter_value() 统一读取，再由协议钩子完成对应消息语义。
Config 的含义由需要它的 Encoder / Decoder / Transport 实现，不会自动产生业务功能。
Model 不必重复声明 Provider 的扩展字段即可继承其解析结果；ProtocolDefaults 子类的新字段也自动加入解析。
独立自定义协议沿用同一流程，见[接入指南](../guides/adding-protocol.md)。

## SDK 参数与协议字段

所有原生请求在翻译后保持字典，经 Transport 统一过滤，再通过 `**parameters` 传给官方 SDK。
`$` 根参数必须符合 SDK 方法签名；兼容字段使用 SDK 已提供的 extra_body 等正式扩展入口，
声明一个路径不会为 SDK 增加新的调用参数。
路径声明也不会新增原生 Codec / Transport 的结构变体。例如将 Anthropic 的 thinking.type 改为 vendor，
仍会受到本项目原生 Transport 的标准结构校验；需要扩展对应 Codec / Transport 实现该变体。
这属于本项目的协议语义约束，SDK 的类型注解本身不代表运行时必然拒绝该值。
SDK 正式 extra_body 等开放字典可以承载自定义数据。
Profile.extra_body 在 Chat / Responses 中对应 SDK extra_body；Anthropic / Ollama 使用原生请求参数，
Gemini 使用 config 参数。SDK 不提供的扩展入口不能通过虚构根参数补出。

Transport 按声明的消息、工具路径及官方 SDK 类型处理结构：

| 协议 | 消息路径 | 工具路径 |
| --- | --- | --- |
| Chat | messages | tools |
| Responses | input | tools |
| Anthropic | messages | tools |
| Gemini | contents | config.tools |
| Ollama | messages | tools |

结构化协议字段按 SDK 类型过滤；已识别的 dict / Any 开放字段保留内部应用键值，
例如工具参数、结果对象、JSON Schema 及 extra_body。
SDK 的 extra="allow" 不会自动赋予翻译层未知字段保留能力。
Schema 缓存保存类型规则，不保存 Profile、请求值或输入字段样本，详见[缓存设计](../design/schema-filter-cache.md)。

## 跨协议请求的处理边界

Encoder / Decoder 只读取并构造标准协议及 Profile 明确声明的已支持字段。
标准协议不存在、配置未声明的字段直接跳过，不读取、不校验，也不逐项遍历检测未知字段。
开放字典内部应用数据随其已识别的外层字段保留，不保存原始对象或跨协议恢复记录。
跨协议来回翻译不保证无损，详见[翻译字段边界](../design/translation-boundaries.md)。

标准签名、redacted_thinking、encrypted_content 按对应协议明确处理，不能转换成其他协议的原生签名，
也不通过隐藏记录保存供后续恢复。不能表达的媒体或会改变工具约束的已知 SDK 限制会在请求前报错。
Chat / Ollama 无法原生读取 URL 文档时保留文本引用，不隐式下载文件。
