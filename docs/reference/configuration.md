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
        timeout=Config(default_value=30),
        max_retries=Config(default_value=2),
        usage_mode=Config(default_value="native"),
    ),
    model=ModelProfile(
        model="your-model",
        max_tokens=Parameter(
            path="$.max_output_tokens",
            default_value=2048,
        ),
    ),
    stdio_protocol="openai_chat",
)
```

## 参数与行为配置

以下类型均可从 `my_llm` 导入，`Parameter` 与 `Config` 都是 frozen、kw_only dataclass：

| 类型 | 字段 | 职责 |
| --- | --- | --- |
| `Parameter[T]` | path、default_value=omit、supports=True、comment=None | 绑定参数位置、默认值及支持状态；name/source 由 Profile 固定 |
| `Config[T]` | default_value、comment=None；name 由 Profile 固定 | 控制 Encoder / Decoder / Transport 的代码行为或 SDK 连接 |
| `Omit` / `omit` | 显式省略标记 | 不依赖任何 SDK 的 Omit，复制保持标记身份 |

`Parameter.name`、`Parameter.source` 和 `Config.name` 不接受构造参数，也不能二次赋值。
Profile 的 `@set_attr` 在初始化时检查并绑定字段名；Parameter 的 source 在 Profile 字段 metadata 中硬编码。
各层的同名参数必须保持相同 source/scope；解析时也会检查身份是否已正确绑定。
已有同一身份的 wrapper 可复用，不会重新赋值；不能把已绑定 wrapper 换到另一个字段。
`comment` 可用于展示说明。两种 wrapper 的 default_value 都支持 None 和 omit。
`Parameter(supports=False, ...)` 会将 default_value 强制设为 omit，并停止自动读取或写入（包括块起始和最终快照中的可见字段）；
原生请求中手动提供的字段仍保留，显式请求不受支持的 reasoning 时会报错。

`extra_body` / `extra_headers` 保留 Config，控制请求字段组的组装与合并：extra_body 在 Anthropic / Ollama 对应根字段组，Gemini 对应 config 组，Chat / Responses 对应 SDK overlay 组；headers 还参与 Ollama 客户端配置。
单个需要指定 path 路由的业务参数使用 Parameter。

| source | 参数来源 | 例子 |
| --- | --- | --- |
| `output` | 从接口返回的指定位置读取值，编码到中间态 | content、reasoning_content |
| `input` | 将中间态或生成选项翻译为接口请求，在指定位置应用参数 | max_tokens、reasoning_enable、reasoning_effort |

output 参数不会注入请求；input 参数不会被用来读取响应。
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

**普通字段的 default_value 仅补缺，目标字段存在时（包括 None）保留原值。**
显式 token 控制、reasoning_enable 的配置生成值、reasoning_effort 的映射结果覆盖同路径的原生字段，生成控制值保持最高优先级。
显式 generate 控制参数和由行为配置生成的控制值优先于参数默认值。
`generate(max_tokens=omit)` 删除该请求的 token 目标键；这与禁用 Profile 的参数声明不同。

Parameter 整体覆盖，路径、支持状态和默认值作为一份声明更新，不与上一层分别拼接。
Config 字典按键合并，嵌套值整体替换，字典中的 omit 删除该键。
Profiles 是 frozen dataclass；使用 `dataclasses.replace()` 创建新配置，再创建 MyLLM。
MyLLM 构造时隔离配置的嵌套可变数据；请求只在需要独立所有权的边界复制数据。

```python
from my_llm import Config, ModelProfile, Parameter, ProviderProfile, omit, resolve_profiles

provider = ProviderProfile(
    protocol="openai_chat",
    extra_body=Config(default_value={"a": 1, "b": 2}),
    max_tokens=Parameter(
        path="$.max_completion_tokens",
        default_value=4096,
    ),
)
model = ModelProfile(
    model="demo",
    extra_body=Config(default_value={"a": 3, "b": omit}),
    max_tokens=None,  # 继承 provider 的整份 Parameter
)
resolved = resolve_profiles(provider, model)
assert resolved.get("extra_body") == {"a": 3}
assert resolved.get("max_tokens") == 4096
assert resolved.parameter("max_tokens").path == "$.max_completion_tokens"
```

`ResolvedProfile.get(field)` 返回声明的 default_value，保留实际 None；缺失或 omit 使用调用时的默认值。
`parameter(field, source=...)` 返回对应 Parameter，字段被禁用、supports=False 或来源不符时返回 None。
不再维护与值分离的 parameter_paths，也不提供 ParameterPath / ParameterDefaultV 别名。

## 统一参数路径

路径维持 `$.` 前缀、点分键和非负数值索引的语义。
生成参数的 `$` 表示 SDK 方法的 `**parameters` 解包字典根目录；例如 `$.reasoning.effort` 指根参数 reasoning 内的 effort，
`$.extra_body.vendor_options.enabled` 指 SDK 正式 extra_body 入口内的扩展字段。
output 参数从接口返回对象的根部按同样的路径语法读取。
键名支持连字符，例如 `$.extra_headers.x-api-key`。路径不接受省略 `$.` 的写法，也不使用事件类型字符串充当路径。
事件类型的识别由协议 Encoder / Decoder 负责。

| 协议 | max_tokens（input） | content（output） | reasoning_content（output） |
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
| `protocol` | Provider；默认 None，使用前须显式指定已注册协议名 |
| `base_url` | Provider Config；默认继承 SDK 配置 |
| `api_key` | Provider input Parameter；配置连接密钥的传递位置，default_value 默认为 omit，不出现在 repr |
| `timeout` / `max_retries` | Provider Config；正秒数 / 非负整数，默认使用 SDK 配置 |
| `supports_stream` | Provider Config；默认 True，False/omit 不支持，因为两种返回模式均使用上游流 |
| `usage_mode` | Provider Config；默认 none；可选值由协议 Decoder.usage_modes 声明 |
| `models` | Provider；可选 ModelProfile 列表，生成仍须显式选择模型 |
| `model` | Model；必填非空模型 ID |
| `max_context_window` | Model；默认 256000，仅描述性整数元数据，不截断输入或限制输出 |
| `reasoning_enable.supports` | Parameter；默认 True，替代 supports_reasoning |
| `supports_function_calling` | Model Config；默认 False，发送非空 tools 前开启 |
| `supports_message_name` | Model Config；默认 True，False/omit 在 Chat 请求移除 name |

`ModelProfile.supported_roles` 是固定类属性。
Provider 的 `add_models(ModelProfile | list[ModelProfile])` 按 ID 更新并保留顺序；
`list_models()` 返回列表副本，`delete_model(list[str])` 忽略不存在的 ID。
模型字符串不会触发 catalogue 查找。

`stdio_protocol` 属于 MyLLM，决定输入输出协议。客户端由 MyLLM 延迟创建、复用和关闭，响应流按请求关闭。
Ollama SDK 没有原生 max_retries 选项，仅接受默认值 None / omit / 0，正数在构造时拒绝。

`api_key` 在创建 SDK 客户端时应用，其路径指向连接参数组：`$.api_key` 使用 SDK 密钥参数，
`$.extra_headers.<头名>` 使用指定认证头。Decoder 将请求头映射到 SDK 的 default_headers、
Gemini 的 HttpOptions.headers 或 Ollama 的 headers；密钥不会作为生成参数写入请求体。
配置值是完整字符串，Authorization 的 Bearer 前缀也由配置显式提供：

```python
from my_llm import Parameter, ProviderProfile

sdk_key = ProviderProfile(
    protocol="anthropic",
    api_key=Parameter(
        path="$.api_key",
        default_value="your-api-key",
    ),
)
header_key = ProviderProfile(
    protocol="anthropic",
    api_key=Parameter(
        path="$.extra_headers.Authorization",
        default_value="Bearer your-api-key",
    ),
)
```

标准声明的 default_value 为 omit；Provider 的 None 继承协议声明，omit 禁用声明。
Ollama 默认路径为 `$.extra_headers.Authorization`，其他协议为 `$.api_key`。
Ollama SDK 没有 api_key 参数，须使用 header 路径。显式 extra_headers 中的同名头优先，
generate 的 extra_headers 继续按现有规则覆盖；Ollama 的请求头仍参与客户端缓存键。
省略密钥不会改变 SDK 自身的环境变量读取与认证要求。
例如 Anthropic SDK 的 header 认证检查使用 `Authorization` 或 `X-Api-Key`，应按该大小写配置。
原先的 `api_key=Config(...)` 需改为 `Parameter(...)`，并显式提供 path。

## Provider / Model 共享覆盖项

两者继承公开的 `OverridableProfile`，共享如下声明：

| 字段 | 类型 / 说明 |
| --- | --- |
| `extra_headers` / `extra_body` | Config；请求头及协议原生生成参数，按键合并 |
| `max_tokens` | input Parameter；正整数输出上限，标准缺失默认值 omit |
| `content` | output Parameter；文本位置及缺失时默认值，标准缺失默认值 omit |
| `reasoning_content` | output Parameter；思考内容位置及缺失时默认值，标准缺失默认值 omit |
| `reasoning_enable` | input Parameter；开启思考的参数位置及缺失时默认值 |
| `reasoning_effort` | input Parameter；强度位置及缺失时默认值 |
| `reasoning_effort.supports` | Parameter；Responses 默认 True，其他默认 False，替代 supports_reasoning_effort |
| `reasoning_effort_mapping` | Config；none/low/medium/high/xhigh/max 的映射，映射项 None/omit 禁用对应级别 |
| `replay_reasoning_content_when_tool_calls` | Config；所有内置协议默认 True；控制可见思考回放 |

generate 的显式 max_tokens 覆盖 Profile 默认，omit 删除该请求目标键。
max_tokens 声明的 default_value=omit 时不制造 token 参数；Anthropic 仍要求提供有效 max_tokens。
路径不能覆盖 model、messages、tools 等库管理结构。

reasoning_enable=omit 只关闭该自动路由，仍可独立使用 reasoning_effort；
reasoning_enable.supports=False 才表示模型不支持 reasoning。
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
    replay_reasoning_content_when_tool_calls=Config(default_value=True),
    reasoning_enable=Parameter(
        path="$.extra_body.thinking",
        default_value={"type": "enabled"},
    ),
    reasoning_content=Parameter(
        path="$.choices[0].delta.vendor_reasoning",
        default_value=omit,
    ),
    max_tokens=Parameter(
        path="$.max_tokens",
        default_value=4096,
    ),
)
```

name/source 固定，展示说明通过 comment 提供。reasoning_enable 与 reasoning_effort 的支持状态分别在对应 Parameter.supports 中声明。
覆盖路径时显式提供该参数的 default_value 和 supports，避免位置与默认值从不同层隐式拼接。

## 继承扩展 Profile

ProviderProfile、ModelProfile、ProtocolDefaults 均支持 dataclass 继承扩展。
新增 Parameter 字段须在 metadata 中固定 source；若重写 __post_init__，须调用 super().__post_init__() 完成身份绑定。
新业务参数直接声明 `Parameter[T] | Omit | None`，新分支控制声明 `Config[T] | Omit | None`。
解析器自动发现继承字段并检查类型、路径和 source，不需要路径字段配对或中央参数清单。

```python
from dataclasses import dataclass, field
from my_llm import Config, Omit, Parameter, ProviderProfile


@dataclass(frozen=True, kw_only=True)
class CompatibleProvider(ProviderProfile):
    vendor_budget: Parameter[int] | Omit | None = field(
        default=Parameter(
            path="$.extra_body.vendor_budget",
            default_value=256,
            comment="仅在请求未提供该字段时使用",
        ),
        metadata={"source": "input"},
    )
    vendor_mode: Config[bool] | Omit | None = Config(default_value=True)


provider = CompatibleProvider()  # 使用扩展类声明的预算配置
```

BaseDecoder 的统一请求流程应用所有 input 参数，以上 vendor_budget 无需在协议中另写默认值注入。
output 参数由 BaseEncoder.parameter_value() 统一读取，再由协议钩子完成对应消息语义。
Config 的含义由需要它的 Encoder / Decoder / Transport 实现，不会自动产生业务功能。
Model 不必重复声明 Provider 的扩展字段即可继承其解析结果；ProtocolDefaults 子类的新字段也自动加入解析。

连接 Parameter 通过 dataclass 字段 metadata 声明 scope="client"；内置 `api_key` 已设置。
其余 input Parameter 默认属于 request，解析器一次完成分类，不根据字段名称在过滤器中排除认证参数。
例如给认证字段使用不同名称：

```python
from dataclasses import dataclass, field
from my_llm import Parameter, ProviderProfile


@dataclass(frozen=True, kw_only=True)
class HeaderProvider(ProviderProfile):
    authentication: Parameter[str] = field(
        default=Parameter(path="$.extra_headers.x-auth", default_value="offline-token"),
        metadata={"scope": "client", "source": "input"},
    )
```

该声明进入客户端认证参数，不进入 SDK 请求参数；连接路径沿用 `$.api_key` 或 `$.extra_headers.<头名>`。
连接配置及 SDK 映射在 MyLLM 构造时准备，请求 body、headers、生成控制在每次调用时准备。
`ResolvedProfile.request_parameters` 保存请求声明，`connection` 保存已验证连接快照；均按实例只读使用。

独立自定义协议沿用同一流程，见[接入指南](../guides/adding-protocol.md)。

## SDK 参数与协议字段

所有原生请求在翻译后保持字典，经 Decoder 统一过滤，再通过 `**parameters` 传给官方 SDK。
`$` 根参数必须符合 SDK 方法签名；兼容字段使用 SDK 已提供的 extra_body 等正式扩展入口，
声明一个路径不会为 SDK 增加新的调用参数。
路径声明也不会新增原生 Codec / Decoder 的结构变体。例如将 Anthropic 的 thinking.type 改为 vendor，
仍会受到本项目原生 Decoder 的标准结构校验；需要扩展对应 Codec / Decoder 实现该变体。
这属于本项目的协议语义约束，SDK 的类型注解本身不代表运行时必然拒绝该值。
SDK 正式 extra_body 等开放字典可以承载自定义数据。
Profile.extra_body 在 Chat / Responses 中对应 SDK extra_body；Anthropic / Ollama 使用原生请求参数，
Gemini 使用 config 参数。SDK 不提供的扩展入口不能通过虚构根参数补出。

Decoder 按声明的消息、工具路径及官方 SDK 类型处理结构：

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

支持状态与值一起覆盖，例如禁用 reasoning：

```python
from my_llm import ModelProfile, Parameter

model = ModelProfile(
    model="demo",
    reasoning_enable=Parameter(path="$.reasoning_effort", supports=False),
)
assert model.reasoning_enable.name == "reasoning_enable"
assert model.reasoning_enable.source == "input"
```

`supports_stream`、`supports_function_calling`、`supports_message_name` 控制流协议、工具能力和消息处理分支，
当前没有对应的可路由 Parameter，继续使用独立 Config。`replay_reasoning_content_when_tool_calls`、
`reasoning_effort_mapping` 等行为配置也保持 Config。
