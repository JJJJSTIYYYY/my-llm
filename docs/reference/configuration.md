# 配置指南

`ProviderProfile` 描述供应商连接和 API 默认值，`ModelProfile` 描述模型及能力差异。
两者都使用关键字参数构造，不内置供应商或模型清单。
调用方式、消息格式与结果说明见 [API 使用指南](api.md)。

## 基本配置

```python
import os

from my_llm import ModelProfile, MyLLM, ProviderProfile


def make_llm():
    provider = ProviderProfile(
        api_type="responses",
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=60.0,
        max_retries=2,
        usage_mode="native",
    )
    model = ModelProfile(
        model=os.environ["LLM_MODEL"],
        max_context_window=256000,
        max_tokens=4096,
        supports_function_calling=True,
    )
    return MyLLM(provider=provider, model=model)
```

OpenAI 两套 API 均通过 OpenAI SDK 调用；兼容服务通过 `base_url` 和下文的参数、能力
配置适配。`api_type` 选择请求 API，不限制 `messages` 或 `tools` 的输入格式。
配置声明能力不代表服务端实际支持该能力，请按使用的模型填写。

## 默认值与覆盖顺序

通常按 **协议默认值 → Provider → Model → 本次 generate 参数** 覆盖。
只有接口实际提供的调用参数可以按次覆盖；例如 `usage_mode`、各能力开关和解析
路径只能在 Profile 上设置。`api_type`、`protocol`、连接配置和模型列表归 Provider，
`model`、`max_tokens`、`max_context_window` 和模型专属能力归 Model。

多数可继承字段在 Profile 上的声明默认值为 `None`。以下表格单独给出最终生效值。

- `None`：继承上一层。在调用参数中表示继承已配置值。
- `omit`：从 `my_llm` 导入，禁用整个字段；用于连接字段时，不向 SDK 显式设置该值。
- 字典：按键浅覆盖；相同键的嵌套字典整体替换，不做递归合并。
- 字典中的 `omit`：删除继承的键。字段整体为 `omit` 时清空继承字典。

```python
from my_llm import ModelProfile, ProviderProfile, omit

provider = ProviderProfile(extra_body={"temperature": 0.5, "vendor_flag": True})
model = ModelProfile(
    model="your-model-id",
    extra_body={"vendor_flag": omit},
)
# 使用这两个 Profile 时，extra_body 只保留 temperature。
# generate(..., extra_body={"temperature": 0.2}) 可再覆盖本次的值。
```

Profile 字段不可重新赋值。用 `dataclasses.replace()` 创建新配置，再创建新的
`MyLLM` 实例；已构造实例的配置不会跟着原 Profile 的后续变化自动更新。
模型列表有专用管理方法，见下文。

```python
from dataclasses import replace

from my_llm import ModelProfile, MyLLM, ProviderProfile

provider = ProviderProfile()
model = ModelProfile(model="your-model-id", max_tokens=2048)
updated_model = replace(model, max_tokens=4096)
llm = MyLLM(provider=provider, model=updated_model)
```

## Provider 专属字段

| 字段 | 声明默认值 | 配置值与作用 |
| --- | --- | --- |
| `api_type` | `"chat_completions"` | `"chat_completions"` / `"responses"`，选择 OpenAI API |
| `protocol` | `"openai"` | `openai` / `anthropic` / `gemini` / `ollama`；各自使用官方 SDK |
| `base_url` | `None` | 非空字符串服务地址；未配置时使用 SDK 默认地址 |
| `api_key` | `None` | 非空字符串凭据；未配置时使用 SDK 环境配置；Profile 的 repr 不显示此字段 |
| `timeout` | `None` | 正数秒数，未配置时使用 SDK 默认超时 |
| `max_retries` | `None` | 非负整数；`0` 关闭 SDK 重试，未配置时使用 SDK 默认重试策略 |
| `supports_stream` | `None` | `bool`/`omit`；最终默认 `True`，`False`/`omit` 会报错 |
| `usage_mode` | `None` | 最终默认 `"none"`，其他模式见用量配置 |
| `models` | `None` | `list[ModelProfile]` 或 `None`，用于管理该 Provider 的模型配置 |

以上四个连接字段支持 `omit`；它们也可在 `generate()` 按次覆盖。
若注入 `client=AsyncOpenAI(...)`，连接由注入的 Client 决定，Profile 的连接字段
不用于构造 Client，调用时也不能再提供连接覆盖（包括 `omit`）。
即使 `stream=False`，供应商仍需支持流式响应；这由 `supports_stream` 表达。

## Model 专属字段

| 字段 | 声明默认值 | 配置值与作用 |
| --- | --- | --- |
| `model` | 必填 | 非空字符串，发送给供应商的模型 ID |
| `max_context_window` | `256000` | 任意 `int`，只作为模型描述信息 |
| `max_tokens` | `None` | 正整数输出 Token 限制；`None` 不显式限制，`omit` 禁用显式限制 |
| `supports_reasoning` | `None` | 最终默认 `True`；`False`/`omit` 时请求推理会报错 |
| `supports_function_calling` | `None` | 最终默认 `False`；传非空 tools 前需设为 `True` |
| `supports_message_name` | `None` | 最终默认 `True`；`False`/`omit` 时不发送消息 name |

`max_context_window` 允许任意整数，不接受 `bool` 或非整数。
它不限制历史长度，不计算 Token，不截断消息，也不作为请求参数发送。
`max_tokens` 是独立的输出限制，Responses 使用 `max_output_tokens`，
Chat 使用 `max_completion_tokens`；调用时传 `None` 继承 Model、传 `omit` 不设置限制。
未显式限制时仍受供应商自身规则约束。

`supported_roles` 是固定的类属性 `("system", "user", "assistant", "tool")`，
不接受构造参数配置。

## Provider 与 Model 共有字段

下列字段声明默认值均为 `None`，Model 可以覆盖 Provider 的设置。
字典值及路径支持 `omit`；布尔字段的 `omit` 最终按禁用处理。

| 字段 | 生效默认值与作用 |
| --- | --- |
| `extra_headers` | `{}`，额外请求头 |
| `extra_body` | `{}`，额外 JSON 请求字段 |
| `reasoning_enable_parameter` | 按 API 提供推理开启参数，见推理配置 |
| `reasoning_effort_parameter` | 按 API 提供强度参数，见推理配置 |
| `reasoning_effort_mapping` | 六种强度映射为同名字符串；`None`/`omit` 键值禁用该强度 |
| `supports_reasoning_effort` | Responses 为 `True`，Chat 为 `False` |
| `content_path` | 标准文本增量位置，见输出字段配置 |
| `reasoning_content_path` | 标准推理增量位置，见输出字段配置 |
| `replay_reasoning_content_when_tool_calls` | `True`，每条 Chat/Ollama assistant 消息回传可见推理，缺失时补空字符串；`False`/`omit` 不回传 |

### 额外请求字段

`extra_headers` 和 `extra_body` 支持 Provider、Model、本次调用三个层级的按键覆盖。
例如供应商接受额外参数时，可配置 `extra_body={"temperature": 0.2}`。
字段是否被供应商接受由其 API 决定。

`extra_body` 不能设置 `model`、`stream`、`messages`、`input`、`tools` 或
`previous_response_id`，这些应通过正式接口配置。
自动生成的推理参数优先于同名额外字段；已生成的输出 Token 限制及用量开关也
优先于 `extra_body` 中的同名值。推理配置中的 extra_body 遵循相同的保留字段限制。

## 推理配置

调用参数 `reasoning_effort` 默认 `"none"`，不自动添加推理开启参数。
`reasoning=True` 且强度为 `none` 时使用 `medium`；
`reasoning=False` 与非 `none` 强度同时传入会报错。
非 `none` 强度要求模型 `supports_reasoning=True`。

强度可选 `none`、`low`、`medium`、`high`、`xhigh`、`max`，实际请求取决于配置：

- `supports_reasoning_effort=False`：只使用开启参数，不发送分级强度。
- `supports_reasoning_effort=True`：使用开启参数，再应用强度参数；强度通过
  `reasoning_effort_mapping` 映射。缺失、`None`、`omit` 或空字符串映射会报错。

默认推理配置如下；只有请求非 `none` 强度时才应用：

| 字段 | Chat Completions 默认值 | Responses 默认值 |
| --- | --- | --- |
| `reasoning_enable_parameter` | `{"extra_body": {"thinking": {"type": "enabled"}}}` | `{"reasoning": {"summary": "detailed", "effort": "medium"}}` |
| `reasoning_effort_parameter` | `{"reasoning_effort": "{effort}"}` | `{"reasoning": {"summary": "detailed", "effort": "{effort}"}}` |
| `supports_reasoning_effort` | `False`，默认不使用强度参数 | `True` |
| `reasoning_effort_mapping` | 六种强度均映射为同名字符串 | 六种强度均映射为同名字符串 |

这些是库的可配置默认值，应根据服务支持的参数调整。
参数字典表示 SDK 关键字参数，只允许顶层键 `extra_body`、`reasoning`（Responses）
或 `reasoning_effort`（Chat）。完整字符串 `"{effort}"` 会在嵌套字典/列表中替换为
映射后的强度值；它不是通用字符串插值模板。

例如，兼容服务使用自定义开关与预算字段：

```python
from my_llm import ModelProfile, ProviderProfile, omit

provider = ProviderProfile(
    api_type="chat_completions",
    base_url="https://your-provider.example/v1",
    supports_reasoning_effort=True,
    reasoning_enable_parameter={"extra_body": {"enable_thinking": True}},
    reasoning_effort_parameter={
        "reasoning_effort": omit,  # 删除继承的 Chat 强度参数
        "extra_body": {"thinking_budget": "{effort}"},
    },
    reasoning_effort_mapping={"low": "small", "medium": "normal", "high": "large"},
)
model = ModelProfile(
    model="your-model-id",
    reasoning_effort_mapping={"xhigh": None, "max": None},
)
# generate(..., reasoning_effort="high") 发送 enable_thinking=True、thinking_budget="large"。
# 请求 xhigh/max 会报 UnsupportedCapabilityError。
```

Responses 默认请求可见推理摘要 `summary="detailed"`。可在两个推理参数字典的
`reasoning` 中覆盖 `summary`，或使用 `summary: omit` 删除它：

```python
from my_llm import ProviderProfile, omit

provider = ProviderProfile(
    api_type="responses",
    reasoning_enable_parameter={"reasoning": {"summary": omit, "effort": "medium"}},
    reasoning_effort_parameter={"reasoning": {"summary": omit, "effort": "{effort}"}},
)
```

`replay_reasoning_content_when_tool_calls=True` 时，每条 Chat assistant 消息都会携带
`reasoning_content`，已有推理原样回传，缺失时补 `""`，没有工具调用的消息也一样。
Ollama 对应回传 `thinking` 字段。`False`/`omit` 时不回传这些字段。
system、user、tool 消息不补推理字段。
Responses 返回的原生 reasoning items 通过完整历史回放保留，包括加密字段；
该开关不用于解密、改写或删除原生 reasoning items。

## 用量配置

| `usage_mode` | API | 行为 |
| --- | --- | --- |
| `"none"` | 两者 | 默认；结果不暴露 usage |
| `"native"` | Responses | 从最终响应读取原生 usage 字典 |
| `"stream_options"` | Chat | 请求 include_usage，并读取独立的用量事件 |

Responses 配置 `stream_options` 或 Chat 配置 `native` 会报配置错误。
`omit` 按 `none` 处理。库不估算或统一计算用量；供应商没有返回用量时可为 `None`。

## 输出字段配置

标准服务一般无需更改这两个字段；兼容服务返回不同的文本字段时才需设置。

| 字段 | Chat Completions 生效默认值 | Responses 生效默认值 |
| --- | --- | --- |
| `content_path` | `$.choices[0].delta.content` | `response.output_text.delta` |
| `reasoning_content_path` | `$.choices[0].delta.reasoning_content` | `response.reasoning_summary_text.delta` |

Chat 路径支持点分键/属性和非负数字下标，例如 `$.choices[0].delta.thinking`。
不支持通配符、过滤器或完整 JSONPath。字段缺失视为空增量，匹配到非字符串时报错。
Responses 默认值是事件类型名称：匹配事件后读取其 `delta`；也可以使用以 `$`
开头的上述简单路径读取字段。`omit` 禁用对应文本提取。

```python
from my_llm import ProviderProfile, omit

provider = ProviderProfile(
    api_type="chat_completions",
    reasoning_content_path="$.choices[0].delta.thinking",
    content_path=omit,
)
```

这些字段仅影响可见文本的读取。工具描述、消息输入及工具调用返回不需要也不提供
独立模板配置，统一按选定 API 的格式自动识别、转换和解析。

## 模型列表

```python
from my_llm import ModelProfile, MyLLM, ProviderProfile

provider = ProviderProfile(models=[ModelProfile(model="first")])
provider.add_models(ModelProfile(model="second", max_context_window=128000))
provider.add_models([ModelProfile(model="third"), ModelProfile(model="second", max_tokens=1024)])
models = provider.list_models()
selected = next(model for model in models if model.model == "second")
llm = MyLLM(provider=provider, model=selected)
provider.delete_model(["first", "third"])
```

- `add_models(model_profile: list[ModelProfile] | ModelProfile) -> None`：追加模型，
  同 ID 后加入的配置替换旧配置，保留该 ID 的位置；初始化列表也遵循该规则。
- `list_models() -> list[ModelProfile]`：返回新的列表，复用 ModelProfile 对象；
  改动该列表的成员不会增删 Provider 的模型。未配置时返回 `[]`。
- `delete_model(model_id: list[str]) -> None`：按 `ModelProfile.model` 删除，
  忽略不存在的 ID。参数无效时抛 `ConfigurationError`，不作部分修改。

默认 `models=None`；空操作保持 `None`，已有列表删除全部元素后为 `[]`。
模型列表只用于管理配置，不改变调用行为、不自动选择模型，也不随请求发送。
模块级 `generate(model="second", ...)` 只创建默认 ModelProfile，不读取列表里的
`max_tokens` 等配置。需要使用列表中的配置时，如示例一样显式取出 ModelProfile。

## 配置校验

模型上下文元数据与模型列表在 Profile 构造或管理时校验；其他配置在创建
`MyLLM`、调用 `generate()` 或实际处理相关请求时校验。
无效值抛 `ConfigurationError` 或其子类，尚未实现的协议抛
`UnsupportedProviderError`，未启用的模型能力抛 `UnsupportedCapabilityError`。
具体异常和资源管理参见 [API 使用指南](api.md#错误处理)。


## 原生协议配置

Provider / Model 的字段、默认值、`None` 继承、`omit` 删除与按键覆盖规则保持一致。
`api_type`、消息 `name`、OpenAI 的增量路径及 `stream_options` 不适用于三种原生请求，
Codec 忽略这些请求字段，不修改 Profile。各 SDK 的缺省参数使用其原生省略机制：
Anthropic 使用 `anthropic.omit`；Gemini / Ollama 不接受 OpenAI 的 Omit 对象，因此不传对应参数。
原生 `extra_body` 只选取当前 SDK 支持的生成字段，不向 SDK 透传不兼容的参数。
`tools`、历史及模型等库管理字段不能通过 extras 或推理绑定覆盖。

| 配置 | Anthropic | Gemini | Ollama |
| --- | --- | --- | --- |
| SDK | `anthropic.AsyncAnthropic` | `google.genai.Client.aio` | `ollama.AsyncClient` |
| `max_tokens` | 必填，发送 `max_tokens` | `max_output_tokens` | `options.num_predict` |
| `timeout` | 秒 | 转换为毫秒 | 秒 |
| `max_retries` | SDK 原生重试 | SDK `attempts = max_retries + 1` | SDK 不提供该选项，忽略 |
| `extra_headers` | 请求头 | `HttpOptions.headers` | 自建 Client 时传入；注入 Client 时由调用者配置 |
| `reasoning=True` | 默认 thinking budget 1024，输出上限须大于 budget | 开启可见 thoughts，默认动态 budget | `think=True` |
| 支持强度时 | `output_config.effort`：low/medium/high/xhigh/max | `thinking_level`：minimal/low/medium/high | `think`：low/medium/high |
| 原生 extras | temperature/top_p/top_k/stop_sequences/thinking/output_config | temperature/top_p/top_k/stop_sequences/seed/thinking_config/response_mime_type/response_schema/response_json_schema/presence_penalty/frequency_penalty | format/keep_alive/think/options |

仅在模型确实支持强度时设置 `supports_reasoning_effort=True`。沿用原有
`reasoning_effort_mapping` 将统一强度映射到模型支持的值；未映射的 xhigh/max 等
会明确报错。模型真实能力由配置声明，供应商拒绝未知模型或未声明的差异时保留 SDK 异常。
`supports_reasoning=False/omit` 或 `supports_function_calling=False/omit` 仍拒绝相应请求。
通过现有推理绑定字段可设置原生推理参数，例如 Anthropic 的 budget 或 adaptive：

```python
from my_llm import ModelProfile, ProviderProfile

provider = ProviderProfile(protocol="anthropic", usage_mode="native")
model = ModelProfile(
    model="your-model-id", max_tokens=8192,
    reasoning_enable_parameter={
        "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 2048}},
    },
)
# 支持 adaptive 和 effort 的模型可改为 thinking={"type": "adaptive"}，
# 并显式设置 supports_reasoning_effort=True。
```

Gemini 强度模式会使用 `thinking_level`，省略互斥的 `thinking_budget`。
所有原生协议都只接收显式传入的本地函数定义，Gemini 自动函数执行始终关闭。
消息支持统一文本块；Anthropic / Gemini 还转换 image/file 的 URL 或 base64 data URI，
Ollama 接收 base64 data URI 图片。不下载远程媒体，也不跨厂商转换上传文件的 file_id；
无法表达的内容块直接报错。签名等原生元数据保存在现有 `response_output` 中，
多轮调用应回传完整返回消息。

原生用量默认关闭。`usage_mode="native"` 或 `"stream_options"` 都读取原生返回值，
后者不会向原生 SDK 发送 OpenAI 的 stream_options。

官方 SDK 参考：[Anthropic](https://github.com/anthropics/anthropic-sdk-python)、
[Google Gen AI](https://github.com/googleapis/python-genai)、
[Ollama](https://github.com/ollama/ollama-python)。
