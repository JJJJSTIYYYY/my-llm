# 配置指南

`ProviderProfile` 描述连接和上游 API；`ModelProfile` 描述模型与覆盖项。
MyLLM 构造时按协议默认值 → provider → model 合并，保存两份 profile 副本和一份
`resolved_profile: ResolvedProfile`。generate 不接受动态 provider/model 或连接参数。

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
| `protocol` | `openai_chat`；可选 openai_responses / anthropic / gemini / ollama |
| `base_url` | None，官方 SDK 默认地址 |
| `api_key` | None，使用 SDK 环境变量等默认行为；不出现在 profile repr |
| `timeout` | None，SDK 默认；显式值必须为正秒数 |
| `max_retries` | None，SDK 默认；显式值必须是非负整数 |
| `supports_stream` | 解析后 True；False/omit 不支持，因为两种返回模式均用上游流 |
| `usage_mode` | `none`；Responses/原生协议使用 `native`，Chat 使用 `stream_options` |
| `models` | None，可选 ModelProfile 列表；生成仍须显式选择模型 |

只有一个 `protocol` 字段，没有单独的 API 类型选择。
`stdio_protocol` 是 MyLLM 的输入输出协议，不属于 provider。
所有 SDK client 都由请求创建并关闭，不接受自定义 client。
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

| 字段 | 说明 |
| --- | --- |
| `extra_headers` | 请求 headers 按键合并 |
| `extra_body` | 协议原生生成参数按键合并 |
| `supports_reasoning_effort` | Responses 默认 True，其他默认 False |
| `reasoning_enable_parameter` | 开启思考时注入的 SDK 参数字典 |
| `reasoning_effort_parameter` | 支持强度时额外注入的 SDK 参数字典 |
| `reasoning_effort_mapping` | none/low/medium/high/xhigh/max 映射；None/omit 禁用某级 |
| `content_path` | OpenAI 文本增量路径或 Responses 事件类型 |
| `reasoning_content_path` | 兼容 Chat 服务的可见思考扩展路径或 Responses 事件类型 |
| `replay_reasoning_content_when_tool_calls` | Chat 默认 False，Ollama 默认 True；兼容 Chat 服务显式启用后回放可见思考，仅思考开启且当条有工具调用时补空值 |

Chat 默认文本路径 `$.choices[0].delta.content`，思考路径
`$.choices[0].delta.reasoning_content`；该字段属于兼容服务扩展，并非官方 OpenAI Chat 字段。Responses 默认事件为
`response.output_text.delta` / `response.reasoning_summary_text.delta`。
`$.` 路径支持属性/字典键与数值索引；omit 禁用路径。
原生协议使用原生字段（例如 Anthropic thinking、Gemini thought、Ollama thinking）。

replay=False/omit 在 Chat/Ollama 请求中去掉可见思考字段；True 保留已有内容。
官方 Chat 默认关闭该兼容字段回放，避免生成官方未定义的 reasoning_content；Ollama thinking 是原生字段，默认保留。
没有思考内容时，普通 assistant 消息保持不带思考字段；仅在思考开启且有工具调用时
分别补 `reasoning_content=""` / `thinking=""`。签名回放保持协议原生块。

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
SDK 会验证目标协议具体参数和字段；未知扩展在 Codec 转换中保留，不代表目标 API 一定支持。

## 跨协议请求的处理边界

外来字段不再因为同名就直接进入目标消息。目标 Decoder 仅投影已知可表达语义；
未知扩展仍留在中间态和同协议快照中。跨协议的未知事件不透传成其他协议的事件。

外来思考在 Chat / Responses / Anthropic / Gemini 中按普通文本表达，在 Ollama 中可使用
原生 thinking；不会伪造 Anthropic 签名或 Responses 加密 reasoning。Gemini 的外来工具历史
使用其[官方迁移标记](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures#faqs)
`skip_thought_signature_validator`；原生 Gemini 历史的签名和并行调用位置不改变。

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
