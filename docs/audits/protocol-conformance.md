# 协议定义与测试预期复核

> 历史验收记录：本文描述对应审计时点。当前翻译字段契约以[翻译字段边界](../design/translation-boundaries.md)为准；文中未知字段透传、恢复记录及开放模型扩展结论已撤销。

复核日期：2026-10-17。依据本次查阅的官方在线定义和本地官方 SDK 类型，
检查五协议之间消息、工具与流事件的预期结果；发现的实现偏差与测试样例同步修正。

## 定义来源与范围

| 协议 | 官方定义 | 本地校验 SDK |
| --- | --- | --- |
| openai_chat | [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) | openai 2.54.0 |
| openai_responses | [Responses 流事件](https://developers.openai.com/api/reference/resources/responses/streaming-events)、[创建 Responses](https://developers.openai.com/api/reference/resources/responses/methods/create) | openai 2.54.0 |
| anthropic | [Messages](https://platform.claude.com/docs/en/api/messages/create)、[Streaming](https://platform.claude.com/docs/en/build-with-claude/streaming)、[Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) | anthropic 0.125.0 |
| gemini | [GenerateContent API：Content、Part、FunctionDeclaration、UsageMetadata](https://ai.google.dev/api/generate-content) | google-genai 1.75.0 |
| ollama | [Chat](https://docs.ollama.com/api/chat)、[Tool calling](https://docs.ollama.com/capabilities/tool-calling) | ollama 0.6.3 |

版本号记录实际运行校验的 SDK，不表示这些版本始终为最新发行版。
Gemini 对应 `generate_content` / `generate_content_stream`，不包含独立的 Interactions API；
字典采用 Python SDK 的 snake_case，REST 文档中的 camelCase 由 SDK 处理。

## 测试如何避免自我验证

[原生样例](../../tests/helpers/protocols.py)分别手写五协议历史、工具声明和完整流事件，
不调用待测转换器生成源样例。[协议契约测试](../../tests/contract/test_protocol_conformance.py)
独立比较目标原生结构，并使用官方 SDK 的消息输入类型、工具输入类型与具体事件类型校验。
覆盖消息 25 条路径、工具 25 条路径、图片 25 条路径，以及实时/完整批次流各 25 条路径。
另有截断、过滤、思考块、Responses 请求元数据和校验器反例测试。

SDK 解析通过不等于协议符合：部分模型允许未知字段，部分 TypedDict 会静默忽略额外字段，
Iterable 内容块还可能延迟验证。测试会消费延迟校验结果、递归拒绝被忽略或不属于原生类型的键，
并补充 REST 定义要求而 SDK 标为可选的 Gemini 函数名称、响应对象及声明描述检查。
事件序列另行检查块的开启/关闭、Responses 项目 ID 对应、终态快照与单调序号。

`扩展保留测试`（原文件已移除）和部分
[流语义测试](../../tests/contract/test_decoding.py)故意含有未知字段或兼容服务扩展。
它们验证保留规则，不将这些样例视为对应官方 API 可以直接接收的请求。
例如 `reasoning_content` 属于兼容 Chat 服务扩展，不属于官方 OpenAI Chat 消息定义。

## 本次校正

| 范围 | 已校正的预期与实现 |
| --- | --- |
| Chat 流 | 结束原因在正常 choice 中；启用用量时，随后独立发送 `choices=[]` 的 usage 块。只有 Chat SSE 使用 `[DONE]` 尾标记。 |
| Responses 消息 | assistant output message 补齐 id/status；reasoning item 补齐 id；跨来源 input_image 默认 detail=auto。函数声明提供 parameters/strict；外来思考只作可表达文本，不伪造原生加密 reasoning。 |
| Responses 流 | 补齐 response 必填 envelope、text delta/done 的 logprobs、arguments.done 的 name，保留工具设置元数据；文本与工具完整生命周期及输出项 ID 一致。 |
| Responses 终态 | length 转 response.incomplete/max_output_tokens；content_filter 转 response.incomplete/content_filter。上游原生 incomplete 仍按既有异常边界抛出。 |
| Anthropic | 工具声明/调用/结果使用其原生层级；message_delta.usage 包含 output_tokens；签名不从 Gemini 伪造迁移。 |
| Gemini | 从关联调用恢复 function_response.name；response 保持对象；函数声明使用 parameters_json_schema，缺少 description 时补空串。 |
| Ollama | function.arguments 保持对象，结果关联 tool_name；跨协议输出调用采用 function 包装，不强加 Chat ID/type；声明不携带 unsupported strict。图片保持 SDK 支持的 base64 字符串。 |
| 外来字段 | 跨协议渲染时排除目标类型不支持的 thought_signature、signature、媒体 detail 等已知来源字段；同协议原生快照仍保留。 |
| Token 计数 | 按各 API 的输入/输出、缓存和思考计数定义换算，避免将来源 usage 字典直接放入目标字段。 |

[用量专项测试](../../tests/unit/test_usage_conversion.py)明确验证：Anthropic 总输入包含未缓存、
cache-read 和 cache-write；Gemini prompt 已含缓存，额外工具提示计入输入，thoughts 计入输出。
累计流用量采用覆盖更新，不将累计计数重复相加。Ollama 独有耗时指标不迁移为其他 API 的 token 字段。

## 合成字段与验证边界

跨协议来源没有 Responses 所需的项 ID/status、logprobs 或请求设置时，转换器生成项 ID、
完成状态、空 logprobs，并为 tools/tool_choice/parallel_tool_calls 使用有效默认值。
来源为 Responses 且报告这些设置时保留原值；默认值不代表原上游请求的实际配置。
Responses 必需的详细 token 拆分未报告时使用 0；来源仅有 total 而没有输入/输出计数时，
省略需要完整拆分的 Chat/Responses usage，不能据总量编造拆分。

未知字段按原层级保留、转换值优先以及同协议与未经修改中间态的无损规则继续成立，
见[当前翻译字段边界](../design/translation-boundaries.md)。完整流的累积/重建与原生块的精确往返是不同测试路径。
协议专属签名和不透明块不保证跨协议的有效语义。

本次校验覆盖已实现的文本、函数工具回合、图片、思考及相关流式终态；
不代表音视频、各家全部服务端工具、未来新增事件或 Gemini Interactions 均已实现。
验证包括本地 schema、语义断言与真实官方 SDK 的本地 transport，未调用远端模型服务。

## 本轮追加验收

后续独立审计 A01—A11 已修复，并接入本次最新官方来源快照的完整请求/流结构。
额外修复 Responses 必填 nullable 字段、output_text.logprobs、原生父 item ID 与多 part 分组，
明确外来字段投影与不能安全表达时的本地拒绝策略。官方 Chat 兼容字段回放改为显式开启。
实际命令、验证范围、SDK 边界和代码归属见[修复验收记录](protocol-fixes-2026-10-17.md)。
