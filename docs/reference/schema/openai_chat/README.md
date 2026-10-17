# openai_chat 标准请求与返回

快照：2026-10-17。此处描述原生 HTTP 协议，字段完整性以所附 Schema 的全部 `$defs` 为准；示例按场景拆分。

接口：`POST https://api.openai.com/v1/chat/completions`。

## 文件

| 文件 | 用途 |
| --- | --- |
| [http.json](http.json) | URL、鉴权 header、流内容类型和结束标记。 |
| [request.json](request.json) | 普通非流式请求模板；替换占位符后选择适用模型。 |
| [response.json](response.json) | 完整非流式响应模板。 |
| [error.json](error.json) | HTTP 错误 body 模板；不是成功响应。 |
| [templates.json](templates.json) | 工具、媒体、思考、结构化输出、拒绝等独立场景。 |
| [request.schema.json](request.schema.json) | 全量请求字段、分支与嵌套类型。 |
| [response.schema.json](response.schema.json) | 全量返回字段和内容分支。 |
| [stream-event.schema.json](stream-event.schema.json) | 单个已解析流帧的原生 JSON 结构。 |
| [error.schema.json](error.schema.json) | 错误 body 结构。 |
| [stream.error.sse](stream.error.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.refusal.sse](stream.refusal.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.sse](stream.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.tools.sse](stream.tools.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |

## 请求顶层字段

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `metadata` | `Metadata` | 否；可能有条件要求 |
| `top_logprobs` | integer / null | 否；可能有条件要求 |
| `temperature` | number / null | 否；可能有条件要求 |
| `top_p` | number / null | 否；可能有条件要求 |
| `user` | string / null | 否；可能有条件要求 |
| `safety_identifier` | string / null | 否；可能有条件要求 |
| `prompt_cache_key` | string / null | 否；可能有条件要求 |
| `prompt_cache_retention` | `in_memory` / `24h` / null | 否；可能有条件要求 |
| `prompt_cache_options` | `PromptCacheOptionsParam` | 否；可能有条件要求 |
| `messages` | array of `ChatCompletionRequestMessage` | 是 |
| `model` | `ModelIdsShared` | 是 |
| `service_tier` | `ServiceTier` | 否；可能有条件要求 |
| `modalities` | `ResponseModalities` | 否；可能有条件要求 |
| `verbosity` | `Verbosity` | 否；可能有条件要求 |
| `reasoning_effort` | `ReasoningEffort` | 否；可能有条件要求 |
| `max_completion_tokens` | integer / null | 否；可能有条件要求 |
| `frequency_penalty` | number / null | 否；可能有条件要求 |
| `presence_penalty` | number / null | 否；可能有条件要求 |
| `web_search_options` | object | 否；可能有条件要求 |
| `response_format` | `ResponseFormatText` / `ResponseFormatJsonSchema` / `ResponseFormatJsonObject` | 否；可能有条件要求 |
| `audio` | object / null | 否；可能有条件要求 |
| `store` | boolean / null | 否；可能有条件要求 |
| `moderation` | `ModerationParam` / null | 否；可能有条件要求 |
| `stream` | boolean / null | 否；可能有条件要求 |
| `stop` | `StopConfiguration` | 否；可能有条件要求 |
| `logit_bias` | object / null | 否；可能有条件要求 |
| `logprobs` | boolean / null | 否；可能有条件要求 |
| `max_tokens` | integer / null | 否；可能有条件要求 |
| `n` | integer / null | 否；可能有条件要求 |
| `prediction` | `PredictionContent` / null | 否；可能有条件要求 |
| `seed` | integer / null | 否；可能有条件要求 |
| `stream_options` | `ChatCompletionStreamOptions` | 否；可能有条件要求 |
| `tools` | array of `ChatCompletionTool` / `CustomToolChatCompletions` | 否；可能有条件要求 |
| `tool_choice` | `ChatCompletionToolChoiceOption` | 否；可能有条件要求 |
| `parallel_tool_calls` | `ParallelToolCalls` | 否；可能有条件要求 |
| `function_call` | `none` / `auto` / `ChatCompletionFunctionCallOption` | 否；可能有条件要求 |
| `functions` | array of `ChatCompletionFunctions` | 否；可能有条件要求 |

## 完整响应顶层字段

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `id` | string | 是 |
| `choices` | array of object | 是 |
| `created` | integer | 是 |
| `model` | string | 是 |
| `metadata` | `Metadata` | 否；可能有条件要求 |
| `service_tier` | `ServiceTier` | 否；可能有条件要求 |
| `system_fingerprint` | string | 否；可能有条件要求 |
| `object` | `chat.completion` | 是 |
| `usage` | `CompletionUsage` | 否；可能有条件要求 |
| `moderation` | `ChatCompletionModeration` / null | 否；可能有条件要求 |

“否”仅表示来源 Schema 没有将字段设为所有分支必需，不能据此省略协议语义要求、角色要求或模型要求。对象内部的字段和所有工具/媒体分支见对应 Schema。

## 流事件

此协议没有统一的事件 type 字符串，使用原生响应对象字段判断增量和终态。单帧顶层字段如下：

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `id` | string | 是 |
| `choices` | array of object | 是 |
| `created` | integer | 是 |
| `model` | string | 是 |
| `obfuscation` | string | 否；可能有条件要求 |
| `service_tier` | `ServiceTier` | 否；可能有条件要求 |
| `system_fingerprint` | string | 否；可能有条件要求 |
| `object` | `chat.completion.chunk` | 是 |
| `usage` | `CompletionUsage` / null | 否；可能有条件要求 |
| `moderation` | `ChatCompletionModeration` / null | 否；可能有条件要求 |

## 语义与组合规则

- 请求消息包含 developer / system / user / assistant / tool，以及已弃用的 function 变体。图片、音频、文件、文本和 refusal 各自使用原生内容类型；角色允许的内容并不相同。
- 函数 arguments 是 JSON **字符串**；工具结果通过 tool_call_id 关联。function_call/functions 是已弃用分支，与当前 tools 分支分别保留。
- 官方原生请求不使用 reasoning_content、thinking、enable_thinking 等兼容服务扩展。reasoning_effort 的可选值和模型支持范围以官方模型文档为准。
- `max_tokens` 是旧参数，优先使用 `max_completion_tokens`；某些推理模型不接受普通采样选项。audio 与 modalities、response_format 与 strict JSON Schema 的限制需一起判断。
- 流增量在 choices[].delta，finish_reason 是 choice 级字段。include_usage 启用后最终用量块 choices=[]；读取到 [DONE] 才消费完传输尾标记。
- refusal 应累积到拒绝内容，不能当作空文本丢弃。content_filter 与普通 stop 不等价。
- 仅 stream.sse / stream.tools.sse / stream.refusal.sse 是成功流。stream.error.sse 是已启动流发生错误的单帧片段，不带成功尾标记。

## 来源

- [官方来源 1](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [官方来源 2](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)
- [版本、哈希和结构调整记录](../sources.json)
- [目录说明及验证方法](../README.md)
