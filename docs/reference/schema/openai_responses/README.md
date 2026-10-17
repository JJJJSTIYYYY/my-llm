# openai_responses 标准请求与返回

快照：2026-10-17。此处描述原生 HTTP 协议，字段完整性以所附 Schema 的全部 `$defs` 为准；示例按场景拆分。

接口：`POST https://api.openai.com/v1/responses`。

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
| [request.upstream.schema.json](request.upstream.schema.json) | 未调整 InputItem oneOf 的上游结构快照。 |
| [stream.error.sse](stream.error.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.failed.sse](stream.failed.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.incomplete.sse](stream.incomplete.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.refusal.sse](stream.refusal.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.sse](stream.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.tools.sse](stream.tools.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |

## 请求顶层字段

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `metadata` | `Metadata` | 否；可能有条件要求 |
| `top_logprobs` | integer | 否；可能有条件要求 |
| `temperature` | number / null | 否；可能有条件要求 |
| `top_p` | number / null | 否；可能有条件要求 |
| `user` | string / null | 否；可能有条件要求 |
| `safety_identifier` | string / null | 否；可能有条件要求 |
| `prompt_cache_key` | string / null | 否；可能有条件要求 |
| `prompt_cache_retention` | `in_memory` / `24h` / null | 否；可能有条件要求 |
| `prompt_cache_options` | `ResponsePromptCacheOptionsParam` | 否；可能有条件要求 |
| `previous_response_id` | string / null | 否；可能有条件要求 |
| `model` | `ModelIdsResponses` | 否；可能有条件要求 |
| `background` | boolean / null | 否；可能有条件要求 |
| `max_tool_calls` | integer / null | 否；可能有条件要求 |
| `text` | `ResponseTextParam` | 否；可能有条件要求 |
| `tools` | `ToolsArray` | 否；可能有条件要求 |
| `tool_choice` | `ToolChoiceParam` | 否；可能有条件要求 |
| `prompt` | `Prompt` | 否；可能有条件要求 |
| `access_programs` | `AccessProgramsParam` | 否；可能有条件要求 |
| `service_tier` | `ServiceTierResponses` | 否；可能有条件要求 |
| `truncation` | `auto` / `disabled` / null | 否；可能有条件要求 |
| `reasoning` | `Reasoning` / null | 否；可能有条件要求 |
| `input` | `InputParam` | 否；可能有条件要求 |
| `include` | array of `IncludeEnum` / null | 否；可能有条件要求 |
| `parallel_tool_calls` | boolean / null | 否；可能有条件要求 |
| `store` | boolean / null | 否；可能有条件要求 |
| `instructions` | string / null | 否；可能有条件要求 |
| `moderation` | `ModerationParam` / null | 否；可能有条件要求 |
| `stream` | boolean / null | 否；可能有条件要求 |
| `stream_options` | `ResponseStreamOptions` | 否；可能有条件要求 |
| `conversation` | `ConversationParam` / null | 否；可能有条件要求 |
| `context_management` | array of `ContextManagementParam` / null | 否；可能有条件要求 |
| `max_output_tokens` | integer / null | 否；可能有条件要求 |

## 完整响应顶层字段

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `metadata` | `Metadata` | 是 |
| `top_logprobs` | integer / null | 否；可能有条件要求 |
| `temperature` | number / null | 是 |
| `top_p` | number / null | 是 |
| `user` | string / null | 否；可能有条件要求 |
| `safety_identifier` | string / null | 否；可能有条件要求 |
| `prompt_cache_key` | string / null | 否；可能有条件要求 |
| `prompt_cache_retention` | `in_memory` / `24h` / null | 否；可能有条件要求 |
| `previous_response_id` | string / null | 否；可能有条件要求 |
| `model` | `ModelIdsResponses` | 是 |
| `background` | boolean / null | 否；可能有条件要求 |
| `max_tool_calls` | integer / null | 否；可能有条件要求 |
| `text` | `ResponseTextParam` | 否；可能有条件要求 |
| `tools` | `ToolsArray` | 是 |
| `tool_choice` | `ToolChoiceParam` | 是 |
| `prompt` | `Prompt` | 否；可能有条件要求 |
| `service_tier` | `ServiceTierResponses` | 否；可能有条件要求 |
| `truncation` | `auto` / `disabled` / null | 否；可能有条件要求 |
| `id` | string | 是 |
| `object` | `response` | 是 |
| `status` | `completed` / `failed` / `in_progress` / `cancelled` / `queued` / `incomplete` | 否；可能有条件要求 |
| `access_programs` | `AccessProgramsBody` / null | 是 |
| `created_at` | number | 是 |
| `completed_at` | number / null | 否；可能有条件要求 |
| `error` | `ResponseError` | 是 |
| `incomplete_details` | object / null | 是 |
| `output` | array of `OutputItem` | 是 |
| `reasoning` | `Reasoning` / null | 否；可能有条件要求 |
| `instructions` | string / array of 组合类型 / null | 是 |
| `output_text` | string / null | 否；可能有条件要求 |
| `usage` | `ResponseUsage` / null | 否；可能有条件要求 |
| `prompt_cache_options` | `PromptCacheOptions` | 否；可能有条件要求 |
| `prompt_cache_diagnostics` | `PromptCacheDiagnostics` | 否；可能有条件要求 |
| `moderation` | `Moderation` / null | 否；可能有条件要求 |
| `parallel_tool_calls` | boolean | 是 |
| `conversation` | `ResponseConversation` / null | 否；可能有条件要求 |
| `max_output_tokens` | integer / null | 否；可能有条件要求 |

“否”仅表示来源 Schema 没有将字段设为所有分支必需，不能据此省略协议语义要求、角色要求或模型要求。对象内部的字段和所有工具/媒体分支见对应 Schema。

## 流事件

此协议没有统一的事件 type 字符串，使用原生响应对象字段判断增量和终态。单帧顶层字段如下：

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `type` | `response.custom_tool_call_input.done` | 是 |
| `sequence_number` | integer | 是 |
| `delta` | string | 否；可能有条件要求 |
| `output_index` | integer | 否；可能有条件要求 |
| `item_id` | string | 否；可能有条件要求 |
| `code` | string / null | 否；可能有条件要求 |
| `response` | `Response` | 否；可能有条件要求 |
| `content_index` | integer | 否；可能有条件要求 |
| `part` | object | 否；可能有条件要求 |
| `message` | string | 否；可能有条件要求 |
| `param` | string / null | 否；可能有条件要求 |
| `arguments` | string | 否；可能有条件要求 |
| `command_index` | integer | 否；可能有条件要求 |
| `command` | string | 否；可能有条件要求 |
| `obfuscation` | string | 否；可能有条件要求 |
| `output` | array of `FunctionShellCallOutputContent` | 否；可能有条件要求 |
| `item` | `OutputItem` | 否；可能有条件要求 |
| `summary_index` | integer | 否；可能有条件要求 |
| `status` | `incomplete` | 否；可能有条件要求 |
| `text` | string | 否；可能有条件要求 |
| `refusal` | string | 否；可能有条件要求 |
| `logprobs` | array of `ResponseLogProb` | 否；可能有条件要求 |
| `partial_image_index` | integer | 否；可能有条件要求 |
| `partial_image_b64` | string | 否；可能有条件要求 |
| `size` | string | 否；可能有条件要求 |
| `quality` | string | 否；可能有条件要求 |
| `background` | string | 否；可能有条件要求 |
| `output_format` | string | 否；可能有条件要求 |
| `annotation_index` | integer | 否；可能有条件要求 |
| `annotation` | `Annotation` / null | 否；可能有条件要求 |
| `input` | string | 否；可能有条件要求 |

## 语义与组合规则

- input 可以是字符串或原生 item 列表；文本输入为 input_text，assistant 输出为 output_text / refusal。函数工具定义是扁平结构，函数参数 arguments 是 JSON 字符串。
- input/output 包含消息、函数、reasoning、压缩、检索、浏览器/计算机、代码执行、MCP 等多种 item。完整定义在 Schema 的 $defs，模板不将它们合并为同一个“万能 item”。
- reasoning 的 summary / encrypted_content 与原始 item 一起保留；签名和密文占位符不能代替真实的服务端数据。
- previous_response_id 和 conversation 不应同时设置；存储、后台、prompt、上下文管理及 include 等可选参数需要按实际场景组合。model 可由已保存 prompt 等配置提供，不能仅凭 Schema 未标 required 推断服务端永不要求模型。
- response.output_item.added/done 的 item.id 与所有 item_id 一致；output_index、content_index、summary_index 分别作用于不同层级。sequence_number 按事件递增。
- response.completed / response.incomplete / response.failed 是不同终态；流不使用 Chat 的 [DONE]。error 事件也不是 response.completed。
- output_text 便捷属性不能取代 output 数组；工具、拒绝、推理和其他原生 item 可能存在于 output。
- request.upstream.schema.json 保留上游 oneOf。request.schema.json 仅将重叠的 InputItem.oneOf 改为 anyOf；其余定义不因此收窄。变更依据在 sources.json。
- 当前官方 OpenAPI 将 access_programs 列为 Response envelope 的必填键，并要求 output_text 内容中的 logprobs；模板按本次快照补齐，不能用旧版 SDK 默认值替代当前来源要求。
- stream.error.sse 是独立 error 事件片段；其他 stream.*.sse 给出完整的普通/工具/拒绝/失败/截断生命周期。

## 来源

- [官方来源 1](https://developers.openai.com/api/reference/resources/responses/methods/create)
- [官方来源 2](https://developers.openai.com/api/reference/resources/responses/streaming-events)
- [版本、哈希和结构调整记录](../sources.json)
- [目录说明及验证方法](../README.md)
