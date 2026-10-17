# anthropic 标准请求与返回

快照：2026-10-17。此处描述原生 HTTP 协议，字段完整性以所附 Schema 的全部 `$defs` 为准；示例按场景拆分。

接口：`POST https://api.anthropic.com/v1/messages`。

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
| [request.beta.schema.json](request.beta.schema.json) | 独立 beta 结构；需要相应功能 header 和模型支持。 |
| [response.beta.schema.json](response.beta.schema.json) | 独立 beta 结构；需要相应功能 header 和模型支持。 |
| [stream-event.beta.schema.json](stream-event.beta.schema.json) | 独立 beta 结构；需要相应功能 header 和模型支持。 |
| [stream.citations.sse](stream.citations.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.error.sse](stream.error.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.sse](stream.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.tools.sse](stream.tools.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |

## 请求顶层字段

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `max_tokens` | integer | 是 |
| `messages` | array of `MessageParam` | 是 |
| `model` | `claude-haiku-5-5` / `claude-sonnet-5-5` / `claude-fable-5-1` / `claude-opus-5-5` / `claude-mythos-5-1` / `claude-sonnet-5` / `claude-fable-5` / `claude-mythos-5` / `claude-opus-5` / `claude-opus-4-8` / `claude-opus-4-7` / `claude-mythos-preview` / `claude-opus-4-6` / `claude-sonnet-4-6` / `claude-haiku-4-5` / `claude-haiku-4-5-20251001` / `claude-opus-4-5` / `claude-opus-4-5-20251101` / `claude-sonnet-4-5` / `claude-sonnet-4-5-20250929` / string | 是 |
| `cache_control` | `CacheControlEphemeralParam` / null | 否；可能有条件要求 |
| `container` | `ContainerParams` / string / null | 否；可能有条件要求 |
| `diagnostics` | `DiagnosticsParam` / null | 否；可能有条件要求 |
| `inference_geo` | string / null | 否；可能有条件要求 |
| `metadata` | `MetadataParam` | 否；可能有条件要求 |
| `output_config` | `OutputConfigParam` | 否；可能有条件要求 |
| `service_tier` | `auto` / `standard_only` | 否；可能有条件要求 |
| `speed` | `standard` / `fast` / null | 否；可能有条件要求 |
| `stop_sequences` | array of string | 否；可能有条件要求 |
| `system` | string / array of `TextBlockParam` | 否；可能有条件要求 |
| `thinking` | `ThinkingConfigEnabledParam` / `ThinkingConfigDisabledParam` / `ThinkingConfigBetweenToolsParam` / `ThinkingConfigAdaptiveParam` | 否；可能有条件要求 |
| `tool_choice` | `ToolChoiceAutoParam` / `ToolChoiceAnyParam` / `ToolChoiceToolParam` / `ToolChoiceNoneParam` | 否；可能有条件要求 |
| `tools` | array of `ToolParam` / `ToolBash20250124Param` / `CodeExecutionTool20250522Param` / `CodeExecutionTool20250825Param` / `CodeExecutionTool20260120Param` / `CodeExecutionTool20260521Param` / `BrowserToolset20260801Param` / `MemoryTool20250818Param` / `ComputerToolset20260801Param` / `ToolTextEditor20250124Param` / `ToolTextEditor20250429Param` / `ToolTextEditor20250728Param` / `WebSearchTool20250305Param` / `WebFetchTool20250910Param` / `WebSearchTool20260209Param` / `WebFetchTool20260209Param` / `WebFetchTool20260309Param` / `WebSearchTool20260318Param` / `WebFetchTool20260318Param` / `ToolSearchToolBm25_20251119Param` / `ToolSearchToolRegex20251119Param` | 否；可能有条件要求 |
| `stream` | `True` | 否；可能有条件要求 |

## 完整响应顶层字段

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `id` | string | 是 |
| `container` | `Container` / null | 否；可能有条件要求 |
| `content` | array of `TextBlock` / `ThinkingBlock` / `RedactedThinkingBlock` / `ToolUseBlock` / `ServerToolUseBlock` / `WebSearchToolResultBlock` / `WebFetchToolResultBlock` / `CodeExecutionToolResultBlock` / `BashCodeExecutionToolResultBlock` / `TextEditorCodeExecutionToolResultBlock` / `ToolSearchToolResultBlock` / `ContainerUploadBlock` | 是 |
| `diagnostics` | `Diagnostics` / null | 否；可能有条件要求 |
| `model` | `claude-haiku-5-5` / `claude-sonnet-5-5` / `claude-fable-5-1` / `claude-opus-5-5` / `claude-mythos-5-1` / `claude-sonnet-5` / `claude-fable-5` / `claude-mythos-5` / `claude-opus-5` / `claude-opus-4-8` / `claude-opus-4-7` / `claude-mythos-preview` / `claude-opus-4-6` / `claude-sonnet-4-6` / `claude-haiku-4-5` / `claude-haiku-4-5-20251001` / `claude-opus-4-5` / `claude-opus-4-5-20251101` / `claude-sonnet-4-5` / `claude-sonnet-4-5-20250929` / string | 是 |
| `role` | `assistant` | 是 |
| `stop_details` | `RefusalStopDetails` / null | 否；可能有条件要求 |
| `stop_reason` | `end_turn` / `max_tokens` / `stop_sequence` / `tool_use` / `pause_turn` / `refusal` / `model_context_window_exceeded` / null | 否；可能有条件要求 |
| `stop_sequence` | string / null | 否；可能有条件要求 |
| `type` | `message` | 是 |
| `usage` | `Usage` | 是 |

“否”仅表示来源 Schema 没有将字段设为所有分支必需，不能据此省略协议语义要求、角色要求或模型要求。对象内部的字段和所有工具/媒体分支见对应 Schema。

## 流事件

当前结构包含 8 个事件 type：

- `content_block_delta`
- `content_block_start`
- `content_block_stop`
- `error`
- `message_delta`
- `message_start`
- `message_stop`
- `ping`

## 语义与组合规则

- REST body 使用 model/max_tokens/messages；system 是顶层字段。messages[].role 只有 user / assistant，不存在消息级 system/developer/tool 角色。
- content 可以是字符串或原生内容块列表。tool_use.input 是对象；tool_result 位于 user 内容中，使用 tool_use_id 关联，可携带文本/媒体内容和 is_error。
- 工具结果必须紧接对应调用，结果块在用户文本之前；同一轮的并行工具结果应完整返回。结构 Schema 不检查跨消息历史关联。
- thinking / redacted_thinking 块需要原始签名或不透明数据，必须按原顺序回放。签名占位符只能说明位置，不能发给服务端充当真实签名。
- thinking 的 enabled / adaptive / disabled、budget_tokens / display，与输出上限、工具选择、模型能力有联动限制。预算和 effort 的支持不是所有模型通用。
- 官方允许某些模型以 max_tokens=0 仅填充缓存；本项目当前要求正数，是库自己的配置边界，不是协议字段的统一最小值。
- stream-event.schema.json 在官方 SDK 的六类内容/消息事件之外补入文档中的 ping 和 error；每个内容块都有 start → delta → stop，最后 message_delta → message_stop。
- message_delta.usage 是累计计数，不能逐帧相加；输入缓存读/写与未缓存输入分开计数。citations_delta 的 citation 追加到对应文本块，signature_delta 在思考块关闭前到达。
- request.beta.schema.json、response.beta.schema.json、stream-event.beta.schema.json 是独立预览协议结构，使用对应功能要求的 anthropic-beta header；不要默认对稳定接口启用全部 beta。
- SDK 的 betas、user_profile_id、workspace_id 是 header 参数，已从 JSON body Schema 中移除。稳定版请求和 beta 请求分别保留，未混成一个大对象。
- stream.error.sse 是错误片段，可出现在已经开始的流内；不追加 message_stop 来伪装成功。

## 来源

- [官方来源 1](https://platform.claude.com/docs/en/api/messages/create)
- [官方来源 2](https://platform.claude.com/docs/en/build-with-claude/streaming)
- [官方来源 3](https://platform.claude.com/docs/en/build-with-claude/citations)
- [官方来源 4](https://platform.claude.com/docs/en/api/errors)
- [版本、哈希和结构调整记录](../sources.json)
- [目录说明及验证方法](../README.md)
