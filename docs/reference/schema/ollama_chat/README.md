# ollama_chat 标准请求与返回

快照：2026-10-17。此处描述原生 HTTP 协议，字段完整性以所附 Schema 的全部 `$defs` 为准；示例按场景拆分。

接口：`POST http://localhost:11434/api/chat`。

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
| [stream.error.ndjson](stream.error.ndjson) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.ndjson](stream.ndjson) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.tools.ndjson](stream.tools.ndjson) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |

## 请求顶层字段

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `model` | string | 是 |
| `messages` | array of `ChatMessage` | 是 |
| `tools` | array of `ToolDefinition` | 否；可能有条件要求 |
| `format` | `json` / object | 否；可能有条件要求 |
| `options` | `ModelOptions` | 否；可能有条件要求 |
| `stream` | boolean | 否；可能有条件要求 |
| `think` | `ThinkValue` | 否；可能有条件要求 |
| `keep_alive` | string / number | 否；可能有条件要求 |
| `logprobs` | boolean | 否；可能有条件要求 |
| `top_logprobs` | integer | 否；可能有条件要求 |

## 完整响应顶层字段

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `model` | string | 否；可能有条件要求 |
| `created_at` | string | 否；可能有条件要求 |
| `message` | object | 否；可能有条件要求 |
| `done` | boolean | 否；可能有条件要求 |
| `done_reason` | string | 否；可能有条件要求 |
| `total_duration` | integer | 否；可能有条件要求 |
| `load_duration` | integer | 否；可能有条件要求 |
| `prompt_eval_count` | integer | 否；可能有条件要求 |
| `prompt_eval_cached_count` | integer | 否；可能有条件要求 |
| `prompt_eval_duration` | integer | 否；可能有条件要求 |
| `eval_count` | integer | 否；可能有条件要求 |
| `eval_duration` | integer | 否；可能有条件要求 |
| `logprobs` | array of `Logprob` | 否；可能有条件要求 |
| `remote_model` | string | 否；可能有条件要求 |
| `remote_host` | string | 否；可能有条件要求 |

“否”仅表示来源 Schema 没有将字段设为所有分支必需，不能据此省略协议语义要求、角色要求或模型要求。对象内部的字段和所有工具/媒体分支见对应 Schema。

## 流事件

此协议没有统一的事件 type 字符串，使用原生响应对象字段判断增量和终态。单帧顶层字段如下：

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `model` | string | 否；可能有条件要求 |
| `created_at` | string | 否；可能有条件要求 |
| `message` | object | 否；可能有条件要求 |
| `done` | boolean | 否；可能有条件要求 |
| `done_reason` | string | 否；可能有条件要求 |
| `total_duration` | integer | 否；可能有条件要求 |
| `load_duration` | integer | 否；可能有条件要求 |
| `prompt_eval_count` | integer | 否；可能有条件要求 |
| `prompt_eval_cached_count` | integer | 否；可能有条件要求 |
| `prompt_eval_duration` | integer | 否；可能有条件要求 |
| `eval_count` | integer | 否；可能有条件要求 |
| `eval_duration` | integer | 否；可能有条件要求 |
| `logprobs` | array of `Logprob` | 否；可能有条件要求 |
| `remote_model` | string | 否；可能有条件要求 |
| `remote_host` | string | 否；可能有条件要求 |

## 语义与组合规则

- messages[] 使用原生 role/content，content 为字符串；图片是 base64 字符串数组，不能假设字节一定为 PNG。工具结果 role=tool，使用 tool_name，并按支持情况保留 tool_call_id。
- tool_calls[].function.arguments 是对象，function.index 与可选调用 id 位于其原生层级；不要套用 Chat 的 arguments 字符串或强制的 type=function 调用结构。
- think 接受布尔、null 或模型定义的字符串。先用 /api/show 检查支持值；low/medium/high 不是所有模型的完整通用枚举。
- options 包含模型生成/运行选项；num_predict 用于输出上限，-1/-2 等特殊值是否适用应看模型与当前服务端。不要套用本库 max_tokens 的正数约束。
- REST stream 默认 true；非流式模板显式设 false。NDJSON 每行是独立 JSON，done=true 是终态；没有 SSE event/data 前缀，也没有 [DONE]。
- 同时累积 message.content、thinking 和完整工具调用；最后一行可能携带 done_reason、计数和耗时，不能因为 delta 文本为空而忽略。
- 时间字段是 ISO 8601，耗时字段为纳秒。prompt_eval_cached_count 是缓存输入计数，区别于 prompt_eval_count。
- 上游 OpenAPI 遗漏的消息 thinking/tool_name/tool_call_id、调用 id/index、部分 options、终帧指标和 cloud metadata，已按官方公开服务端类型及文档补列，逐字段依据见 sources.json 的 ollama_additions。
- 结构 Schema 不把所有服务端允许的未知 options 键收窄。内部调试字段和未正式文档化的 runner/shift/truncate 等开关不作为稳定公开请求模板承诺。
- stream.error.ndjson 是错误片段，不追加 done=true 声称成功。

## 来源

- [官方来源 1](https://docs.ollama.com/api/chat)
- [官方来源 2](https://docs.ollama.com/capabilities/tool-calling)
- [官方来源 3](https://docs.ollama.com/capabilities/streaming)
- [官方来源 4](https://docs.ollama.com/api/errors)
- [版本、哈希和结构调整记录](../sources.json)
- [目录说明及验证方法](../README.md)
