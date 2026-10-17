# gemini 标准请求与返回

快照：2026-10-17。此处描述原生 HTTP 协议，字段完整性以所附 Schema 的全部 `$defs` 为准；示例按场景拆分。

接口：`POST https://generativelanguage.googleapis.com/v1beta/models/<MODEL_ID>:generateContent`。

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
| [stream.sse](stream.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |
| [stream.tools.sse](stream.tools.sse) | 原始传输帧模板；error 文件为错误片段，其他文件为完整示例序列。 |

## 请求顶层字段

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `generationConfig` | `GenerationConfig` | 否；可能有条件要求 |
| `tools` | array of `Tool` | 否；可能有条件要求 |
| `cachedContent` | string | 否；可能有条件要求 |
| `contents` | array of `Content` | 是 |
| `serviceTier` | `unspecified` / `standard` / `flex` / `priority` | 否；可能有条件要求 |
| `store` | boolean | 否；可能有条件要求 |
| `systemInstruction` | `Content` | 否；可能有条件要求 |
| `continuationToken` | string | 否；可能有条件要求 |
| `toolConfig` | `ToolConfig` | 否；可能有条件要求 |
| `labels` | object | 否；可能有条件要求 |
| `safetySettings` | array of `SafetySetting` | 否；可能有条件要求 |

## 完整响应顶层字段

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `modelVersion` | string | 否；可能有条件要求 |
| `candidates` | array of `Candidate` | 否；可能有条件要求 |
| `responseId` | string | 否；可能有条件要求 |
| `promptFeedback` | `PromptFeedback` | 否；可能有条件要求 |
| `usageMetadata` | `UsageMetadata` | 否；可能有条件要求 |
| `modelStatus` | `ModelStatus` | 否；可能有条件要求 |

“否”仅表示来源 Schema 没有将字段设为所有分支必需，不能据此省略协议语义要求、角色要求或模型要求。对象内部的字段和所有工具/媒体分支见对应 Schema。

## 流事件

此协议没有统一的事件 type 字符串，使用原生响应对象字段判断增量和终态。单帧顶层字段如下：

| 字段 | 结构类型 / 枚举 | 所有分支均必需 |
| --- | --- | --- |
| `modelVersion` | string | 否；可能有条件要求 |
| `candidates` | array of `Candidate` | 否；可能有条件要求 |
| `responseId` | string | 否；可能有条件要求 |
| `promptFeedback` | `PromptFeedback` | 否；可能有条件要求 |
| `usageMetadata` | `UsageMetadata` | 否；可能有条件要求 |
| `modelStatus` | `ModelStatus` | 否；可能有条件要求 |

## 语义与组合规则

- 本目录采用 GenerateContent 的 REST camelCase。Python SDK 中的 system_instruction / function_call / thought_signature 等名称需要 SDK 转换；不能直接当作 REST 键发送。
- 模型放在 URL 的 models/<MODEL_ID>，JSON body 不再重复 model。generateContent 与 streamGenerateContent 使用相同 body；后者 URL 加 alt=sse，无 body.stream 字段。
- Content.role 为 user / model；systemInstruction 独立于 contents。Part 的 text、inlineData、fileData、functionCall、functionResponse、代码和服务端工具等数据分支互斥，thought/thoughtSignature 是元数据。仅有签名的空 Part 也可能有意义，不能按空文本丢弃。
- functionCall.args 与 functionResponse.response 是对象；函数响应保留名称及模型返回的可选 ID。并行调用和签名位置以原始响应为准。
- functionDeclarations[].parameters 使用 Gemini Schema（OBJECT/STRING 等枚举）；parametersJsonSchema 使用 JSON Schema（object/string）。两者互斥；response 与 responseJsonSchema 也互斥。
- generationConfig.responseSchema 与 responseJsonSchema 互斥。文档中的 JSON 输出、responseMimeType 与新版 responseFormat 需要按模型支持组合；Discovery 还可见已弃用 _responseJsonSchema，不把它作为首选模板。
- thinkingBudget 与 thinkingLevel 的适用模型和可取值不同，不建议同时设置。thoughtSignature 是不透明 base64，必须保留原 Part 和字节，不拼接、伪造或跨厂商迁移。
- candidates[].finishReason 可以是 STOP、MAX_TOKENS 或过滤/函数调用/其他特殊原因。输入拦截可能只有 promptFeedback、没有 candidates；不能要求每个响应都含非空 candidates。
- promptTokenCount 包含缓存；thoughtsTokenCount 和 toolUsePromptTokenCount 是独立计数。totalTokenCount 不能简单等同可见候选文本 token 数。
- 响应帧与完整响应共享 GenerateContentResponse 结构，SSE 无 [DONE]。看到 finishReason 后仍读完传输，以保留最终 usage/metadata。
- 不包含独立的 Interactions、Live 或 Vertex AI 协议；Schema 是官方 v1beta Discovery 的转换结果，语义互斥和模型规则另见官方文档。

## 来源

- [官方来源 1](https://ai.google.dev/api/generate-content)
- [官方来源 2](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures)
- [官方来源 3](https://ai.google.dev/gemini-api/docs/troubleshooting)
- [版本、哈希和结构调整记录](../sources.json)
- [目录说明及验证方法](../README.md)
