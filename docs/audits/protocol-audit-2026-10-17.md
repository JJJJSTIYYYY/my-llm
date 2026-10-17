# 四家官方协议独立审计

审计日期：2026-10-17。代码基线：`5c41153`，工作区起始状态干净。
本轮核对 OpenAI Chat Completions / Responses、Anthropic Messages、
Gemini GenerateContent 和 Ollama Chat，共五种协议实现。

**初次审计发现 11 类差异，其中 3 类 P1、8 类 P2；现已逐项修复。**
下文保留基线 `5c41153` 的问题描述，不代表当前实现仍有这些行为。
初次审计时既有 746 项测试通过；本轮已改为断言修复行为，并增加最新官方 Schema 校验。
完整修复、协议归属和代码复核证据见[修复验收记录](protocol-fixes-2026-10-17.md)。

## 修复结果

| 编号 | 当前行为与回归依据 |
| --- | --- |
| A01 | Gemini 原生 Schema 递归转换类型、nullable、引用与数量约束；JSON Schema 分支保留原值。Ollama SDK 会丢弃约束时在请求前明确拒绝。 |
| A02 | Anthropic tool_choice / metadata、Gemini tool_config / safety_settings 保留到真实 SDK 序列化结果；不支持的原生选项抛出 ConfigurationError。 |
| A03 | Chat / Responses refusal 进入块累积及目标生命周期；序号、项目 ID、完成快照保持对应。 |
| A04 | stop_sequence 转 Chat stop，同 Anthropic 保留匹配序列；过滤转 Gemini SAFETY。无法表达的特殊终态不伪装成功。 |
| A05 | 外来消息、调用与声明只投影目标可表达字段；thought=False、function.index、未知扩展不泄漏。 |
| A06 | 按图片字节辨认 PNG / JPEG / GIF / WebP；无法辨认时明确报错，不伪造 PNG。 |
| A07 | citations_delta 追加到最终文本块；延迟发布的文本也正确输出引用。 |
| A08 | Ollama 工具结果始终为字符串，纯文本块拼接、对象 JSON 序列化；Responses 可保留其支持的多模态结果部分。 |
| A09 | 外来思考降为目标可表达的普通文本或原生 Ollama thinking，不生成无签名 Anthropic thinking；同协议签名保持。 |
| A10 | Chat 不再生成 file.file_url；URL 文档保留为文本引用，不隐式下载。 |
| A11 | 拒绝重复原生终态及终态后内容；按源 Encoder 的声明保留 Chat / Gemini 合法尾部用量与元数据。 |

普通契约回归见 [test_audit_regressions.py](../../tests/contract/test_audit_regressions.py)，
A02 的实际传输及默认配置回归还见[原生 SDK 测试](../../tests/integration/test_native_sdk_transport.py)
和[请求选项测试](../../tests/unit/test_native_options.py)。

## 依据与验收边界

先读取本目录的 API、配置、无损边界、流状态、设计验收和既有协议复核文档，
再检查实现、官方在线文档与本地官方 SDK。SDK 版本为：
`openai 2.54.0`、`anthropic 0.125.0`、`google-genai 1.75.0`、`ollama 0.6.3`。
版本号是本轮运行环境的快照，不表示最新版本或最低依赖版本均已验证。

本轮遵守以下项目约束：

- 无损保证仅针对同协议与未经修改的中间态；不要求 A → B → A 无损。
- Codec 无状态，实时累积属于请求自己的 StreamState。
- 不执行工具，不提供部分工具 JSON 和中断恢复。
- Gemini 范围为 GenerateContent，不将 Interactions 规则套到当前 SDK 路径。
- `reasoning_content` 是兼容 Chat 服务扩展，不能据此声称它是官方 OpenAI 字段。
- 跨协议可以损失无对应表示的数据；生成目标协议不支持的已知字段或错误终态，
  与单纯的信息损失分别判断。

官方依据随具体发现列出。除本地 schema 和独立断言外，对拒绝、引用、
工具 Schema、原生参数、文件 URL 和工具结果，使用真实官方 SDK 与
`httpx.MockTransport` 检查了序列化请求或实际收集结果。
没有调用远端模型；文中“违反官方定义”不表示已经观察到远端 HTTP 错误。

## 发现总表

| 编号 | 原优先级 | 涉及协议 | 基线差异 |
| --- | --- | --- | --- |
| A01 | P1 | Gemini → 其他协议 | 原生 Schema 的大写类型枚举原样进入 JSON Schema。 |
| A02 | P1 | Anthropic / Gemini | 支持的原生请求选项被白名单静默丢弃。 |
| A03 | P1 | OpenAI Chat / Responses | 拒绝内容丢失，或绕过目标流生命周期和消息结构。 |
| A04 | P2 | Anthropic / Gemini / Chat | 停止序列转换产生非法 Chat 终态；过滤被写成 Gemini 正常停止。 |
| A05 | P2 | Gemini / Ollama → OpenAI 等 | 已知来源字段泄漏进目标消息和调用结构。 |
| A06 | P2 | Ollama → 图片协议 | 所有图片一律声明为 PNG，JPEG 字节被错误标注。 |
| A07 | P2 | Anthropic | 引用增量未累积到最终原生消息。 |
| A08 | P2 | Anthropic → Ollama | 纯文本块形式的工具结果无法通过 Ollama SDK。 |
| A09 | P2 | 其他思考格式 → Anthropic | 生成没有原生签名的 Anthropic thinking 块。 |
| A10 | P2 | 文档 URL → OpenAI Chat | 生成官方 Chat 不支持的 file.file_url。 |
| A11 | P2 | 五协议的流状态 | 重复终态和终态后的内容仍被接受，与项目流约束不符。 |

## A01：原生 Gemini Schema 没有转换为 JSON Schema

位置：[Gemini 工具 Encoder](../../src/my_llm/codec/gemini/encoding/tools/definitions.py#L18)。
它将 `parameters` 与 `parameters_json_schema` 作为同一种字典处理。
但官方定义分别使用 Schema 与 JSON Schema，前者有 `OBJECT` / `STRING`
等类型枚举，后者示例使用 `object` / `string`。
见 [FunctionDeclaration / Schema / Type](https://ai.google.dev/api/generate-content#FunctionDeclaration)
和 [OpenAI 函数参数定义](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。

复现使用实际 `types.Tool` / `FunctionDeclaration` / `Schema` 对象，
不是人为给 JSON Schema 填错大小写：

```python
types.Schema(type=types.Type.OBJECT,
             properties={"x": types.Schema(type=types.Type.STRING)})
```

转换到 Chat 后得到 `parameters.type="OBJECT"`，嵌套 x 仍为 `STRING`。
真实 OpenAI SDK 的本地 HTTP 请求也携带这个错误 Schema。
Anthropic / Ollama / Responses 的对应 JSON Schema 路径也未规范化。
现有样例只采用 `parameters_json_schema`，未覆盖 SDK 原生 Schema 分支。

建议：区分两类输入，递归转换已支持的 Schema 语义；不可表达的约束应明确处理。
SDK 类型检查只把参数 Schema 当作字典，不验证其内部是否为合法 JSON Schema。

## A02：原生选项被静默丢弃

位置：[通用选项校验](../../src/my_llm/codec/shared/options.py)、
[Anthropic REQUEST_FIELDS](../../src/my_llm/codec/anthropic/decoding/request/parameters.py#L11)、
[Gemini REQUEST_FIELDS](../../src/my_llm/codec/gemini/decoding/request/parameters.py#L11)。

合并 `extra_body` 后，只返回固定白名单里的键。实际复现：

```python
# Anthropic
extra_body={"tool_choice": {"type": "any"}}
# Gemini
extra_body={"tool_config": {"function_calling_config": {"mode": "ANY"}}}
```

生成请求中两者都消失，未报配置错误。Gemini `safety_settings`、
Anthropic `metadata` 也有相同行为；真实 SDK 的本地请求证实字段未发送。
官方支持 [Anthropic tool_choice](https://platform.claude.com/docs/en/api/messages/create)
和 [Gemini 函数调用模式](https://ai.google.dev/gemini-api/docs/generate-content/function-calling.md)。

原始实现改变了用户指定的生成行为，也与配置指南“原生参数通过 extra_body 配置”
的宽泛表述不一致。建议支持明确允许的原生选项，并对其余选项明确拒绝或说明；
继续保护核心协议字段和禁用 SDK 自动工具执行的约束。

## A03：OpenAI 拒绝内容处理不完整

位置：[Chat 流 Encoder](../../src/my_llm/codec/openai/chat/encoding/stream/chunks.py#L49)、
[Responses 流 Encoder](../../src/my_llm/codec/openai/responses/encoding/stream/chunks.py#L198)、
[Chat 内容渲染](../../src/my_llm/codec/openai/chat/decoding/messages/content.py)。

官方提供 [Chat delta.refusal](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)
与 [Responses refusal.delta / refusal.done](https://developers.openai.com/api/reference/resources/responses/streaming-events)。

Chat Encoder 只累积普通文本、可见思考和工具，不处理 `delta.refusal`。
输入仅含拒绝文本时，真实 SDK 路径最终返回：

```python
{"role": "assistant", "content": None}
```

原生拒绝文本也不出现在重建的 Chat 流中。

Responses 则将 refusal 增量作为 opaque 直接发布。仅含 refusal 的合法流，
重建后先发 `response.refusal.delta`，再发 `response.created`；
序号可出现 `3, 4, 2, 3, 4, 5`，引用的原始 item ID 与重建 ID 也不一致。
转换为 Chat 流时还混入 Responses 事件，未生成对应的 Chat refusal 增量。
最终 Chat 消息能以合法的 refusal 内容部分保留拒绝；该部分不判为结构错误。

建议：显式识别拒绝内容并接入块累积和目标渲染，保持拒绝语义与对应事件生命周期。
同协议 `MyCodec` 快照直通能够精确往返，不能证明 MyLLM 实时重建路径正确。

## A04：停止原因和停止序列转换有偏差

位置：[Anthropic 原因映射](../../src/my_llm/codec/anthropic/encoding/stream/chunks.py)、
[Anthropic 流 Decoder](../../src/my_llm/codec/anthropic/decoding/stream/chunks.py)、
[Gemini 流 Decoder](../../src/my_llm/codec/gemini/decoding/stream/chunks.py#L43)。

官方说明 [Anthropic 命中停止序列时同时返回原因与匹配序列](https://platform.claude.com/docs/en/api/messages/create)，
[Chat 的 stop 也包含命中停止序列](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)。

输入 `stop_reason="stop_sequence", stop_sequence="END"` 时：

- 转 Chat 得到 `finish_reason="stop_sequence"`，官方 `ChatCompletionChunk` 校验失败。
- 同 Anthropic 实时重建保留 stop_reason，却把匹配序列改为 `None`。

另将 Chat `content_filter` 转 Gemini，得到 `finish_reason="STOP"`。
这把过滤改写成自然停止；见 [Gemini FinishReason](https://ai.google.dev/api/generate-content#FinishReason)。
对于不能安全映射的状态，应保留可辨认的原因或明确报不支持，避免宣称正常结束。

## A05：已知来源字段进入不支持它们的目标结构

位置：[Gemini encode_part](../../src/my_llm/codec/gemini/encoding/messages/conversion.py#L43)、
[tool_block / chat_call](../../src/my_llm/codec/shared/conversion.py#L169)。

两组官方原生输入可复现：

```python
# Gemini 的普通文本 Part
{"role": "model", "parts": [{"text": "Hello", "thought": False}]}
# Ollama 的调用字典中的 function
{"index": 0, "name": "f", "arguments": {}}
```

前者转成 Chat 的文本内容部分后仍携带 `thought=False`，转 Anthropic / Responses
也会保留这个目标不支持的键。后者转 Chat 后产生 `tool_calls[].function.index`；
Chat 的 index 属于流调用的外层，function 只有名称和参数。
见 [Gemini Part](https://ai.google.dev/api/generate-content#Part)、
[Ollama 并行调用样例](https://docs.ollama.com/capabilities/tool-calling)、
[Chat 工具增量定义](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)。

Ollama 索引问题确认发生于原生字典历史 / 独立 Codec 路径。
本地 Ollama 0.6.3 的 SDK 响应模型会先忽略此索引，真实 SDK 上游流没有复现该泄漏；
不将这两条路径混为一谈。建议按来源消费已知字段，再按目标结构渲染。
此发现不要求删除任意未知扩展，也不改变同协议无损边界。

## A06：Ollama 图片 MIME 固定为 image/png

位置：[Ollama 消息 Encoder](../../src/my_llm/codec/ollama/encoding/messages/conversion.py#L44)。

官方 [Vision 样例](https://docs.ollama.com/capabilities/vision)使用 JPEG 文件的 base64，
Ollama 图片列表本身不提供统一 PNG 的保证。当前 Encoder 对每张图都填
`media_type="image/png"`，未检查字节。

用本地生成的完整 1×1 JPEG 复现，原始文件有 JPEG 头 `FF D8 FF`，
转 Chat 为 `data:image/png;base64,...`，转 Anthropic / Gemini 也声明 PNG。
这是字节与声明不一致，不是跨协议信息损失。建议识别支持格式，
无法识别时明确拒绝，避免编造 MIME。

## A07：Anthropic citations_delta 未进入最终消息

位置：[Anthropic 流 Encoder](../../src/my_llm/codec/anthropic/encoding/stream/chunks.py#L54)。

官方 [Citations streaming support](https://platform.claude.com/docs/en/build-with-claude/citations#streaming-support)
要求将每个 `citations_delta.citation` 追加到当前文本块的 `citations`。
当前没有对应操作，事件作为 opaque 透传，累积状态却未更新。

在同 Anthropic 输入输出下，逐块流中可以看到原始引用事件，
`generate(stream=False)` 和流的最终结果只剩文本，没有引用。
真实 Anthropic SDK 本地 transport 已复现。
建议把引用增量更新到对应块；这是一项当前未完整实现的官方文本能力。

## A08：Anthropic 文本块工具结果不能发给 Ollama

位置：[Ollama 结果渲染](../../src/my_llm/codec/ollama/decoding/messages/conversion.py#L52)、
[tool_output](../../src/my_llm/codec/shared/conversion.py#L307)。

官方允许 [Anthropic 工具结果采用内容块列表](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)。
只包含 `[{"type":"text","text":"OK"}]` 也会复现，无需图片或服务端工具。
`tool_output` 只对 dict 做 JSON 序列化，list 原样进入 Ollama 的字符串 `content`。
真实 Ollama SDK 在 HTTP 请求之前抛出 `ValidationError`。
对照 [Ollama 工具结果样例](https://docs.ollama.com/capabilities/tool-calling)。

建议对纯文本块结果生成字符串；媒体或不可表达结果另行明确处理。
不要求本库执行工具或跨协议无损保存所有结果媒体。

## A09：外来思考被渲染成缺少签名的 Anthropic 块

位置：[Anthropic 消息 Decoder](../../src/my_llm/codec/anthropic/decoding/messages/conversion.py#L30)。

Gemini `{"text":"Thinking","thought":True}` 转 Anthropic 后为
`{"type":"thinking","thinking":"Thinking"}`，缺少必填 signature。
官方 [ThinkingBlockParam](https://platform.claude.com/docs/en/api/messages/create)
要求原生签名，且原生思考块需要原样回放；官方 SDK 内容类型校验拒绝当前结果。
Chat / Ollama 可见思考转入这一目标也有相同结构问题。

项目已经声明协议专属签名不保证跨协议有效语义；因此本项不是要求伪造签名。
建议将这一限制落实为明确的不支持、排除或另有明示的文本转换策略，
避免生成貌似原生、实际缺少必填字段的请求。不能用空签名冒充服务端认可的签名。

## A10：文档 URL 被当成 Chat 支持的文件输入

位置：[chat_content 的 document 分支](../../src/my_llm/codec/shared/conversion.py#L226)。

原生 Anthropic URL document 转 Chat 得到：

```python
{"type": "file", "file": {"file_url": "https://example.com/a.pdf"}}
```

官方 [File inputs 支持表](https://developers.openai.com/api/docs/guides/file-inputs)
明确区分 Responses 与 Chat：Chat 文件输入不支持外部 URL。
真实 OpenAI SDK 未替本库下载文件或拒绝此键，本地捕获请求中仍有 `file_url`。
因此“交给 SDK 校验”不能保证请求合规。
建议明确报不支持；若另行设计下载或上传路径，应作为独立能力处理。

## A11：终态后的输入未受约束

位置：[StreamState.feed](../../src/my_llm/stream/_state.py#L100)、
[_reason](../../src/my_llm/stream/_state.py#L125)。

[本项目流状态约束](../design/stream-state.md)要求重复完成或完成后继续输入报错。
目前只有调用 `finish()` 后的 `finished` 会拦截后续输入；收到上游终态设置
`terminal=True` 后，feed 仍继续处理所有操作。

分别给五协议合法样例追加第二个原生终态，均被接受并正常结束；
Anthropic message_stop 后再追加文本 delta，文本也被纳入成功消息。
现有“重复完成”测试验证的是重复调用本地 finish，没有覆盖重复原生终态。

建议区分“收到终态”和“完成本地输出”，对终态后的内容与重复终态执行校验。
需要保留 [Chat 最后独立 usage 块](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
等合法尾部元数据，不宜把终态后的所有事件一律禁止。

## 修复验证

原先断言缺陷存在的 `tests/protocol_audit_probe.py` 已由普通契约回归替换，
不再保留要求错误行为继续存在的测试入口。完整命令、实际结果、补充发现和复核边界见
[修复验收记录](protocol-fixes-2026-10-17.md)。

保留原来的协议范围：不把全量音视频、全部服务端工具、未来新增事件、Interactions
或所有模型专属规则都视为已经实现。Schema 目录覆盖的接口字段范围也不等于运行时 Codec
支持全部原生能力。跨协议没有可靠表示的必要内容会明确拒绝；可安全保留的内容继续转换。
