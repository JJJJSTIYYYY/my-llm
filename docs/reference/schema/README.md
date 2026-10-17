# 五种原生协议的请求与返回模板

更新快照：**2026-10-17**。范围仅包含以下五种 HTTP 生成接口：

| 接口 | 目录 | 请求路径 |
| --- | --- | --- |
| OpenAI Responses | [openai_responses](openai_responses/README.md) | `POST /v1/responses` |
| OpenAI Chat Completions | [openai_chat](openai_chat/README.md) | `POST /v1/chat/completions` |
| Gemini GenerateContent | [gemini](gemini/README.md) | `POST /v1beta/models/{model}:generateContent` / `:streamGenerateContent?alt=sse` |
| Anthropic Messages | [anthropic](anthropic/README.md) | `POST /v1/messages` |
| Ollama Chat | [ollama_chat](ollama_chat/README.md) | `POST /api/chat` |

这里使用各家原生字段，不依赖本项目 Codec 的当前支持范围，
也不把兼容服务的扩展当作对应官方协议。

## 文件怎么用

每个目录均提供：

- `http.json`：URL、鉴权 header、流的 Content-Type 和终止方式。
- `request.json`、`response.json`、`error.json`：普通请求、完整响应与 HTTP 错误模板。
- `templates.json`：按场景拆分的独立模板，包含工具、媒体、思考、结构化输出等适用变体。
- `request.schema.json`、`response.schema.json`、`stream-event.schema.json`、`error.schema.json`：
  完整字段结构及全部可达嵌套定义，采用 JSON Schema 2020-12 和本地 `$defs` 引用。
- `stream*.sse` 或 `stream*.ndjson`：原始流帧示例，包括普通文本、工具和适用的拒绝、截断、失败、引用、错误分支。
- `README.md`：顶层字段表、流事件目录、互斥关系、模型条件和来源链接。

“完整”指在本次固定来源快照中，以上接口的请求、返回、错误和流结构包含的全部可达字段与类型分支。
它不表示一个 JSON 能同时启用所有互斥选项，也不表示每个模型都支持所有字段。
应先选择场景，再从 Schema 查阅可选字段；不要把 `templates.json` 的多个场景机械合并。

尖括号值如 `<MODEL_ID>`、`<OPENAI_API_KEY>` 是占位符。
响应 ID、计数和时间是示例数据；密文、签名和服务端历史引用必须来自真实原始响应，
即使占位符可以通过结构校验，也不能据此发送伪造的签名或历史 ID。

## 快照来源

| 来源 | 本次版本 |
| --- | --- |
| [OpenAI 官方 OpenAPI](https://github.com/openai/openai-openapi) | 提交 `d156763a484eb247e116e0ea345e16344b0049ba`，2026-10-17 |
| [Anthropic 官方 SDK 类型](https://github.com/anthropics/anthropic-sdk-python/releases/tag/v1.15.0) | 本次查询的最新 PyPI 发布版 `1.15.0`，另核对 Messages / Streaming 在线文档 |
| [Gemini 官方 Discovery](https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta) | `v1beta`，revision `20261015` |
| [Ollama 官方 OpenAPI](https://github.com/ollama/ollama/blob/2476d5ead197abe04629651028f83d48dc75d2b2/docs/openapi.yaml) | 提交 `2476d5ead197abe04629651028f83d48dc75d2b2`，2026-10-16 |
| [Ollama 官方公开服务端类型](https://github.com/ollama/ollama/blob/7a5ae12ead68342df402177cfabf4e7fc95aaca5/api/types.go) | 补充 OpenAPI 漏列字段，逐项记录来源 |

固定 URL、提交日期、源文件 SHA-256、SDK wheel 哈希、编译与补充规则见
[sources.json](sources.json)；机器可读目录见 [catalog.json](catalog.json)。
许可与署名见 [NOTICE.md](NOTICE.md)。

“最新”是上述日期查询的快照，不是永久跟随上游。
本项目当前安装的 SDK 可能落后于这些定义；本目录不会因此升级运行时依赖，
也不将最新协议定义限制为项目旧 SDK 恰好能够解析的子集。

## 保留的来源差异

1. **OpenAI Responses 的输入联合存在重叠。** 官方 OpenAPI 的 `InputItem.oneOf`
   让文档中的消息同时命中 EasyInputMessage 和 Item/InputMessage，严格 JSON Schema 校验会失败。
   [上游结构](openai_responses/request.upstream.schema.json)保留原值；
   [本地可用结构](openai_responses/request.schema.json)仅将这一处改为 `anyOf`，并带有变更注记。
2. **Anthropic 的 Schema 是派生结果。** 本次未找到公开可下载的官方 OpenAPI；
   从最新官方 SDK 导出结构，移除实际发送到 header 的参数，补入文档中的流控制事件。
   稳定与 beta 类型分别保存，不能省略对应 beta header 或模型限制。
3. **Gemini Discovery 不是 JSON Schema。** 本地转换引用和基础类型，
   将明确写为 `Required.` 的字段标为必需，并把 model 保留在 URL。
   Part 数据互斥、模型限制和其他组合规则仍需按文档判断。
4. **Ollama OpenAPI 有漏项。** 消息思考、工具结果名称、调用索引、终帧统计等
   已按官方文档和公开服务端类型补充；`sources.json` 中列出每个补充位置。

来源文档和 SDK 的要求不能无条件等同于服务器全部校验。
这里没有把这些派生 Schema 重新命名为“官方发布的 JSON Schema”，
也没有自行设置全局 `additionalProperties=false` 来否认未来扩展或上游允许的开放对象。

## 校验

校验脚本使用 `jsonschema`，不导入本项目转换器，不请求模型服务。
无需更改项目依赖，可以在临时环境运行：

```bash
uv run --no-project --with jsonschema==4.26.0 python docs/reference/schema/verify.py
```

它检查所有 JSON 与 Schema、引用闭包、普通模板和场景模板、流帧结构，以及
Responses 序号与项目 ID、Anthropic 块生命周期、Chat 用量尾块、NDJSON 终态等。
`*.upstream.schema.json` 检查结构与引用，不将已知重叠联合用作示例的通过依据。

结构校验不执行工具、不验证真实签名、不推断 token 计数是否来自实际分词，
也不验证模型能力、配额或服务端跨轮历史。新的未知事件需要按官方版本规则处理。

本次运行的校验结果和产物哈希记录于 [verification.json](verification.json)。
