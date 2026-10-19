# 五协议修复与代码复核

> 历史验收记录：本文描述对应审计时点。当前翻译字段契约以[翻译字段边界](../design/translation-boundaries.md)为准；文中未知字段透传、恢复记录及开放模型扩展结论已撤销。

日期：2026-10-17。对应[原始审计](protocol-audit-2026-10-17.md)的 A01—A11。
范围是 OpenAI Responses、OpenAI Chat Completions、Gemini GenerateContent、
Anthropic Messages、Ollama Chat；不包含 Interactions、Live 或 Ollama Generate。

## 修复和行为边界

A01—A11 已改写为普通回归测试，见
[test_audit_regressions.py](../../tests/contract/test_audit_regressions.py)。
原先要求缺陷继续出现的探针已删除。

| 要求 | 实现与直接证据 |
| --- | --- |
| Gemini Schema 类型与语义 | [encoding/content.py](../../src/my_llm/codec/gemini/encoding/content.py) 递归转换 OBJECT/STRING 等类型、nullable、anyOf、defs/ref 与数量约束；应用属性名不改写，parameters_json_schema 不重新解释。同协议快照还原原值。 |
| 原生选项不丢弃 | 各 Decoder 根据当前 SDK 支持的生成字段接收参数；未知选项明确拒绝，核心字段与 Gemini 自动工具执行开关继续受保护。[选项测试](../../tests/unit/test_native_options.py)覆盖错误发生在 client 创建前，[原生传输](../../tests/integration/test_native_sdk_transport.py)检查实际序列化结果。 |
| 拒绝、引用与终态 | refusal 作为独立内容块累积；引用使用通用列表追加操作；终态后的操作由源 Encoder 声明允许的尾部类型。普通与工具后延迟文本、五种原生重复终态、非法后续内容及合法用量尾块均有直接回归。 |
| 跨协议请求可表达性 | 各目标 Decoder 声明字段投影及降级策略。未知扩展不流入其他协议；文本工具结果转字符串，多模态结果在 Responses 可表达时保留。必要的不透明内容或无法保留的约束明确拒绝。 |
| 图片与文件 | Ollama 图片按字节辨认 MIME；未知格式不假定 PNG。Chat 不生成 file.file_url；无法原生读取的 URL 文档以文本引用保留，无隐式下载。 |
| 签名与历史 | 同协议签名、加密数据与来源引用保留。外来思考不伪造签名；Gemini 外来函数历史使用官方迁移标记；官方 Chat 默认不添加 reasoning_content，兼容服务显式开启。 |

兼容策略是避免已知不合法或已改变必要约束的请求；不能只凭字段转换保证所有模型能力、
服务端历史上下文、鉴权或配额都满足。没有可靠表达的必要内容抛出
UnsupportedCapabilityError；无对应语义的附加扩展丢弃。
具体配置与兼容服务示例见[配置指南](../reference/configuration.md#跨协议请求的处理边界)。

## 补充代码复核发现

- **最新 Responses 必填字段不能由旧 SDK 充当校验依据。** 根据本次固定的
  [官方来源快照](../reference/schema/sources.json)，补齐 access_programs、error、incomplete_details、
  instructions、metadata、temperature、top_p 与 output_text.logprobs。
  缺少来源值时使用协议允许的 null / 空值；原生报告的设置完整保留。
- **原生 Responses item 身份必须贯穿流。** 保留真实 item ID、output_index、
  content_index / summary_index；一个原生 reasoning 的多个 summary 部分仍属于同一个 item，
  不将加密数据复制成多个生成 ID 的 reasoning item。最终 output 与 item.done 一一对应。
  客户端工具仍等参数完整及上游 EOF 后发布；opaque 原生进度保留身份并重新安排流序号。
- **SDK 可静默丢弃合法约束。** 当前 Ollama 0.6.3 的工具参数类型会丢弃部分 JSON Schema
  关键字，例如属性上的 minItems。外来 Schema 进入 SDK 前检查是否完整保留；
  不通过丢弃约束或手写 HTTP 绕过这个边界。
- **输出不能借用请求元数据。** Responses envelope 与原生项目按输出所有权复制，
  修改已发布帧不会污染后续快照或请求状态。
- **媒体内容必须符合消息角色。** Responses 请求中的外来 assistant 图片历史使用
  input message 的 input_image；Chat 的 assistant URL 图片以文本引用表达。
  无法表示的二进制回复在本地拒绝，不生成非法 assistant 图片块。
- **单消息结果不能静默丢弃额外候选。** Chat 的 n 和 Gemini 的 candidate_count
  仅支持缺省、None 或 1；其他值在请求前拒绝。
- **生成参数不能破坏原生配置对象。** Gemini ThinkingConfig 和 Ollama Options
  先按公开 SDK 数据结构读取再合并，保留显式 include_thoughts=False，且不修改传入对象。
- **最终结果在成功消费后提交。** 完整输出帧已消费、目标转换成功且资源正常关闭后，
  才设置 stream.result；目标终态不可表达或消费者提前关闭时不发布成功结果。
- **Ollama SDK 不提供重试次数设置。** 正数 max_retries 在其 Decoder 校验阶段拒绝，不静默忽略；None / omit / 0 保留原生行为。
- **Ollama 强度字符串由模型定义。** 保留 profile 提供的原生映射，不把协议收窄成全局 low/medium/high 枚举。
- **Chat 的兼容字段应显式启用。** 官方默认采用 reasoning_effort；
  reasoning_content 回放默认 False。Ollama 原生 thinking 的回放默认 True，
  各协议默认值由各自 Decoder 声明。

Gemini 迁移标记只用于外来工具历史，依据
[GenerateContent Thought signatures FAQ](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures#faqs)。
使用 `skip_thought_signature_validator` 字符串，并通过真实 GenAI SDK 的 HTTP body
检查其保持字面值；不误发字符串的二次 base64 编码，不修改原生返回的签名位置。

## 协议差异归属

| 行为 | 当前唯一责任位置 |
| --- | --- |
| 全列表原生格式识别 | BaseEncoder.matches 统一遍历；各 `encoding/encoder.py` 实现 match_item；BaseCodec 只委托 Encoder，MyCodec 按注册顺序选择。 |
| 原生消息、内容、工具、流片段、消费字段与工具 Schema | 各 `encoding/`。 |
| 原生计数归一化 / 目标用量结构 | 各 Encoder / Decoder 的 usage；共享层只合并累计快照及计算通用计数组。 |
| 目标字段投影与媒体/思考/工具结果策略 | 各 `decoding/decoder.py` 和 `decoding/content.py`。 |
| 默认字段路径、推理绑定、用量模式 | 各 Decoder 的 profile_defaults、request_parameters、usage_modes；profile resolver 获取声明后通用合并校验。 |
| 请求体字段、system 指令与服务端历史能力 | 各 Decoder 的 assemble_request 和 allows_external_calls。 |
| SDK、端点、HTTP 单位、客户端 headers、关闭方式 | 各协议根目录的 Transport；由 BaseCodec 统一组合，共享 owned_stream 只管理响应资源栈。 |
| Responses 列表结果与其他消息结果 | 对应 Decoder.result。 |
| 块身份、片段、引用追加、尾部许可、参数完整性与 EOF | 请求自己的 StreamState；源引用固定为 BlockRef，不按厂商名分支。 |

删除全局 codec/request.py、openai/_parameters.py、utils/native.py、utils/preservation.py，
以及共享 helper 中的厂商分支。MyLLM 不再导入原生 SDK client 或选择端点。
Encoder / Decoder 类移到 encoder.py / decoder.py；所有包初始化文件仅导出接口。

公共机制复用同一套字典保留与投影、选项解析、用量快照合并、历史关联和资源管理；
没有保留旧入口转发或两条兼容实现。协议的字段集合和 native / foreign 语义判断
保留在其责任目录内，未用共享层的协议枚举 switch 取代旧 switch。

[边界回归](../../tests/contract/test_codec_boundaries.py)检查：

- 运行时代码的厂商条件与 SDK client 导入位于对应 encoding / decoding。
- 一个原生实现不导入另一个原生实现。
- 包初始化文件不定义类、函数或执行构造调用。

代码复核还逐一检查了请求组装、profile 默认值、格式识别、内容投影、用量换算、
原生引用、opaque 处理、终态尾部和清理路径。结构扫描与测试共同作为证据；
没有把一次搜索无命中当作全部协议语义符合的证明。

## 验证

官方版本与派生规则见 [sources.json](../reference/schema/sources.json)。
运行时 SDK 保持 openai 2.54.0、anthropic 0.125.0、google-genai 1.75.0、ollama 0.6.3，
没有为导入最新 Schema 而升级运行时依赖。新增 jsonschema 4.26.0 开发校验依赖及锁文件。

[test_protocol_conformance.py](../../tests/contract/test_protocol_conformance.py) 的独立手写样例
继续验证消息、工具、图片及普通/批量流的 25 对组合；另验证五来源到五目标的思考流、
外来消息与声明扩展的投影、拒绝、引用、停止原因、真实 ID、快照及元数据所有权。

[Schema 校验辅助](../../tests/helpers/schema.py)直接读取固定的官方结构，
检查真实 SDK 序列化的请求体和重建的流帧；Gemini 使用公开 model_dump(by_alias=True)
输出 REST 字段。SDK 递归类型校验仍拒绝被忽略的字段，并消费延迟校验的 Iterable。
旧 SDK 不认识的 Responses 字段由完整的当前官方 Response Schema 校验。
缺必填字段的反例会失败，避免把生成代码自身用作唯一的期望来源。

50 个 SDK 生成组合覆盖五上游 × 五 I/O × 流式/收集；并发专项继续覆盖 30 个场景、
230 次本地请求，以及取消、异常、提前关闭和后续复用。

```bash
uv sync --locked --group dev
.venv/bin/python -m pytest -q
.venv/bin/python docs/reference/schema/verify.py
.venv/bin/python -m ruff check src tests examples
.venv/bin/python -m ruff format --check src tests examples
.venv/bin/python examples/local_demo.py
git diff --check
```

最终验证：pytest **913 passed**；24 个 Schema、59 个模板、19 个流文件的 90 帧
全部通过校验；Ruff 检查及 176 个 Python 文件的格式检查通过；
五协议离线示例与 git diff --check 通过。
没有调用远端模型；不把本地结构和生命周期验证称为真实服务端全模型测试。
