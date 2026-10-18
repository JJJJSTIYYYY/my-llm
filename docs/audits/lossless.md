# 字段保留与轮次恢复

2026-10-18 更新。转换保留信息，Transport 在实际请求前过滤不支持的字段。
此前“仅保证同协议与中间态无损”的范围已扩展到未修改转换结果的原生轮次恢复。

## 保留规则

- Encoder 将已识别字段翻译为中间态，未知字段保留原层级、嵌套字典与列表位置。
- Decoder 在目标结构可共存的路径保留未知扩展；转换字段发生冲突时，翻译后的值优先。
- 合并、拆分或省略内容导致路径无法一一对应时，原始字段保存在恢复记录中，不按下标挂到其他消息或内容块。
- 当路径形状冲突，或需要保留签名、密文及其他目标协议无法表达的信息时，结果携带 `__my_llm_codec_replay__` 恢复记录。
- 工具参数、工具结果对象和 JSON Schema 是完整应用值，不把其中的 type、role 等键当作协议字段解释或删除。
- 全部操作使用副本，不修改输入、配置或调用者保存的历史。

原路径示例：

```text
源：    a.b = v1，a.c = v2
中间态：d   = v1，a.c = v2
```

原字段与转换值重名时，中间态用 `__my_llm_codec_translated_fields__` 区分；
字典读取、工具关联、流累积和 Decoder 均以转换值为准。
目标消息无法同时容纳原字段时，原值保存在恢复记录内，不覆盖目标字段。

## 未修改结果的恢复

恢复记录包含原始中间态、目标翻译结果和对应输出快照，以普通字典随结果保存。
重新编码时，比较整个转换分组的当前值与输出快照：

- 完全一致：恢复原始中间态，保留原生块、字段、顺序和工具 ID。
- 分组被编辑、重排、截断或恢复记录被移除：按当前目标协议数据重新翻译，不套用旧快照。
- 分组后新增消息或工具结果：旧分组仍可恢复，新增数据正常翻译；无 ID 协议的工具结果重新关联原始调用 ID。

因此 Anthropic → Chat → Anthropic 可以恢复 thinking.signature 和 redacted_thinking.data；
Responses → Chat → Responses 可以恢复 reasoning.encrypted_content 与完整 reasoning item。
恢复记录支持 JSON 保存、读取和新实例处理，不依赖全局缓存或 SDK 客户端。

要保留这个能力，应保存完整消息字典，包括恢复记录。流式调用保存 `stream.result`；
仅保存拼接文本会丢失原生信息。完整事件列表的 MyCodec.convert(chunks=...) 同样携带恢复数据；
实时逐事件翻译不积累完整历史，以最终结果作为多轮请求的回放依据。

## 请求过滤

BaseCodec 组合翻译后，由 BaseTransport.request 根据公开 SDK 的消息、工具类型递归过滤协议结构。
仅过滤请求副本，原消息仍保留扩展与恢复记录；它们不会进入实际 SDK 请求。
兼容服务主动配置的 extra_body、extra_headers 和思考回放按 Profile 策略处理。

加密数据不做跨厂商解密或改签。请求原协议时可以原样回放，目标为其他协议时只翻译可见内容。
恢复数据不代替服务端对模型、完整上下文或签名有效性的检查；无可靠语义表示的必要内容仍可能无法转换或请求。

## 保留命名空间

`__my_llm_codec_source__`、`__my_llm_codec_translated_fields__`、
`__my_llm_codec_function_definitions__`、`__my_llm_codec_opaque_message__`、
`__my_llm_codec_replay__` 是库保留键，统一定义在 [conversion.py](../../src/my_llm/codec/shared/conversion.py)。
原生输入不应使用这些键承载应用扩展。普通应用字典中的其他键不会被当作内部控制结构。

## 验证与依据

- [原路径保留契约](../../tests/contract/test_intermediate_preservation.py)：直接检查层级、索引、冲突优先级和输入不变。
- [恢复与过滤契约](../../tests/contract/test_replay_preservation.py)：五协议消息、工具、完整流往返，JSON 保存、修改失效、密文和工具关联。
- [真实 SDK 两轮回放](../../tests/integration/test_replay_transport.py)：本地 HTTP 响应驱动 Anthropic SDK，检查实际请求中的签名、密文及扩展过滤。
- [协议符合性](../../tests/contract/test_protocol_conformance.py)：独立样例校验翻译后的协议字段；保留的外来扩展不被当作目标 API 字段。

[Anthropic 思考与签名](https://platform.claude.com/docs/en/build-with-claude/thinking#thinking-encryption)要求完整回放思考块；
[OpenAI Responses reasoning](https://developers.openai.com/api/docs/guides/reasoning)支持保留加密 reasoning 供后续请求使用。
测试使用本地构造的签名与密文，不验证真实服务端的密码学有效性。
