# 协议转换与公开接口审计

审计日期：2026-11-22。源码基准：`a0d351ebb29689794c1e6dc7865c9b9427949e17`。
本报告记录审查现象及最终处理决定；审计期间只调整文档及其索引，没有修改实现或测试。
五项观察均已按业务范围决定不修复，不作为后续修复清单。

## 结论

记录五项可复现现象，按用户确认的业务范围均不修复：

| 编号 | 业务归类 | 处理状态 | 观察项 | 触发条件 |
| --- | --- | --- | --- | --- |
| A01 | 低影响保真差异 | 不修复（当前决定） | 原生空 item 经 metadata 终态回填，导致输出顺序改变 | 有效 Responses 流中，空 item 位于有内容的 item 前 |
| A02 | 展示分段差异 | 不修复（已验证回传一致） | reasoning summary 的流内 part 内容与完成 item 内容矛盾 | 有效 Responses reasoning item 含多个 summary part |
| A03 | 保障范围外 | 不修复 | 公开 finish 的内容副本被外部改成不一致 | 调用方修改或构造不一致的 messages / output_blocks |
| A04 | 保障范围外 | 不修复 | 公开流被外部删减后，目标生命周期不完整 | 调用方删除 block_start，或仅提供 finish |
| A05 | 业务场景外 | 不修复 | 错误类型或非法值输入的处理不一致 | 非字符串 text / function name、非法 role 等输入 |

A01、A02 不需要修改输入或调用内部接口，直接由有效原生流触发。
A01 当前没有已证实的文本或工具参数损坏；A02 已验证正式历史回传保持原样。
A03、A04 需要外部主动破坏数据一致性或事件序列，不承诺破坏后的系统行为。
A05 的错误类型和非法值不属于本次确认的业务场景，不为此增加防御性校验。
保留事实、样本和已验证范围，不将这些记录列为修复待办。

实现中，内容、项边界和顺序分布于消息快照、索引块、原生 metadata 及 feedback_item。
这作为结构观察保留，不据此推定业务错误或要求重构。本轮没有确认仍需在约定业务范围内修复的事项。

## 范围与审查口径

业务边界为 MyLLM.generate，以及 MyCodec.encode / decode / convert。
内部组件不承担可独立使用的业务契约，不因其缺少重复校验而判为问题。
检查三种内置协议：openai_chat、openai_responses、anthropic。

保障范围不包括外部主动破坏中间层快照或事件序列后的行为，也不因 A05 中业务不会提供的
错误类型／非法值而要求补充校验。这不改变已支持的正常业务数据和正式回传内容应保持正确的要求。

按本次会话确认的原则审查：

1. 协议转换通过中间层协作；协议差异应有明确所有者。
2. 公共基类负责契约、通用约束和流程组装，不承载具体协议算法。
3. 不为某个协议的兼容分支给公开中间层消息增加控制字段。
4. Encoder 丢弃中间层未支持的原生字段；Decoder 对目标不能表达的已支持语义登记损失。
5. 修复拥有该数据的契约和流程，不以下游猜测、补回或重复转换掩盖上游信息缺失。

以下已接受的取舍不列为缺陷：

- 内部按 Responses 来源对 assistant items 分组，并共享 reasoning feedback 决策。
- 未指定协议时，guess 使用字段排除条件辅助识别。
- 既定的工具关联规范化，包括重复调用处理、补空结果及孤立结果处理。
- usage 等辅助信息的容错，以及显式启用的语义损失 hook。
- 通过声明式能力检查处理公共生成参数，不因参数目前仅被某协议支持就判为协议分支。

ItemIdentity、protocol、feedback_item 不仅凭名称判为补丁。审查重点是其实际职责：
是否是明确建模的身份／原样回放语义，还是用来补偿主转换路径已经丢失的内容和边界。
本报告不建议为已接受的分组取舍新增公共兼容字段。

## A01：空 item 的终态补回改变原生顺序

**处理决定：当前不修复。** 已证明的是空 item 的位置改变，没有证明文本或工具参数损坏，
也没有真实供应商触发频率的证据。业务只读取可见文本时，该复现不改变答案。
保留记录和样本，不据此扩展公开模型或引入新的流缓冲／重排机制。

### 复现与影响

向 MyCodec.convert 传入含完整 item / text 生命周期的 Responses 流，原生最终 output 为：

```text
[message(id=empty_before, content=[]), message(id=text_after, content=["hello"])]
```

以 openai_responses 作为输入和输出协议，实际得到：

```text
source: ['empty_before', 'text_after']
target: ['text_after', 'empty_before']
added:  [(0, 'text_after'), (1, 'empty_before')]
done:   [(0, 'text_after'), (1, 'empty_before')]
```

输入逐事件通过安装的官方 ResponseStreamEvent Schema 校验。
输出内部索引与自身最终数组一致，但已经改变源 output 的顺序；因此仅检查目标索引自洽不足以发现问题。

### 根因

- [Responses Encoder 的 _item_metadata](../../src/my_llm/codec/openai/responses/encoding/encoder.py#L476)
  增加 empty 标记，经公开 MessageEvent.metadata 的 response_items 暴露。
- [StreamState.finish](../../src/my_llm/runtime/state.py#L457) 主要按内容块生成终态。
  空 item 没有块，无法在这条序列中保有自身位置。
- [Responses Decoder 的 _address](../../src/my_llm/codec/openai/responses/decoding/decoder.py#L695)
  按内容首次出现分配目标索引；有文本的 item 先占据索引 0。
- [_completed_output](../../src/my_llm/codec/openai/responses/decoding/decoder.py#L660)
  到完成时才根据 metadata 的 empty 恢复空 item，并给它分配后续索引；finish 分支再补出 added。

empty 有明确用途：区分原生真正为空的 item，与原本有内容但投影／能力过滤后没有剩余块的 item，
避免错误重建后者。不能仅因这个标记存在就认定它是补丁。
两个原生空 message 编码后，finish.messages 只有一个无内容的 assistant，output_blocks 为空；
原有 item 数量与身份保存在 metadata。已确认的缺陷是目标位置按内容块首次出现分配，
导致空 item 的位置较晚确定，而非该标记无法判断是否保留空 item。

### 修复要求与测试缺口

若后续重新处理，应先检查 Responses codec 内的 item 生命周期与目标索引分配，无需新增公开消息字段。
反例中的上游 item 本来按序完成，不需要为此引入乱序处理。
不应仅排序最终数组：实时阶段一旦发布错误索引，终态重新排序也无法修正已发布事件。

[现有空 item 测试](../../tests/contract/test_native_stream_projection.py#L86) 将结果转换为字典比较，
generate 场景使用集合比较，均忽略顺序。验收需同时断言源顺序、目标生命周期、最终数组，
并覆盖空 item 位于开头、中间和末尾，以及过滤其他内容后的情况。

## A02：多段 summary 的流事件与终态内容矛盾

**处理决定：当前不修复。** 按用户明确标准，仅展示不一致不修复；若给上游接口回传的内容不一致，
可能影响缓存输入，则需要修复。本轮进一步检查 SDK 实际发出的下一轮 HTTP 请求，
32 个场景中的原生 reasoning item 均完整保留。现有证据将 A02 限定为流内展示分段差异，
不将其认定为原生历史回传或缓存输入被改变。

### 复现与影响

有效原生 reasoning item 的 summary 为两段 a、b，各自有 summary_index 0、1 的
added / delta / text.done / part.done，随后 item.done 和 response.completed。
同协议转换实际得到：

```text
reasoning_summary_part.done: summary_index=0, text='a\nb'
response.output_item.done.summary: ['a', 'b']
response.completed.output[0].summary: ['a', 'b']
```

目标 part.done 声明索引 0 的内容是 a 换行 b，而完成 item 的同一位置是 a；索引 1 没有对应的目标 part 生命周期。
按增量构建结果与直接读取终态的消费者因此得到不同结构和内容。

### 根因

- [Encoder 处理第二段 summary](../../src/my_llm/codec/openai/responses/encoding/encoder.py#L328)
  时插入换行；[后续 reasoning delta](../../src/my_llm/codec/openai/responses/encoding/encoder.py#L397)
  统一使用 part 0，公开可见 reasoning 被压平。
- Decoder 用压平的 reasoning 构造流内 delta / part.done。
- [Decoder._output_item](../../src/my_llm/codec/openai/responses/decoding/decoder.py#L609)
  在完成时从 feedback_item 原样恢复多段原生 summary。

合并可见摘要本身是既有取舍，不能仅因形状不同就判错。缺陷在于同一个目标流中，
已经关闭的 part 与完成 item 的对应位置互相矛盾。原生回放与可见流投影需要有一致的输出契约。

### 展示契约的测试缺口

展示流和完成 item 的分段确实不同，但当前处理决定是不为该展示差异修改实现。
尤其不应为统一展示而在 completed 分支改写原生 summary，这反而会改变已验证正确的回传内容。

[既有回归测试](../../tests/contract/test_audit_regressions.py#L533) 验证 item.done 与 completed 一致，
还明确预期所有 delta 都落在 summary_index 0，却未比较 part.done 与完成 item 的对应内容。
验收需覆盖多段、空段、不同分片、JSON 往返，并验证三个层次之间的对应关系。

### 下一轮 HTTP 回传与缓存输入验证

本轮从原生多段 reasoning 加工具调用响应开始，使用公开 generate 完成第一轮，
再将返回的正式结果连同工具结果交给同一 MyLLM 的下一轮 generate。
SDK 通过现有 LocalSDK 接入本地 HTTP / SSE；直接读取 SDK 发出的 HTTP request.content，
没有通过内部 prepare 或重新编码结果来替代最终 wire 验证。

矩阵共 32 个场景：

- 输入／输出格式为 Responses 或 middleware。
- 第一轮采用流式消费后读取 stream.result，或直接收集式返回。
- 本项目 enable_message_cache 开启或关闭；开启时按增量批次提供历史，关闭时提供完整历史。
- should_feedback_reasoning_item 为 always 或 when_tool_calls；样本包含实际工具调用与对应结果。
- summary 分别为两段普通文本，或三段包含段内换行、空段及 Unicode 的文本；原生 delta 按字符分片。

结果：32/32 下一轮 HTTP input 中的 reasoning item 与原生项完全相等，
规范化 JSON 指纹也一致。比较包含 summary 数量与顺序、每段文本、原生 item ID、status 和 encrypted_content。
16 个开启消息缓存的场景还重复提交相同增量批次，实际 HTTP input 与上次完全一致，未重复追加或改写摘要。

因此，使用 MyLLM 返回的正式结果作为历史时，没有发现 A02 会因摘要合并改变发给上游的 reasoning，
不能据此推断供应商缓存失效。此验证不测量真实服务的 cached tokens 或缓存命中率，
只证明所列场景的回传内容一致；也不保证其他参数或历史变化不会影响供应商缓存。

回传时必须区分可见 reasoning 文本和正式历史：middleware 使用返回消息中保留的 feedback_item；
Responses 使用 stream.result、收集式结果或完成事件中的原生 item。
若调用方自行把展示 delta 重新拼成新的原生 summary，那条自定义重建路径不在上述保真结论内。

### 业务可达性与证据等级

此问题不要求上游乱序或调用方修改数据。触发条件是：同一个原生 reasoning item
返回多段 summary，并将输出交付为 Responses 原生事件流。
多个 reasoning items 各只有一段 summary，不等同于这个触发条件。

[官方推理指南](https://developers.openai.com/api/docs/guides/reasoning?api-mode=responses)
规定显式开启摘要后，通过 reasoning item 的 summary 数组返回摘要；
[官方流事件定义](https://developers.openai.com/api/reference/resources/responses/streaming-events)
通过 summary_index 定位每个摘要 part。它是协议允许的路径，但文档和单事件 Schema
不能证明某个供应商实际返回多段的频率。
项目 Responses 默认推理配置包含 summary="detailed"；是否实际发送该配置仍取决于是否启用推理及 Profile 覆盖。

本轮用安装的真实 OpenAI SDK 消费本地 HTTP / SSE 响应，再调用公开 MyLLM.generate，已复现：

| 路径 | 两段 summary 样本的结果 |
| --- | --- |
| generate，stdio=Responses，stream=True | part.done 为 a 换行 b，终态 summary 为两段；A02 可达 |
| 同一次调用的 stream.result | 保留原生两段，样本中与原 item 完全一致 |
| generate，stdio=Responses，收集式返回 | 保留原生两段，样本中与原 item 完全一致 |
| MyCodec 的完整流 encode → decode | typed / wire 公开事件均复现；JSON 往返需纳入修复验收 |
| MyCodec 的消息转换 | 同协议原生 reasoning item 完整保留，没有这项流内矛盾 |
| generate，stdio=middleware | 可见 reasoning 为合并文本，feedback_item 完整保留；属于既有双用途契约 |

本地 HTTP 模拟只替换 SDK 的网络响应，没有重写 codec、累积器或业务返回转换。
这证明问题经过业务入口可以触发，不代表已经在真实服务响应中观察到。

此前 SenseNova 报告中的 Responses 推理回放用例通过，但仅保存原生项与最终结果／下轮请求的
指纹和相等性，没有 summary 段数或逐 part 生命周期证据。
其 _source_reasoning 直接读取 response.completed.output，因此这些结果不能排除 A02，
也不能证明此前真实请求触发过 A02。本轮没有重新请求供应商。

直接受影响的是按 part 生命周期组装或展示 Responses 推理摘要的消费者；
仅拼接可见文本、读取最终结果或回放原生 item 的路径，在上述样本中没有丢失推理文本／原生项。

### 若未来重新处理：影响范围与约束

以下为此前评估的改动范围，当前不执行。只有出现正式回传内容变化的反例，
或业务明确要求保留展示段边界时，才重新评估处理决定。

优先目标是让 Responses 事件流遵守最终原生 item 的 part 边界，并保留原样回放；
不应仅将最终 summary 改成一个合并段来迎合当前流，因为这会改变原生项和后续回放内容。

需要评估的实现范围是 Responses Encoder、Decoder，以及两者之间对摘要段边界的内部传递。
当前 Encoder 已压平 summary_index；修复不能承诺只改 Decoder 的一个完成分支。
离线完整流可读取终态 feedback_item 中的原始 summary；实时渲染不能假设未来的终态已经到达，
必须保留当时已知的段边界，或明确接受缓冲带来的实时性变化。
不能按换行猜分段，原生单个 part 自身也允许包含换行。

没有必要先给公开 ReasoningContent 增加 Responses 专用字段。
如果内部段边界能在协议上下文中保留，公开合并文本与原生回放项的现有数据结构可以继续使用；
但完整公开流经新 MyCodec 解码的结果仍必须与实时 generate 一致，不能仅修实时内部状态。
若考虑将一个公开 reasoning 块改成多个块，则会改变公开内容形状及其他目标协议的投影，属于更大范围的契约变更。

回归范围包括：多段／空段／段内换行、任意 delta 分片、typed / JSON 独立解码、
Responses 实时与收集结果、原生 item 和下一轮 wire 回放，以及 Chat / Anthropic / middleware
继续按既有规则处理可见 reasoning。现有明确预期 summary_index 全为 0 的测试及流设计文档需要同步修正。
目前没有证据要求修改请求参数、工具关联、缓存或客户端生命周期。

## A03：公开 finish 的双份内容缺少权威来源

**处理决定：不修复。** 本复现依赖外部主动破坏快照之间的一致性。
不承诺这类破坏后的系统行为，不新增快照同步、矛盾检测或恢复分支。

### 复现与影响

从公开 encode 得到正常流，修改 finish.messages 中的文本，再分别翻译消息与完整流：

```text
decode(messages=finish.messages): EDITED_PUBLIC_MESSAGE
decode(chunks=stream):           原文本 Hello
异常:                           无
```

反向修改 output_blocks 的内容同样被接受。这里需要调用方修改或构造不一致的公开输入；
该差异发生于外部破坏数据一致性之后，属于本次明确排除的保障范围。

### 实现原因与处理边界

[MessageEvent](../../src/my_llm/middleware/stream.py#L129) 的 messages 和 output_blocks
分别持有可变内容模型，[from_validated_wire](../../src/my_llm/middleware/stream.py#L152) 分别构造它们。
[validate_complete_stream](../../src/my_llm/middleware/validation.py#L440) 仅检查局部结构、末尾唯一 finish 和必填字段。
Responses 完成输出主要读取 output_blocks，消息翻译读取 messages，因此结果取决于所走入口。

这解释了破坏后不同入口的结果差异，不构成当前保障范围内的修复要求。
此前关于新增一致性校验和破坏性输入验收的建议撤回。

## A04：完整流入口未验证块生命周期

**处理决定：不修复。** 本复现依赖外部主动删除正常事件。
不承诺事件序列被破坏后的系统行为，不新增完整生命周期校验或目标事件补齐逻辑。

### 复现与影响

正常 Chat 流经公开 encode 后得到：

```text
start → block_start → delta → block_stop → finish
```

删除 block_start，保留原有且互相一致的终态快照，再调用公开 decode：

| 目标 | 实际结果 |
| --- | --- |
| Chat | 接受并输出文本 |
| Anthropic | 输出 content_block_delta 和 content_block_stop，没有对应 content_block_start |
| Responses | 输出 output_text.delta 与 done，没有对应 output_item.added / content_part.added |

仅保留 finish 也被接受：Responses 最终 output 仍含 Hello，Chat / Anthropic 则没有文本事件。
这不是 A03 的副本矛盾：即使终态两份内容一致，序列仍然缺少生命周期。

### 实现原因与处理边界

[完整流校验](../../src/my_llm/middleware/validation.py#L440) 没有检查 start、块开启／更新／关闭及其索引关联。
[Anthropic Decoder](../../src/my_llm/codec/anthropic/decoding/decoder.py#L313) 和
[Responses Decoder](../../src/my_llm/codec/openai/responses/decoding/decoder.py#L370)
按事件逐项渲染，依赖入口已保证序列完整。

上述输出只记录删减事件后的实际行为，不构成对残缺序列的支持承诺。
此前关于新增生命周期校验和破坏性输入验收的建议撤回。

## A05：公开翻译入口只做弱结构校验

**处理决定：不修复。** 这些错误类型与非法值不在已确认的业务场景中，
不为此补充值校验、异常包装或静默类型转换。

### 复现与影响

公开模型的 Python 类型标注不在构造时强制执行；下列样本刻意传入不符合字段类型的值。

```python
codec.decode(
    messages=[AssistantMessage("assistant", [TextContent(17)])],
    target_format=target,
)
```

| 目标 | 实际结果 |
| --- | --- |
| Chat | 底层 TypeError：expected str instance, int found |
| Responses | 正常返回 output_text.text=17 |
| Anthropic | 正常返回 text=17 |

Message(role="unknown", content=[TextContent("hello")]) 也可以输出原生 unknown role。
encode 方向，三个 source 都接受 role="invalid" 并生成对应中间模型；三个 source 的函数声明
name=17 均可生成 FunctionTool.name=17。公开 delta.value=17 也能进入原生 Chat content。

### 实现原因与处理边界

- [BaseEncoder.messages](../../src/my_llm/codec/base/encoder.py#L131) 直接投影到中间模型，
  不经过 generate 输入路径使用的原生校验。
- [middleware 内容与消息校验](../../src/my_llm/middleware/validation.py#L163)
  主要检查必填键和类型标签；text、role、function name 等已识别字段的值约束不完整。
- [BaseDecoder.messages / chunks](../../src/my_llm/codec/base/decoder.py#L588) 复用这套结构校验。
  它们不像 generate 那样经过 typed generation_input 和最终 SDK request 验证。

按当前业务范围，上述行为不作为需要补充防御的缺陷。此前关于统一值校验和错误值输入验收的建议撤回。
这不否定现有校验，也不授权移除其他正常业务路径仍需要的校验。

## 其他观察与未确认范围

### O01：公开完整流存在重复校验

MyCodec.decode 先经 validate_complete_stream 调用 validate_events，随后
BaseDecoder.chunks 再次 validate_events。这是同一事件树的重复遍历和职责重叠，
未测量性能影响，也未将其算作补丁型行为缺陷。应结合公开边界收敛处理，不能直接删除仍承担输入保护的校验。

### O02：Chat 旧式函数调用的消息／流支持范围不一致

Chat 消息 Encoder 支持 function_call；流 Encoder 只消费 delta.tool_calls，
不消费 delta.function_call。官方 Schema 接受的旧式函数流可以只剩 finish_reason=function_call，
没有中间 ToolCall。此项记录为能力覆盖差异：原生 Schema 接受某字段，不等于项目承诺翻译它。
在明确旧式流是否属于支持范围前，仅保留该能力覆盖观察，不要求通过补丁扩展支持。

### 未发现新增确定反例的路径

本轮继续检查了 generate 的完整请求准备、同步／异步 input hook、output hook 与最终结果、
缓存提交与重复批次、流早关／失败、partial result，以及响应和 SDK client 的关闭顺序。
结合现有测试，未发现这些路径新增的确定行为缺陷；这不构成穷尽证明。

未发现需要推翻 Encoder / Decoder 语义损失方向的证据。未建模原生字段的丢弃、
跨协议 opaque reasoning 的损失，以及用户显式配置排除内容，必须分别判断，不能混为静默丢数据。

## 验证记录与限制

- Python 3.12.12；OpenAI SDK 2.54.0；Anthropic SDK 0.125.0；Pydantic 2.14.1。
- 本轮完整既有测试：`2764 passed in 14.48s`。
- 独立的三源 × 三目标双工具转换矩阵：9/9 通过。Chat / Responses 使用交错参数增量，
  Anthropic 使用两个工具块；原生输入先经过各自官方 SDK Schema，直接检查目标调用 ID、名称和 JSON 参数，
  未通过重新编码目标结果来自证。该矩阵只验证所列工具场景，不代表全部流生命周期通过。
- 前述空 item、summary 的相关既有测试 18 项也通过；它们缺少对应语义断言，不能用全绿推翻复现。
- 原生反例先用安装的官方 SDK TypeAdapter 校验，再调用公开 MyCodec。
  Schema 校验只证明单事件形状，顺序、生命周期和增量／终态一致性另作比较。
- A03—A05 的公开边界反例刻意破坏数据或使用非业务输入；按最终处理决定，不要求入口新增拒绝或恢复行为。
- A02 补充了 32/32 下一轮实际 HTTP 请求的回传一致性验证，以及其中 16 个消息缓存场景的重复批次一致性验证。
- 本轮仅作本地代码和测试审计，未调用远端供应商，也未使用或保存验证密钥。
- 没有为本报告修改实现、增加回归测试或提交修复；A01—A05 均已决定不修复。

## 可运行复现

在项目根目录的开发环境运行下列代码。只调用公开业务入口；原生样本独立构造，
不复用项目的编解码实现生成预期结果。所有 ID、模型名和 opaque 值均为虚构测试数据。

### A01 / A02：有效原生流

```python
from my_llm import MyCodec
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

base = {
    "id": "audit_response", "object": "response", "model": "audit",
    "created_at": 0, "parallel_tool_calls": True,
    "tool_choice": "auto", "tools": [],
}


def created():
    return {
        "type": "response.created",
        "response": {**base, "status": "in_progress", "output": []},
    }


def completed(items):
    return {
        "type": "response.completed",
        "response": {**base, "status": "completed", "output": items},
    }


def convert(events):
    validator = TypeAdapter(ResponseStreamEvent)
    for index, event in enumerate(events):
        event["sequence_number"] = index
        validator.validate_python(event)
    return MyCodec().convert(
        chunks=events,
        input_format="openai_responses",
        target_format="openai_responses",
    )["chunks"]


part = {"type": "output_text", "text": "hello", "annotations": [], "logprobs": []}
empty = {
    "type": "message", "id": "empty_before", "role": "assistant",
    "status": "completed", "content": [],
}
text = {**empty, "id": "text_after", "content": [part]}
address = {"output_index": 1, "item_id": text["id"], "content_index": 0}
events = [
    created(),
    {"type": "response.output_item.added", "output_index": 0,
     "item": {**empty, "status": "in_progress"}},
    {"type": "response.output_item.done", "output_index": 0, "item": empty},
    {"type": "response.output_item.added", "output_index": 1,
     "item": {**text, "status": "in_progress", "content": []}},
    {"type": "response.content_part.added", **address, "part": {**part, "text": ""}},
    {"type": "response.output_text.delta", **address, "delta": "hello", "logprobs": []},
    {"type": "response.output_text.done", **address, "text": "hello", "logprobs": []},
    {"type": "response.content_part.done", **address, "part": part},
    {"type": "response.output_item.done", "output_index": 1, "item": text},
    completed([empty, text]),
]
output = convert(events)
print("A01", [item["id"] for item in output[-1]["response"]["output"]])
# 实际 ['text_after', 'empty_before']；源顺序相反。

reasoning = {
    "type": "reasoning", "id": "audit_reason", "encrypted_content": "opaque",
    "summary": [{"type": "summary_text", "text": value} for value in ("a", "b")],
}
events = [
    created(),
    {"type": "response.output_item.added", "output_index": 0,
     "item": {**reasoning, "summary": []}},
]
for index, part in enumerate(reasoning["summary"]):
    address = {"output_index": 0, "item_id": reasoning["id"], "summary_index": index}
    events.extend([
        {"type": "response.reasoning_summary_part.added", **address,
         "part": {"type": "summary_text", "text": ""}},
        {"type": "response.reasoning_summary_text.delta", **address, "delta": part["text"]},
        {"type": "response.reasoning_summary_text.done", **address, "text": part["text"]},
        {"type": "response.reasoning_summary_part.done", **address, "part": part},
    ])
events.extend([
    {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
    completed([reasoning]),
])
output = convert(events)
print("A02 part.done", [
    (event["summary_index"], event["part"]["text"])
    for event in output if event["type"] == "response.reasoning_summary_part.done"
])
print("A02 completed", output[-1]["response"]["output"][0]["summary"])
# part.done 为 [(0, 'a\nb')]；completed 仍是两个 part。
```

### A03 / A04 / A05：公开边界

```python
from my_llm import AssistantMessage, Message, MyCodec, TextContent
from openai.types.chat import ChatCompletionChunk
from pydantic import TypeAdapter

codec = MyCodec()
base = {"id": "audit_chat", "object": "chat.completion.chunk", "model": "audit", "created": 0}
native = [
    {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"},
                         "finish_reason": None}]},
    {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
]
for event in native:
    TypeAdapter(ChatCompletionChunk).validate_python(event)

stream = codec.encode(chunks=native, input_format="openai_chat").stream
finish = stream[-1]
finish.messages[0].content[0].text = "EDITED_PUBLIC_MESSAGE"
print("A03 messages", codec.decode(
    messages=finish.messages, target_format="openai_responses",
)["messages"])
print("A03 stream", codec.decode(
    chunks=stream, target_format="openai_responses",
)["chunks"][-1]["response"]["output"])
# 前者为 EDITED_PUBLIC_MESSAGE，后者仍为 Hello，均不报错。

stream = codec.encode(chunks=native, input_format="openai_chat").stream
incomplete = [event for event in stream if event.type != "block_start"]
for target in ("openai_chat", "openai_responses", "anthropic"):
    output = codec.decode(chunks=incomplete, target_format=target)["chunks"]
    print("A04", target, [event.get("type", event.get("object")) for event in output])
    for label, messages in (
        ("text", [AssistantMessage("assistant", [TextContent(17)])]),
        ("role", [Message("unknown", [TextContent("hello")])]),
    ):
        try:
            print("A05", target, label, codec.decode(messages=messages, target_format=target))
        except Exception as error:
            print("A05", target, label, type(error).__name__, str(error))
```

这些代码仅用于复查已记录的行为，不是修复待办或新增测试要求。
后续若业务范围改变或出现正常业务路径的反例，应重新评估保障范围及对应预期。
