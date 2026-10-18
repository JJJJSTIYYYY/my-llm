# 五协议请求路径审计

核对日期：2026-10-19。审计对象是当前工作树，包含已有的未提交重构。
本次仅新增这份报告，没有修改运行时代码或测试。

结论：

1. **存在目前仅由一个协议触发的公共分支。** 主要是协议声明的能力、请求字段路径和中间流操作；
   未发现公共编排按具体协议名称硬编码执行的判断，不能将这些单协议分支直接认定为架构缺陷。
2. **仍有可复用逻辑分散实现。** 最明显的是 Chat / Responses 的 SDK 参数封装，以及五个 Encoder 的事件批次收尾；
   核心参数解析、恢复记录、字段过滤和资源所有权已集中共享。
3. **抽象方法清单基本完整，但全部接口约束尚未覆盖。** 具体缺口是 Decoder 的返回类型没有表达来源分组要求，
   新增协议也不会自动进入现有的全部契约检查。签名、组件身份、无状态和流生命周期还需类型检查或行为测试保证。

## 请求路径与协议分工

五种上游协议使用同一编排链；输入 / 输出协议由 stdio_protocol 独立决定。

```text
MyLLM.generate
  → 固定 / 识别 stdio_protocol，Transport.validate_input
  → 输入 Encoder.messages / tools → 中间态
  → MyCodec.request → BaseCodec.request
      → 上游 Transport.validate_parameters → PreparedOptions
      → 上游 Decoder.request
          → messages / tools → assemble_request → request_parameters
      → 上游 Transport.request
          → 历史关联 → 字段过滤 → SDK 类型检查 → validate_request
  → MyLLM._get_client → 上游 Transport.upstream → 官方 SDK
  → 上游 Encoder.chunks → StreamState.feed / finish
  → 输出 Decoder.chunks / messages / result → MessageStream / 最终结果
```

入口见 [invoke.py:96](../../src/my_llm/invoke.py#L96)，固定请求阶段见
[BaseCodec.request:48](../../src/my_llm/codec/base/codec.py#L48)、
[BaseDecoder.request:22](../../src/my_llm/codec/base/decoder.py#L22)、
[BaseTransport.request:101](../../src/my_llm/codec/base/transport.py#L101)。
stream=False 同样消费这条流执行链，不存在单独的非流式 SDK 路径。

| 上游协议 | 请求历史 | 工具位置 | 官方 SDK 执行入口 | 响应资源归属 |
| --- | --- | --- | --- | --- |
| openai_chat | messages | tools | client.chat.completions.create | owned_stream，关闭响应流 |
| openai_responses | input | tools | client.responses.create | owned_stream，关闭响应流 |
| anthropic | messages，system 单独组装 | tools | client.messages.create | owned_stream，关闭响应流 |
| gemini | contents，system_instruction 位于 config | config.tools | client.aio.models.generate_content_stream | 请求独立 ExitStack，关闭迭代器与跟踪的响应体 |
| ollama | messages | tools | client.chat | owned_stream，调用响应迭代器 aclose |

客户端均归 MyLLM 实例缓存，响应资源归单次请求。Gemini 的响应体跟踪和 Ollama 的请求头缓存键
位于各自 Transport，没有进入公共编排。

## 1. 公共函数中的单协议分支

下面的“唯一触发”指当前五套原生实现的生产路径；其他协议以后可以产生相同的通用操作。
必须区分来源协议和目标协议，例如 Responses 的来源分组会在四种其他目标 Decoder 中触发。

| 公共位置 | 当前触发者与条件 | 代码依据 | 判断 |
| --- | --- | --- | --- |
| BaseTransport.validate_parameters；validate_history | Responses 接受 previous_response_id，并允许工具结果引用服务端历史中的调用 | [公共参数检查:89](../../src/my_llm/codec/base/transport.py#L89)、[历史检查:125](../../src/my_llm/codec/shared/transport.py#L125)、[Responses 声明:33](../../src/my_llm/codec/openai/responses/transport.py#L33) | 能力分派合理；不过公共 API / 基类确实保留了这一目前专属的参数。 |
| coalesce_groups | Responses 来源的相邻 assistant items 合并为一个 assistant 分组 | [合并条件:173](../../src/my_llm/codec/shared/conversion.py#L173)、[唯一标记生产者:132](../../src/my_llm/codec/openai/responses/encoding/encoder.py#L132) | 通用分组机制合理；Chat、Anthropic、Gemini、Ollama Decoder 均消费它。 |
| BaseTransport._parent | Gemini 的 tool_path=(config, tools) 触发嵌套遍历 | [公共路径解析:137](../../src/my_llm/codec/base/transport.py#L137)、[Gemini 路径声明:63](../../src/my_llm/codec/gemini/transport.py#L63) | 通用路径算法合理。 |
| StreamState._snapshot；messages | Responses 的终态 snapshot 校验并补齐最终块，最终消息走快照返回 | [快照处理:172](../../src/my_llm/stream/_state.py#L172)、[消息返回:297](../../src/my_llm/stream/_state.py#L297)、[快照生产者:372](../../src/my_llm/codec/openai/responses/encoding/encoder.py#L372) | 必要的终态协调，不应为消除单协议调用而复制累积器。 |
| StreamState._metadata | Responses 声明 merge=[response_items] | [通用合并:133](../../src/my_llm/stream/_state.py#L133)、[操作生产者:341](../../src/my_llm/codec/openai/responses/encoding/encoder.py#L341) | 合并字段由原生 Encoder 声明，公共层不理解原生字段。 |
| StreamState._update_block | Chat 声明 merge=[function] | [通用合并:202](../../src/my_llm/stream/_state.py#L202)、[操作生产者:90](../../src/my_llm/codec/openai/chat/encoding/content.py#L90) | 同上，属于通用块更新。 |
| StreamState._begin | Responses 显式产生 start 操作 | [处理:143](../../src/my_llm/stream/_state.py#L143)、[生产者:180](../../src/my_llm/codec/openai/responses/encoding/encoder.py#L180) | 操作模型的正常子集。其他协议也通过公共 _start 产生输出开始事件。 |
| StreamState._stop_block | Anthropic 显式产生块停止操作 | [处理:222](../../src/my_llm/stream/_state.py#L222)、[生产者:182](../../src/my_llm/codec/anthropic/encoding/encoder.py#L182) | 对应原生块生命周期。 |
| StreamState._annotation | Anthropic citations_delta 产生引用注释操作 | [处理:153](../../src/my_llm/stream/_state.py#L153)、[生产者:152](../../src/my_llm/codec/anthropic/encoding/encoder.py#L152) | 保证注释与正确块关联，属于共享状态能力。 |
| StreamState._reason 的非 finish 路径 | Anthropic 在 message_delta 更新原因，直到 message_stop 才终结 | [处理:146](../../src/my_llm/stream/_state.py#L146)、[生产者:183](../../src/my_llm/codec/anthropic/encoding/encoder.py#L183) | 必须保留原因与终态的区别。 |

因此，“公共层无协议名判断”成立，但“公共函数所有分支都至少由两个协议执行”不成立。
现有 [AST 边界检查](../../tests/contract/test_codec_boundaries.py#L32)主要检查协议名称字面量，
不能单独证明不存在由能力标记、字段路径或操作类型形成的单协议路径。

另外，[StreamState._delta:217](../../src/my_llm/stream/_state.py#L217)读取 event.extras，
但当前 src 和 tests 没有任何操作生产者写入这个键，在五条路径中实际执行 update({})。
这是可清理的闲置入口，没有证据表明它造成现有功能错误。

## 2. 可复用逻辑的重复实现

### 2.1 Chat / Responses 的 SDK overlay 流水线

这是最值得集中维护的重复机制。

- 参数准备：Chat [transport.py:59](../../src/my_llm/codec/openai/chat/transport.py#L59)
  与 Responses [transport.py:38](../../src/my_llm/codec/openai/responses/transport.py#L38)，
  都执行保护字段检查、绑定验证、extra_body 合并、token 路径校验和 PreparedOptions 替换。
- 请求输出：Chat [decoder.py:66](../../src/my_llm/codec/openai/chat/decoding/decoder.py#L66)
  与 Responses [decoder.py:54](../../src/my_llm/codec/openai/responses/decoding/decoder.py#L54)，
  连续执行 overlay_generated、headers/body 写入、apply_max_tokens 和返回；这一段 AST 完全一致。

可以在 shared/options.py 增加由调用者指定 overlay 容器的辅助函数。
原生 reasoning_effort / reasoning 的类型检查和生成方式仍应留在各自协议中。
抽取时必须保留 omit 删除、嵌套兄弟字段、生成值优先级、token 路径和所有权隔离。
当前属于维护一致性问题，没有证据表明重复已经引发行为错误。

### 2.2 五个 Encoder 的事件批次收尾

五条 _chunks 路径均重复：遍历 ops，将 usage 原生值交给 self.usage 转换，
再调用 batch(event, ops, PROTOCOL, self._consumed_fields(event))。

位置分别是 Chat [encoder.py:194](../../src/my_llm/codec/openai/chat/encoding/encoder.py#L194)、
Responses [encoder.py:349](../../src/my_llm/codec/openai/responses/encoding/encoder.py#L349)、
Anthropic [encoder.py:201](../../src/my_llm/codec/anthropic/encoding/encoder.py#L201)、
Gemini [encoder.py:175](../../src/my_llm/codec/gemini/encoding/encoder.py#L175)、
Ollama [encoder.py:186](../../src/my_llm/codec/ollama/encoding/encoder.py#L186)。

可以共享一个接受 usage 转换函数与 consumed 规则的批次封装函数，
复用位置邻近 shared/stream.batch；不需要把原生事件解析提升到基类，
也不需要因此增加新的必需抽象方法。代码量很小，优先级低于 2.1。

### 2.3 其他较小候选

| 重复机制 | 位置 | 建议 |
| --- | --- | --- |
| Chat / Ollama 的 tool_block 规范化 | [Chat:63](../../src/my_llm/codec/openai/chat/encoding/content.py#L63)、[Ollama:26](../../src/my_llm/codec/ollama/encoding/content.py#L26) | 可以共享字段投影与 arguments 解析；Ollama 额外剔除 function.index，需声明这个差异。不能直接互相导入实现。 |
| Chat / Ollama 的 reasoning replay 空字段补齐 | [Chat:145](../../src/my_llm/codec/openai/chat/decoding/decoder.py#L145)、[Ollama:144](../../src/my_llm/codec/ollama/decoding/decoder.py#L144) | 循环相同，仅输出字段不同；可以传入字段名和已判定的 enabled。原生 thinking_enabled 判断保留在协议侧。 |
| Chat / Ollama 的 assemble_request | [Chat:74](../../src/my_llm/codec/openai/chat/decoding/decoder.py#L74)、[Ollama:56](../../src/my_llm/codec/ollama/decoding/decoder.py#L56) | 函数 AST 完全一致，但只是短小的请求信封组装；为了这几行增加通用框架收益有限。 |

已经共享的部分包括 resolve_options / PreparedOptions、merge_bindings、token 路径、
消息投影、工具配对、replay、SDK 字段过滤、owned_stream 和 StreamState。
各协议的内容类型、usage 字段、流终态、SDK 生命周期及 ID / name 关联差异应继续由协议拥有。

## 3. 抽象基类对新增协议的约束

### 3.1 必需接口清单完整

核对消费端实际访问后，未发现“新增协议还必须实现某个基类没有声明的方法”的漏项。

| 基类 | 必需接口 |
| --- | --- |
| BaseCodec | protocol_or_api_style、encoder、decoder、transport |
| BaseEncoder | protocol、match_item、_messages、_tools、_chunks、usage |
| BaseDecoder | protocol、_messages、_tools、_chunks、usage、request_parameters、assemble_request |
| BaseTransport | message_path、tool_path、message_type、tool_type、validate_request、create_client、upstream |

validate_parameters、validate_input、filter_messages / filter_tools、validate_profile、
close_client、client_key、allows_external_calls 和 Decoder.result 都有默认实现；
它们可以按协议覆盖，不能仅因不是 abstractmethod 就判定为漏声明。

### 3.2 实际缺口：Decoder 返回类型未表达来源分组要求

[BaseDecoder._messages / _tools:79](../../src/my_llm/codec/base/decoder.py#L79)
仅标注 list[dict]，但 [retain:87](../../src/my_llm/codec/shared/replay.py#L87)
只在返回值为 Translations 时读取 groups，随后才执行原路径扩展合并。
返回普通 list 满足声明，却会跳过这部分逻辑；replay 仍可保存来源，不能称为信息全部永久丢失。

复现方式：继承 ResponsesDecoder，唯一改动是让 _messages 返回
list(super()._messages(values, profile, **parameters))；输入来自 Anthropic，
其文本块带 vendor_block="keep"。通过公开 messages 入口转换后：

```text
sorted(__abstractmethods__)    []
原实现 content[0].vendor_block   keep
普通 list 实现的同一路径          缺失
普通 list 输出中的 replay        仍存在
```

_tools 同样可复现：将 ChatCompletionsDecoder._tools 的 Translations 转成 list 后，
Gemini 工具容器的 vendor_bundle 原路径扩展缺失。现有五个 Decoder 均正确使用
Translations / decode_functions，这属于新增协议的契约缺口。

建议准确声明这两个钩子返回 Translations，并检查来源分组与扩展保留行为。
保留公开 messages / tools 返回 list 的接口；接入指南已有来源分组说明，
但当前类型签名和接口测试没有保证新实现遵守它。

### 3.3 新增协议的验证覆盖需要同步约束

当前 Protocol Literal、运行时 PROTOCOLS、MyCodec.codecs 和测试样例集合恰好一致，
但没有测试断言这些集合一致。
[测试协议清单](../../tests/helpers/protocols.py#L7)与
[边界检查 OWNERS / SDK 名单](../../tests/contract/test_codec_boundaries.py#L11)独立硬编码。

仅完成接入指南列出的运行时注册后，新协议可能不进入现有参数化契约测试。
新目录也会被 native_owner 当作共享目录，遗漏原生实现间的隔离检查或产生误判。

建议保留独立手写原生样例，同时增加注册集合一致性断言；
原生 owner 可根据已注册组件的模块位置确定，避免新增目录后忘记纳入。

### 3.4 ABC 不能保证的约束

最小探针确认，以下实现均能实例化：

- _messages 省略 profile / **parameters：直到固定入口调用才抛 TypeError。
- 子类遗漏空 __slots__：实例重新获得 __dict__，可以保存请求状态。
- Chat Codec 组合 Chat Encoder 与 Gemini Decoder：组件身份没有一致性校验。

另外，@final 是静态标记，不会在运行时阻止覆盖固定入口。
流操作形状、成功终态、响应关闭、客户端缓存键和并发隔离也是行为约束，
需要契约 / SDK 边界测试；单纯增加 abstractmethod 无法保证这些语义。

建议优先补准确的返回类型、注册一致性及现有行为契约的扩展覆盖，
无需为所有约束增加运行时元类或复杂的子类注册机制。

## 验证与后续优先级

本次重新执行：

- 全量测试：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`，
  **1258 passed in 9.15s**。
- 五协议原生 history / tool stream 探针，检查实际操作、分组、嵌套字段路径及 previous_response_id 能力。
- 原生方法 AST 比对，确认完整函数重复和上述参数封装片段。
- Decoder 普通 list 返回、错误签名、遗漏 slots 和混合组件的最小子类探针。

建议处理顺序：先补 Decoder 返回值与新增协议验证覆盖，再集中 SDK overlay 参数流水线；
事件批次收尾、工具规范化、空字段补齐和闲置 extras 读取可作为小范围整理。

测试和探针使用本地 SDK、独立样例和本地响应，未请求远端模型。
本次没有发现由所列单协议公共分支或重复代码直接造成的现有五协议功能错误；
测试通过不代表新增协议的全部语义约束已经由抽象基类保证。
