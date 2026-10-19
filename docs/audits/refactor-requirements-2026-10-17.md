# 最初重构需求与当前实现对照

> 历史验收记录：本文描述对应审计时点。当前翻译字段契约以[翻译字段边界](../design/translation-boundaries.md)为准；文中未知字段透传、恢复记录及开放模型扩展结论已撤销。

本页记录此前重构快照；2026-10-18 的字段回放及 Transport 验证职责以[当前架构](../design/architecture.md)为准。
截至 2026-10-18，已按后续要求取消歧义输入默认 Chat 的规则，仅唯一匹配时自动绑定，其他情况需显式指定 stdio_protocol。

核对日期：2026-10-17。基线是用户提供的 `myllm_refactor.md`，共 233 行，
SHA-256 为 `573045c4a0ba3ec98e914070937d4dfcb9d6e715ddc05fde800a905fc72cf498`。
以下“原文行号”均指该文件；没有改写原始需求文件。
实现依据为当前工作区，包括尚未提交的协议修复和目录调整。

**结论：按标准协议输入范围及后续明确要求核对，核心行为要求已落地。**
与最初示例的差异主要是 Decoder 职责扩展、SDK 构建位置、转换边界、目录和属性命名，
具体分类见下表。本轮核对只更新文档。

## 验收范围与结论更正

用户明确：调用者输入遵循五家的标准协议，不会携带库内部的
`__my_llm_codec_*` 控制字段。这些键属于中间态的私有命名空间。
未知字段保留要求适用于协议数据及扩展，不包含伪造内部控制结构的输入。

本报告前一版把人为构造的保留键碰撞列为未知字段保留和工具 ID 校验缺口，
超出了本项目的输入契约，现撤回该结论。R05、R23 按标准协议输入判定为符合；
不据此增加隔离、转义或额外校验实现。

## 已知设计差异与边界

| 项目 | 原始要求 / 示例 | 当前实现与判定 |
| --- | --- | --- |
| Encoder / Decoder 对称性 | 第 73—75 行要求双向对称转换。 | messages / tools / chunks / usage 双向对称；完整抽象接口因职责不同而不对称。Encoder 另需 match_item，继承固定 matches；Decoder 另需 request_parameters、assemble_request、profile_defaults、usage_modes，继承固定 request，并有转换校验、结果形状与服务端历史能力钩子。SDK 职责独立到 BaseTransport，由 BaseCodec 统一组合，见[接入指南](../guides/adding-protocol.md)。 |
| SDK 构建位置 | 第 218 行由 MyLLM 构建官方 SDK client。 | MyLLM 管理 SDK 客户端缓存，通过 Codec.transport 的 create_client / close_client 执行原生创建与关闭，端点及响应清理位于协议根目录的 transport.py。客户端按实例复用，响应按请求独占，见[生命周期设计](../design/client-lifecycle.md)。 |
| 任意格式转换 | 第 139—141 行支持任意源协议到任意目标协议。 | 五乘五路线存在，但不代表任意原生特性都可转换。未知附加字段跨协议过滤；必要的不透明内容、签名语义、不可表达媒体或无法保留的工具 Schema 约束会拒绝。属于后续“保证不报错前提下最大兼容”的范围收敛，见[配置边界](../reference/configuration.md#跨协议请求的处理边界)。 |
| 离线流转换 | 原文描述 MyLLM 管理状态，并允许独立转换流块。 | MyCodec.decode 对完整的跨协议流批次会创建本次调用的局部 StreamState 并 finish；实时请求的状态仍由 MyLLM 创建。实例不保留该局部对象，所以仍满足对象无状态，但多了原文没有展开的批次累积路径。见 [converter.py](../../src/my_llm/codec/converter.py)。 |
| 协议歧义 | 原文未规定同时匹配多种协议时的优先级。 | 当前注册顺序优先 openai_chat；整段 messages 与 tools 一起识别，显式 stdio_protocol 可消歧。不能将示例字典的排列顺序视为已约定的 Anthropic 优先规则。见 [注册与识别](../../src/my_llm/codec/converter.py)、[绑定](../../src/my_llm/invoke.py#L49)。 |
| MyLLM 属性命名 | 构造示例写 stdio_protocol / profile；后文又写 protocol_or_api_style / resolved_profile。 | 当前统一为 stdio_protocol 和 resolved_profile，没有重复提供 MyLLM.profile / protocol_or_api_style 别名。语义符合输入输出协议与上游协议分离，但与原文部分名称不同；原文自身需统一名称。 |
| 目录命名 | 第 66—67 行写 base/encode.py、base/decode.py。 | 当前为 encoder.py、decoder.py；原生 encoding / decoding 的转换方法直接落在实现类，辅助按 content 与 transport 分开，公共机制位于 shared。目录树已随 2026-10-18 整理更新，见[目录树](../design/architecture.md#原生协议目录)。 |
| 消息内部异常 | 第 227 行说“不处理”内容或工具调用中断。 | 当前检测缺少终态、重复终态、快照冲突和无效工具 JSON，会抛出错误并附可表达的部分结果；没有自动续写、重试修复、执行工具或发布半成品调用。将“检测并上抛”和“中断恢复”区分后，与设计边界相容。见 [StreamState](../../src/my_llm/stream/_state.py#L309)、[异常上抛](../../src/my_llm/invoke.py#L143)。 |
| 中间态编辑 | 原文要求开放字典，未规定编辑后的同协议合并行为。 | 当前原生快照优先，同协议精确回放针对未经修改的中间态；它不是可编辑消息的公开 API。不能从“开放字典”推导出修改语义字段后必然覆盖快照的承诺。见 [original()](../../src/my_llm/codec/shared/conversion.py)。 |

原文伪代码存在不完整的条件 return、protocol/profile 混写等情况。
本次按行为要求核对，没有把不完整示例逐字执行作为验收标准。

## 逐项核对

原文现状分析的七项问题分别由 R01、R10、R11、R19、R20、R14、R21 承接；
下表同时覆盖重构方案中的独立约束。

| 编号 | 原文位置与要求 | 判定与当前证据 |
| --- | --- | --- |
| R01 | 61—68、124—132：共同基类，五个 Codec 继承 BaseCodec。 | 符合。原有三个基类均为抽象类，并新增独立的 BaseTransport，由 BaseCodec 组合；五协议继承相应契约。见[实例与抽象测试](../../tests/unit/test_native_codecs.py)。 |
| R02 | 73：每种协议独立一对无状态 Encoder / Decoder。 | 符合。五对实现独立，实例无 __dict__ / profile；[对象契约](../../tests/unit/test_native_codecs.py#L8)、[目录边界](../../tests/contract/test_codec_boundaries.py)。 |
| R03 | 73：协议参数差异收入 **parameters。 | 符合。六个 BaseCodec 转换方法均有 profile=None 和 **parameters；[统一方法](../../src/my_llm/codec/base/codec.py)。previous_response_id 由请求阶段的参数字典传递。 |
| R04 | 74—75、89—116：消息、工具、流块双向转换。 | 符合核心对称契约；完整 ABC 的职责扩展见上表。[BaseCodec](../../src/my_llm/codec/base/codec.py)、[转换测试](../../tests/contract/test_protocol_conformance.py)。 |
| R05 | 76：中间态不使用固定模板，未知字典保留层级。 | 符合。中间态为开放字典，协议数据的嵌套扩展、列表索引和冲突字段有直接检查；验收输入范围见上文。`保留测试`（原文件已移除）。 |
| R06 | 80：BaseCodec 无状态且不保存 profile。 | 符合。只声明协议身份和转换器，无请求级字段；[BaseCodec](../../src/my_llm/codec/base/codec.py)。 |
| R07 | 119—120：match_protocol 返回协议名或 None。 | 符合。[BaseCodec.match_protocol](../../src/my_llm/codec/base/codec.py)委托自身 Encoder；MyCodec 遍历所有候选直到匹配，全部失败返回 None。 |
| R08 | 136—138：MyCodec 同时持有五个无状态实例，不固定 profile。 | 符合。仅保存 codecs 注册表；[MyCodec](../../src/my_llm/codec/converter.py)。 |
| R09 | 139—141：接受五协议并允许源、目标任意组合。 | 已实现 25 条路线；特性可表达性有限，见上表。消息、工具各 25 组，原生流实时 / 批次共 50 组有独立预期与 SDK / Schema 校验；[协议契约](../../tests/contract/test_protocol_conformance.py)。 |
| R10 | 35：路由内聚，不让 MyLLM 感知具体 Codec。 | 符合。invoke.py 只导入 MyCodec；具体类型只在注册表组合；[边界测试](../../tests/contract/test_codec_boundaries.py)、[invoke.py](../../src/my_llm/invoke.py#L12)。 |
| R11 | 39、200—203、214：整段输入只识别并绑定一次。 | 符合。首次合并 messages/tools 识别，绑定后不重新猜测；[一次绑定测试](../../tests/unit/test_invoke.py#L147)、[工具消歧测试](../../tests/integration/test_message_inputs.py)。遍历消息做格式校验不等于逐条选择不同源协议。 |
| R12 | 192—209：构造时接收 provider/model，保存副本与解析快照。 | 符合。分别 deepcopy 并保存 resolved_profile；[构造](../../src/my_llm/invoke.py#L34)、[快照所有权](../../tests/unit/test_ownership.py)。 |
| R13 | 209：实例 generate 不再动态接收 provider/model。 | 符合。实例方法没有这些参数、client 或连接覆盖；单独的一次性 generate 是新建实例的入口，不是修改现有实例；[接口测试](../../tests/unit/test_public_api.py)、[错误参数测试](../../tests/unit/test_invoke.py#L116)。 |
| R14 | 51、209、230：合并 protocol / api_type。 | 符合。只有五值 Protocol；ProviderProfile / ResolvedProfile 无 api_type，公开 API 无 APIType；[配置类型](../../src/my_llm/profile/configuration.py#L16)、[接口测试](../../tests/unit/test_public_api.py)。 |
| R15 | 213—216：标准消息 / 工具输入按 I/O 协议编码，profile=None。 | 符合。[generate](../../src/my_llm/invoke.py#L83)对 stdio_protocol 编码，未传 provider profile。 |
| R16 | 217—218：按上游 profile 解码并发起官方流请求。 | 符合行为；构建 client 的位置已调整。[request](../../src/my_llm/codec/converter.py)将 resolved_profile 传入目标 Decoder；[五协议 SDK 测试](../../tests/integration/test_native_sdk_transport.py)、[OpenAI 测试](../../tests/integration/test_sdk_transport.py)。 |
| R17 | 222：返回的 SDK 事件按上游 profile 编码。 | 符合。[上游事件编码](../../src/my_llm/invoke.py#L133)传 input_format=profile.protocol、profile=profile；[配置路径回归](../../tests/unit/test_invoke.py#L160)。 |
| R18 | 223：输出按绑定 I/O 协议解码，profile=None。 | 符合。[标准事件渲染](../../src/my_llm/invoke.py#L116)与 [result](../../src/my_llm/codec/converter.py)都使用调用者协议且不传上游 profile。 |
| R19 | 43：OpenAI Chat / Responses 编解码分离。 | 符合。各自独立 encoding / decoding；原生实现之间不互相导入；[边界检查](../../tests/contract/test_codec_boundaries.py)。 |
| R20 | 47、229：公开输入输出不再使用自定义消息协议。 | 符合。公开返回原生字典或 Responses items；Message / MessageChunk / ToolCall 等类型已移除。StreamState / BlockState 是内部累积实现；[公开接口](../../tests/unit/test_public_api.py)、[标准协议输出](../../tests/contract/test_protocol_conformance.py)。 |
| R21 | 55：仅思考开启且当条有调用时补空思考字段。 | 符合空值补充条件。Chat / Ollama 的 replay 开关、thinking、调用、已有可见思考共 32 种组合直接验证；[条件测试](../../tests/unit/test_encoding.py#L201)。已有可见思考的回放和缺失值的补充不是同一行为。 |
| R22 | 227：外部工具调用必须完整，不做内部中断恢复。 | 符合。工具及后续块等待成功终态和 EOF，参数先完整解析再发布；失败不输出半成品，无自动修复流程；[流状态测试](../../tests/contract/test_stream_state.py#L264)。 |
| R23 | 228：处理消息间的调用 / 结果 ID 异常。 | 符合。标准协议输入中的重复、缺失、未知、乱序及可选 ID 情况均检查。[历史测试](../../tests/unit/test_messages.py)、[validate_history](../../src/my_llm/codec/shared/transport.py)。 |
| R24 | 231：移除自定义 client 支持。 | 符合。构造和实例 generate 均无 client 注入；各原生 transport 创建官方 SDK client，MyLLM 实例复用并统一关闭；[资源测试](../../tests/unit/test_client_lifecycle.py)、[并发验证](../../tests/integration/test_concurrency.py)。 |
| R25 | 233：无需旧版兼容路径，代码清晰、无冗余。 | 未保留旧消息包、旧 client 参数或旧文件转发；职责按当前目录归位，初始化文件仅导出。见[结构检查](../../tests/contract/test_codec_boundaries.py)、[发行包检查](../../tests/packaging/test_distribution.py)。“最简洁”是设计评价，不能以测试通过证明绝对最优；Decoder 扩展与局部批次累积的具体差异已单列。 |

## 文档表述问题

本轮还校正了两处配置说明，不改变运行时行为：

- Ollama max_retries 不是“设置后不生效”：None / omit / 0 可用，正数在构造时明确拒绝。
- “Ollama 默认 think=True”应指显式启用推理后的生成值。
  未指定 reasoning / 非 none effort 时，代码不自动增加 think；不据此推断服务端或具体模型的默认行为。

架构验收和无损说明已统一到上述标准协议输入范围。

## 验证与范围

- 当前完整测试：**915 passed**。实际读取相关测试的输入、断言和参数化范围后，才将其用于上表。
- 独立反射检查：五种 Codec 继承关系、五对实例无 profile / __dict__、六个转换签名、移除的公开 API 均通过。
- 文档链接与结构约束专项共 10 项通过，Ruff 检查及格式检查通过。
- 本轮仅修改核对报告和文档，业务及测试代码保持原样。
- 全部验证在本地执行，没有调用模型服务；保留键构造输入不列入本次需求缺陷。
