# 仓库清理与架构约束复核

> 历史验收记录：本文描述对应审计时点。当前翻译字段契约以[翻译字段边界](../design/translation-boundaries.md)为准；文中未知字段透传、恢复记录及开放模型扩展结论已撤销。

核对日期：2026-10-18。范围为当前工作区，包括尚未提交的重构与修复。
依据是[原始需求对照](refactor-requirements-2026-10-17.md)中的行为约束，以及后续明确的
Transport 分离、客户端实例复用、信息保留和内部参数无需兼容兜底要求。
旧目录、旧接口和历史测试数量不作为当前实现要求；当前职责见[架构文档](../design/architecture.md)。

结论：下表所列明确架构约束均有当前代码和验证依据，未发现未满足项。
这不意味着所有供应商特性都能跨协议等价表达，也不把测试通过当作代码已经绝对最简的证明。

## 清理内容

- 删除 185 个本地生成文件，共 2,027,522 字节：143 个 Python 字节码文件、
  3 个 `.DS_Store`、pytest / Ruff 缓存，以及旧 `dist` 目录内的 wheel、源码包和生成标记。
  共移除 34 个文件或目录入口；没有删除任何被 Git 跟踪的生成文件。
- 删除 `resolve_options` 未使用的 `previous_response_id` 参数；该参数由 Transport
  显式接收并校验，仍由 Responses Decoder 写入原生请求。
- 删除 `client_parameters` 中不可达的 Omit 分支：`ResolvedProfile.get` 已把 Omit 归一为默认值。
- 核对全部 72 个运行时模块的导入、动态路由、包导出及测试和文档引用，没有发现可确认删除的业务模块、
  函数或常量。公开门面、协议钩子、流事件回调、仅导出的 `__init__.py` 和 `py.typed` 均有用途。
- 当前文档明确区分 MyLLM、StreamState、MessageStream 的职责，并补齐请求路径中的参数准备阶段；
  开发指南中的旧目录及能力检查说明已标为历史快照。

没有移除 `.git`、开发虚拟环境、协议参考资料或历史审计依据。清理后项目目录没有残留空目录。
构建验证在临时目录执行，验证完成即移除产物，仓库根目录不保留旧发行包。

## 约束与当前证据

| 约束 | 当前实现与核验依据 |
| --- | --- |
| 五协议独立实现，Codec、Encoder、Decoder、Transport 无请求状态；对应 R01—R04、R06、R08、R19。 | [四个基类](../../src/my_llm/codec/base/codec.py)和五套原生实现采用统一契约；Profile 按调用传入，六个转换方法保留 profile 与参数入口。MyCodec 只保存注册表。[组件测试](../../tests/unit/test_native_codecs.py)检查继承、必需接口和无实例字典；[依赖测试](../../tests/contract/test_codec_boundaries.py)检查跨协议及反方向导入。 |
| MyLLM 统一编排，不包含具体供应商分支；对应 R10。 | [invoke.py](../../src/my_llm/invoke.py)通过 MyCodec 调用；[BaseCodec.request](../../src/my_llm/codec/base/codec.py)固定验证、翻译、原生过滤校验三个阶段。[依赖测试](../../tests/contract/test_codec_boundaries.py)限制厂商判断和 SDK 客户端引用的归属。 |
| Encoder / Decoder 负责翻译和信息保留；Transport 负责验证、字段过滤及官方 SDK。 | [BaseEncoder](../../src/my_llm/codec/base/encoder.py)恢复未修改分组，[BaseDecoder](../../src/my_llm/codec/base/decoder.py)保留转换扩展并组织请求；[BaseTransport](../../src/my_llm/codec/base/transport.py)承担输入、能力、历史关联及 SDK 边界验证。`接口隔离测试`（原文件已移除）与[依赖测试](../../tests/contract/test_codec_boundaries.py)均覆盖。 |
| 共同选项只准备一次，内部已确定的参数显式传递。 | Transport 返回请求独立的 PreparedOptions，Decoder.request 显式接收；后续重构将 request_parameters 固定在基类执行，再把参数字典传给 assemble_request(..., request=request)，协议不重复解析或应用参数。[选项测试](../../tests/unit/test_prepared_options.py)验证五协议各解析一次、验证先于翻译、嵌套 omit 和所有权隔离。 |
| 整批唯一识别并固定 stdio_protocol，上游协议与 I/O 协议独立；对应 R07、R11 及后续消歧要求。 | [MyCodec.match_protocol](../../src/my_llm/codec/converter.py)整批匹配，[MyLLM.guess_stdio_protocol](../../src/my_llm/invoke.py)只在唯一匹配时绑定；零匹配、多匹配均报 MessageValidationError，不默认选 Chat。显式 stdio_protocol 跳过推测，仍按所选协议验证输入。[推测边界测试](../../tests/unit/test_protocol_detection.py)、[调用测试](../../tests/unit/test_invoke.py)与[消息输入测试](../../tests/integration/test_message_inputs.py)检查唯一绑定、歧义拒绝和工具消歧。 |
| 构造时固定 provider / model 及快照，统一 Protocol；对应 R12—R14。 | [配置解析](../../src/my_llm/profile/resolution.py)按纯配置表合并并复制；Profile 不依赖 Codec、调用编排或流状态。实例 generate 不接受动态 provider / model / client。[配置测试](../../tests/unit/test_profiles.py)、[快照测试](../../tests/unit/test_ownership.py)和[公共接口测试](../../tests/unit/test_public_api.py)覆盖。 |
| 五乘五转换路线，输入与输出使用标准协议；对应 R09、R15—R18、R20。 | 输入按 stdio_protocol 且不带上游 Profile 编码；SDK 事件带上游 Profile 编码，输出按 stdio_protocol 且不带上游 Profile 解码。[独立预期](../../tests/contract/test_protocol_conformance.py)、[SDK 传输](../../tests/integration/test_native_sdk_transport.py)和[公开接口](../../tests/unit/test_public_api.py)验证消息、工具和流事件。 |
| 开放中间字典保留原路径扩展，恢复数据不进入 SDK 请求；对应 R05。 | [conversion](../../src/my_llm/codec/shared/conversion.py)和`replay`（原文件已移除）保存来源、分组与恢复记录；`原路径保留`（原文件已移除）、`JSON 保存与恢复`（原文件已移除）及[两轮 SDK 回放](../../tests/integration/test_replay_transport.py)验证。工具参数、结果和 JSON Schema 不按协议字段递归裁剪。 |
| 思考回放受显式配置和当前消息条件约束；对应 R21。 | [条件测试](../../tests/unit/test_encoding.py)覆盖回放开关、思考启用、工具调用和已有思考的组合；已有内容回放与缺失值补充保持区分。五协议的参数绑定统一使用公共 merge_bindings，按层递归合并并消费 omit；不保留 Responses 专属的 raw binding 路径，见[统一绑定测试](../../tests/unit/test_parameter_defaults.py)与[选项测试](../../tests/unit/test_prepared_options.py)。 |
| 流状态按请求独立，块身份明确，工具只在成功终态后完整发布；对应 R22、R23。 | [StreamState / BlockState](../../src/my_llm/stream/_state.py)维护显式操作和来源引用；[流状态测试](../../tests/contract/test_stream_state.py)覆盖分片、JSON 往返、相同文本的独立身份、快照冲突、签名和终态。[消息测试](../../tests/unit/test_messages.py)检查调用与结果 ID；检测错误并上抛，不自动修复或执行工具。 |
| 官方 SDK 客户端由 MyLLM 实例复用，响应资源按请求关闭；对应 R24 及后续客户端要求。 | [MyLLM](../../src/my_llm/invoke.py)缓存客户端并登记关闭回调；协议 Transport 创建与关闭 SDK 客户端，MessageStream 接收可关闭的异步生成器。[生命周期测试](../../tests/unit/test_client_lifecycle.py)、[回调测试](../../tests/unit/test_stream_callbacks.py)和[并发集成测试](../../tests/integration/test_concurrency.py)验证复用、取消、失败、提前退出、跨任务关闭与临时实例所有权。 |
| 无旧接口兼容层，初始化文件仅导出；对应 R25。 | 当前无旧 Message / MessageChunk / ToolCall / APIType、client 注入或旧路径转发模块。[公共接口测试](../../tests/unit/test_public_api.py)、[初始化与导入约束](../../tests/contract/test_codec_boundaries.py)和[发行包测试](../../tests/packaging/test_distribution.py)检查。所有运行时模块均可从新 wheel 独立导入。 |

MyCodec 的离线完整流转换会在当前调用内使用 `stream._state.StreamState`。
这是[原始对照](refactor-requirements-2026-10-17.md)已记录的局部批次累积设计，
有`完整流往返测试`（原文件已移除），不将状态保存在 Codec 实例中；
没有为消除包层级上的双向引用而另建第二套累积器。

## 验证结果与范围

- 全量测试：**1258 passed**，包括全部协议组合、并发隔离、字段保留、客户端和发行包测试。
- Ruff 检查及 119 个 Python 文件的格式检查通过；`git diff --check` 通过。
- 固定协议资料验证通过：24 个 Schema、59 个模板、19 个流文件、90 帧事件。
- 五协议离线示例通过。
- 临时目录构建 wheel / sdist，Twine 严格检查通过；隔离进程从新 wheel 导入全部 72 个运行时模块通过。
- 清理和检查使用禁止生成字节码、禁用测试及 lint 缓存的命令，避免重新留下本地缓存。

验证使用当前依赖、本地响应和固定协议资料，没有访问远端模型服务；
不用于承诺供应商的实时协议变更、签名密码学有效性、网络吞吐或跨线程共享客户端能力。
