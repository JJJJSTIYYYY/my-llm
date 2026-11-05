# 输入修复与工具图片端到端测试

[SenseNova 脚本](../../examples/sensenova_e2e.py) 和
[DeepSeek 脚本](../../examples/deepseek_reasoning_e2e.py) 共用
[输入用例](../../examples/e2e_inputs.py)，直接构造 Chat、Responses、Anthropic 原生字典
和类型化 Middleware 消息，不通过被测转换器生成测试输入。

接口地址、鉴权、模型、输出上限、思考开关、参数路径和能力声明均由
`ProviderProfile` / `ModelProfile` 管理。example 不手工补齐思维、工具结果或删除孤立消息，
推理回传场景用只读观察器包装现有 Transport，实际调用仍交给原 Transport 和官方 SDK。
API 密钥只在运行时读取，不进入源码、报告或命令参数。

## 覆盖范围

| 用例 | 输入与上游 | 检查 |
| --- | --- | --- |
| `reasoning_missing` | 四种输入 → Chat | 缺少推理字段；MyLLM 准备的工具调用消息包含 `reasoning_content=""` |
| `reasoning_null` | Chat → Chat | 原生字段显式为 `null`，与缺失字段分开观测 |
| `reasoning_empty` | Chat → Chat | 原生字段显式为空字符串，不能预先认定 API 拒绝 |
| `missing_tool_result` | 四种输入 × 三种上游 | 缺少工具结果时补空结果，调用 ID 一一配对 |
| `orphan_tool_result` | 四种输入 × 三种上游 | 保留正确结果，丢弃没有先前调用的结果消息及其标记文字 |
| `tool_result_images` | Responses、Anthropic → Chat | 图文工具结果先触发语义损失；启用预置 hook 后保留工具文字、配对和顺序，将 Base64 图片移至末尾 user 消息 |
| `tool_result_image_placeholders` | Responses、Anthropic → Chat | Profile 声明不支持视觉时，预置 hook 写入 `[Unsupported Image]`，仍保留工具文字 |
| `reasoning_replay` | 三种上游各自的原生格式与 Middleware，共六条路径 | 第一轮必须返回非空推理；逐项比较原始 SDK 响应、MyLLM 返回值、下一轮准备结果与实际 HTTP JSON 中的回传数据 |

输入修复共 **34 个场景**。每个历史修复场景先发一次未经消息修复的原生 API 对照，再执行
`generate(stream=False)` 和 `generate(stream=True)`；图片场景先在本地验证严格策略抛出
`SemanticLossError`，然后通过库提供的 `move_tool_images_to_user_message` 完成两种调用。
完整成功执行共 **98 次 API 请求**。两种 generate 模式均使用库的上游 SSE 路径。

推理回传另有 **6 个场景、24 次请求**，每项在严格 `raise` 策略下分别完成流式、汇总两轮工具调用。
工具调用和非空推理必须由第一轮真实 API 生成，再直接回传 MyLLM 返回的消息；不手工构造推理，
也不以空字符串与空字符串相等作为成功。检查 Responses 原生 reasoning item 和 Anthropic 原生
thinking / redacted_thinking 块，包括它们携带的签名或加密数据；Chat 检查完整字符串相等。
只读观察器记录 SDK 接收的原生事件和 SDK 序列化后的 HTTP 请求，比较不依赖 encoder 生成预期值。
报告仅保留比较结果和 SHA-256 摘要，原始思维与签名留在进程内存。两组合计 **40 个场景、122 次请求**。

原生对照与 MyLLM 请求使用不同且未调用过的 ID，避免服务端按旧 ID 恢复思维。
对照只复用 Profile 生成的请求参数，跳过消息转换与修复；HTTP 400/422 且错误指向思维或
工具关联时记为 `rejected`，正常完成记为 `accepted`。鉴权、限流和服务异常使场景失败，
不计为历史消息拒绝。报告保存原生对照、准备后的请求摘要、完成结果及失败尝试；
后续失败不会覆盖已经获得的对照证据。

测试断言输入对象不被修改、修复后的 ID 和结果数量正确，并检查实际调用能返回指定标记。
图片测试检查发送结构及 API 接受情况，不评估模型的图片识别准确率。
默认损失策略沿用 example 的 `ignore`，用于跨协议输出；图片场景显式使用预置 `hook`。
可选择 `raise` 验证严格转换，默认成功不表示原生推理、签名等元数据跨协议无损。

## 运行

DeepSeek 默认模型为 `deepseek-flash`，默认执行这 40 个场景；SenseNova 默认模型为
`sensenova-6.8-flash-lite`，默认还执行原有正文、工具往返和能力策略场景。

```sh
uv run python examples/deepseek_reasoning_e2e.py --validate
uv run python examples/sensenova_e2e.py --validate

uv run python examples/deepseek_reasoning_e2e.py --report /tmp/deepseek-inputs.json
uv run python examples/sensenova_e2e.py \
  --case reasoning_missing reasoning_null reasoning_empty missing_tool_result orphan_tool_result tool_result_images tool_result_image_placeholders reasoning_replay \
  --report /tmp/sensenova-inputs.json
```

`--validate` 不读密钥、不联网，只检查准备后的请求结构；报告的 `mode` 为 `offline`。
推理回传要求真实响应，不包含在 `--validate` 的 34 个场景中。真实运行的报告为 `live`。
可用 `--case reasoning_replay` 单独运行回传验证；`--upstream`、`--stdio` 可缩小范围。
SenseNova 默认每次请求前等待三秒；限流时增大 `--request-interval`，再补跑：

```sh
uv run python examples/sensenova_e2e.py --retry-failed \
  --request-interval 4 --report /tmp/sensenova-inputs.json
```

SDK 自动重试关闭。补跑保留已通过项和 `previous_attempts`，并检查模型、服务地址和运行模式一致，
不能将离线报告补跑为真实 API 结果。
普通 pytest 使用本地 HTTP transport 驱动官方 SDK，不读取真实密钥、不联网。

## 官方协议依据

- [DeepSeek 思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)：携带工具时的推理回传规则。
- [DeepSeek Responses](https://api-docs.deepseek.com/zh-cn/guides/responses_api/)：独立 function items、明文推理和图文工具输出。
- [DeepSeek Anthropic](https://api-docs.deepseek.com/zh-cn/guides/anthropic_api/)：调用与结果的关联规则，以及遗漏思维时可能按 ID 恢复的行为。
- [DeepSeek 模型列表](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)：默认模型和视觉能力。
- [SenseNova 文档](https://platform.sensenova.cn/docs)：三种接口的模型、参数、鉴权和工具调用格式。

协议说明用于构造对照；是否报错以当前 API 的真实响应为准。

## 2026-11-05 DeepSeek 实测

`deepseek-flash` 的新增矩阵 **34/34 通过，98 次真实请求**。默认输出损失策略为
`ignore`，图片用例为显式预置 `hook`；每项均完成汇总与流式调用。

| 原生 API 对照 | Chat | Responses | Anthropic |
| --- | --- | --- | --- |
| 思考模式下缺少 `reasoning_content` | 400 | 不属于 Chat 字段补齐用例 | 不属于 Chat 字段补齐用例 |
| 思考模式下 `reasoning_content=null` | 400 | — | — |
| 思考模式下 `reasoning_content=""` | 接受 | — | — |
| 缺少工具结果 | 400 | 400 | 400 |
| 结果引用不存在的调用 ID | 400 | 400 | 400 |

因此，“空思维会报错”需要区分字段缺失、`null` 与空字符串；MyLLM 当前补的是空字符串，
并非生成或恢复原始思维正文。修复后的请求全部通过本轮 API 调用。

额外使用 `raise` 冒烟时，曾有一次 `reasoning_null` 场景在输出转换阶段触发
`target protocol cannot preserve feedback_item alongside visible reasoning`。
这与输入字段补齐是不同的检查；当时默认策略通过，严格检查仍有上述失败。

## 2026-11-05 SenseNova 实测

`sensenova-6.8-flash-lite` 的新增矩阵最终 **34/34 通过**。首轮 22 项通过、12 项触发
RPM 限流；将请求间隔从 0.5 秒增至 4 秒后，仅补跑失败项并全部通过。
最终成功项包含 98 次请求，加上保留的失败尝试共 **122 次请求**。
两种图片来源的真实搬移和不支持视觉时的占位场景均通过汇总、流式调用。

| 原生 API 对照 | Chat | Responses | Anthropic |
| --- | --- | --- | --- |
| 思考模式下缺少 / `null` / 空字符串推理 | 接受 | 本轮未测试此类字段补齐 | 本轮未测试此类字段补齐 |
| 缺少工具结果 | 接受 | 接受 | 接受 |
| 结果引用不存在的调用 ID | 接受 | 400 | 接受 |

这些是本轮指定模型、配置和输入下的实际行为，不应将其他协议的拒绝规则套用到全部接口。
即使原生 API 接受异常历史，仍单独断言 MyLLM 准备后的补齐、丢弃、ID 配对及调用成功。
原生 API 接受不表示服务端保证了输入语义完整性。

本机报告为 `/tmp/myllm-deepseek-live-inputs.json` 与
`/tmp/myllm-sensenova-live-inputs.json`；SenseNova 报告保留首轮限流的 `previous_attempts`。
密钥未写入报告。完整离线回归为 **2560 passed**，包括官方 SDK HTTP 边界、错误分类、
失败证据保留、脱敏与补跑模式隔离检查。

## 2026-11-06 空推理字符串修复

Chat 流式 encoder 现在保留上游显式返回的 `reasoning_content=""`，与回传数据的读取保持一致。
默认配置下，中间层得到 `ReasoningContent(reasoning="", feedback_item="")`，
不再因 `None != ""` 在严格输出转换时误报语义损失。字段缺失或为 `null` 时仍不创建推理块。

回归通过本地 HTTP transport 驱动官方 SDK，覆盖严格 `raise` 策略、Chat / Middleware 输出、
流式 / 汇总调用，以及只有空字符串和空片段与非空片段混合的响应；真实 API 复跑结果见下文。
真正不同的非空回传文本和无法表达的反馈对象仍受语义损失检查约束。
此前真实失败未保留原始 SSE；本次修复针对已稳定复现的空值路径。

推理回传测试还补齐了 SenseNova Chat 的 `feedback_reasoning_item` Profile 路径，使其与可见推理
一同读取服务返回的 `reasoning`。标准 Chat 结果再次读入时，如果自定义回传字段未提供，
encoder 保留结果已有的原生 `reasoning_content`；显式的自定义值或 `null` 仍按 Profile 读取。
这不会生成推理或用空字符串替代原文。Anthropic 开启思考的 Profile 声明
`budget_tokens=1024`，对应输出上限为 2048，满足官方 SDK 的参数结构要求。

## 2026-11-06 真实 API 复跑与回传一致性

修正上述问题后，两家服务都重新完整执行了 34 个输入修复场景和 6 个推理回传场景，
下表中的完整运行均一次通过，无需补跑。

| 服务与模型 | 完整矩阵 | 真实请求 | 其中严格推理回传 |
| --- | --- | --- | --- |
| DeepSeek `deepseek-flash` | 40/40 | 122 | 6/6 |
| SenseNova `sensenova-6.8-flash-lite` | 40/40 | 122 | 6/6 |

每家服务的六个回传场景包含十二次完整两轮调用，覆盖 Chat、Responses、Anthropic 上游的
同协议与 Middleware 输出，以及流式和汇总模式。全部要求源推理非空，且原始响应、交付结果、
下一轮准备结果及下一轮实际 HTTP 回传逐项相等；不存在用补空结果通过一致性检查的情况。

DeepSeek 另外以 `raise` 复测 `reasoning_missing`、`reasoning_null`、`reasoning_empty`，
**3/3 通过，共 9 次请求**；缺失和 `null` 的原生对照仍返回 400，空字符串对照仍被接受，
MyLLM 修复后的汇总、流式调用均成功，未再出现此前的空值语义损失误报。

完整离线回归为 **2594 passed**，Ruff 与格式、diff 检查通过。
预检发现的 Profile 配置问题及其失败尝试保留在独立 replay 报告中；最终完整运行的报告分别为：

- `/tmp/myllm-deepseek-e2e-2026-11-06.json`
- `/tmp/myllm-sensenova-e2e-2026-11-06.json`
- `/tmp/myllm-deepseek-strict-2026-11-06.json`

密钥仅通过隐藏输入传入进程，未写入源码或报告。
