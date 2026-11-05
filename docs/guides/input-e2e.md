# 输入修复与工具图片端到端测试

[SenseNova 脚本](../../examples/sensenova_e2e.py) 和
[DeepSeek 脚本](../../examples/deepseek_reasoning_e2e.py) 共用
[输入用例](../../examples/e2e_inputs.py)，直接构造 Chat、Responses、Anthropic 原生字典
和类型化 Middleware 消息，不通过被测转换器生成测试输入。

接口地址、鉴权、模型、输出上限、思考开关、参数路径和能力声明均由
`ProviderProfile` / `ModelProfile` 管理。example 不手工补齐思维、工具结果或删除孤立消息，
也不替换 Transport。API 密钥只在运行时读取，不进入源码、报告或命令参数。

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

共 **34 个场景**。每个历史修复场景先发一次未经消息修复的原生 API 对照，再执行
`generate(stream=False)` 和 `generate(stream=True)`；图片场景先在本地验证严格策略抛出
`SemanticLossError`，然后通过库提供的 `move_tool_images_to_user_message` 完成两种调用。
完整成功执行共 **98 次 API 请求**。两种 generate 模式均使用库的上游 SSE 路径。

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

DeepSeek 默认模型为 `deepseek-flash`，默认仅执行这 34 个输入场景；SenseNova 默认模型为
`sensenova-6.8-flash-lite`，默认还执行原有正文、工具往返和能力策略场景。

```sh
uv run python examples/deepseek_reasoning_e2e.py --validate
uv run python examples/sensenova_e2e.py --validate

uv run python examples/deepseek_reasoning_e2e.py --report /tmp/deepseek-inputs.json
uv run python examples/sensenova_e2e.py \
  --case reasoning_missing reasoning_null reasoning_empty missing_tool_result orphan_tool_result tool_result_images tool_result_image_placeholders \
  --report /tmp/sensenova-inputs.json
```

`--validate` 不读密钥、不联网，只检查准备后的请求结构；报告的 `mode` 为 `offline`。
真实运行的报告为 `live`。`--upstream`、`--stdio` 和 `--case` 可缩小范围。
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
这与输入字段补齐是不同的检查；默认策略下通过不代表该严格输出问题已修复。
本轮未修改生产代码来处理该输出问题。

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
