# 2026-11-25 无消息缓存连续请求矩阵

## 基线与环境

- 源码基线：`d20e3dd`，在当前工作区运行。
- 开始前已有未提交修改：`src/my_llm/codec/openai/chat/prebuilt/` 下的 `hook.py`、
  `reasoning.py`、`tool_images.py`。初次矩阵及首轮补偿补验保留这些修改；后续图片哈希修改见下文。
- Python 3.12.12，OpenAI SDK 2.54.0，Anthropic SDK 0.125.0，httpx 0.28.1，pytest 9.1.1。
- 使用本地官方 SDK HTTP 模拟及合成响应，无真实模型调用、密钥或供应商缓存命中观测。
- 范围、回填步骤和复现命令统一见[测试方法](../history-prefix.md)。

## 初次结果：哈希修改前

主矩阵每格 40 个用例。下表记录“原始消息前缀一致的用例数／已测用例数”，不是 pytest
通过率；检测到预期的不一致也有明确断言。

| 输入 stdio | Chat 上游 | Responses 上游 | Anthropic 上游 |
| --- | --- | --- | --- |
| middleware | 36/40 | 40/40 | 40/40 |
| openai_chat | 40/40 | 40/40 | 40/40 |
| openai_responses | 36/40 | 40/40 | 40/40 |
| anthropic | 36/40 | 40/40 | 40/40 |

主矩阵 **480 passed**：468 个用例前缀一致，12 个图片补偿用例前缀不一致。
额外推理回填 **72 passed**，前缀均一致；主矩阵及推理回填合计 **552 passed**，覆盖 2,688 次本地 HTTP 请求。

不一致均为 `tool-images` 场景：其他三种输入携带工具图片，关闭消息缓存，显式启用 Chat
补偿 hook。每轮重新翻译会生成不同的图片引用 ID，并将图片消息移动到当前历史尾部。
第二轮对同一完整历史的重试也不相等。四种流式／非流式顺序均复现。
这是哈希修改前的补偿行为；测试直接保留原始差异，没有忽略失败或归一化 ID。

Chat 原生工具消息不能携带图片，该输入行使用工具文本结果加用户图片，不能用该行结果
推断原生 Chat 支持工具结果图片。其他格使用默认语义损失 `raise`，没有全局忽略损失。

初次矩阵没有改变 hook 的图片引用生成或搬移规则。除这一明确边界外，已测的完整历史重翻译、
同历史重试、流式 finish 消息回填和可原生回放的推理策略均保持前缀一致。

## 补验：所有补偿 hook 从首轮触发（哈希修改前）

按“第一轮已触发补偿，以补偿后的实际请求为基线，第二轮比较完整消息前缀”的判据，
追加 60 个用例。每组覆盖 middleware、Responses、Anthropic 三种输入及四种调用顺序。
第二轮仅追加首轮返回消息和用户消息，不新增待补偿内容；流式回填仍取 `finish.messages`。

| hook 与首轮补偿内容 | 前缀一致 | 前缀不一致 |
| --- | --- | --- |
| `reasoning_content_feedback`：推理 | 12 | 0 |
| `move_tool_images_to_user_message`：工具图片 | 0 | 12 |
| `chat_semantic_loss_hook`：仅推理 | 12 | 0 |
| `chat_semantic_loss_hook`：仅工具图片 | 0 | 12 |
| `chat_semantic_loss_hook`：推理加工具图片 | 0 | 12 |

补验 **60 passed**：24 个用例前缀一致，36 个图片补偿用例前缀不一致。
测试确认每个 hook 首轮已调用，且补偿后的消息已发送到 SDK HTTP 边界；
没有把“第二轮首次补偿”或“补偿前后草稿不同”当作前缀变化。
所有不一致均如实保留，生产补偿逻辑没有改动。
加上本次补验，前缀测试共 612 个用例、2,808 次本地 HTTP 请求。

## 哈希修改前的回归验证

| 命令 | 结果 |
| --- | --- |
| `pytest -q tests/integration/test_uncached_history_prefix.py`（补验前） | 552 passed |
| `pytest -q tests/integration/test_uncached_history_prefix.py -k compensation_from` | 60 passed, 552 deselected |
| `pytest -q`（补验后） | 3379 passed，含全部 612 个前缀测试；补验前为 3319 passed |
| `pytest -q tests/integration/test_documentation.py` | 24 passed |
| `ruff check src tests examples` | 通过 |
| `ruff format --check src tests examples` | 203 files already formatted |
| `git diff --check` | 通过 |

以上使用仓库 `.venv/bin/` 中的命令执行。未重跑真实供应商 E2E 或性能基准。

## 后续修改：图片来源使用稳定哈希

按后续要求，将工具图片的随机 UUID 引用改为图片来源的 SHA-256。
来源字段及哈希规则统一见[语义损失指南](../../guides/semantic-loss.md)。
相同来源跨输入协议、实例、工具回合、缓存清理、关闭缓存后的重试均使用相同引用；
相同图片重复出现时复用引用，每次图片出现仍保留，未新增图片去重或缓存。

图片消息仍追加到当前历史末尾，本次没有修改消息搬移位置。矩阵及首轮补偿补验仍分别有
12 个、36 个图片场景的完整前缀不一致，原因现在仅为图片消息在新增对话后的相对位置变化。
测试已强化为要求两轮的图片引用及图片消息完全一致、图片消息之外的旧历史前缀一致。
同一完整输入的重试现在要求整个请求完全一致。

更新了原有缓存、新工具回合、重复图片和跨协议测试，并追加 9 个用例验证跨实例及模型视觉能力的
哈希一致性，以及 URL、Base64 数据或媒体类型变化只改变对应图片的引用。

哈希修改后全量执行 `pytest -q`：**3388 passed**，包含全部 612 个前缀用例。
`ruff check src tests examples`、`ruff format --check src tests examples`（203 个文件）及
`git diff --check` 均通过。仍使用本地 SDK HTTP 模拟，没有真实供应商请求。
