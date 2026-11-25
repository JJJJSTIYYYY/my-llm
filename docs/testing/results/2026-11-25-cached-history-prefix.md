# 2026-11-25 开启消息缓存的连续请求矩阵

基线为 `a460228`，工作区将原无缓存测试参数化为缓存开启／关闭，并重命名为
`tests/integration/test_history_prefix.py`。生产代码未修改。
Python 3.12.12、OpenAI SDK 2.54.0、Anthropic SDK 0.125.0、httpx 0.28.1、pytest 9.1.1。
使用官方 SDK 配本地 HTTP transport 和合成响应，不访问真实供应商。
复现步骤及验收判据统一见[测试方法](../history-prefix.md)。

## 开启缓存的结果

每格 10 种场景 × 4 种流式／非流式顺序。数字是“全部连续请求前缀一致的用例数／已测用例数”。

| 输入 stdio | Chat 上游 | Responses 上游 | Anthropic 上游 |
| --- | --- | --- | --- |
| middleware | 40/40 | 40/40 | 40/40 |
| openai_chat | 40/40 | 40/40 | 40/40 |
| openai_responses | 40/40 | 40/40 | 40/40 |
| anthropic | 40/40 | 40/40 | 40/40 |

主矩阵 480/480、额外推理策略 72/72、首轮触发全部补偿 hook 的补验 60/60，
**开启缓存共 612/612，已测消息前缀一致率 100%**，涉及 2,808 次本地 HTTP 请求。
四轮场景检查每轮对全部前轮的原始消息前缀，合计 3,372 次前缀比较全部相等；
主矩阵还覆盖 480 次同输入重试，完整请求对象全部相等。

关闭缓存时保留[先前记录](2026-11-25-history-prefix.md)的图片消息位置差异，
开启缓存后，图片补偿及组合补偿的全部场景前缀一致。首轮补偿后的图片消息保存在缓存原位置，
后续只追加新增消息；首轮补偿补验还确认第二轮没有再次调用 hook。

这里按缓存契约使用增量输入：首轮提交初始历史，后续提交上一轮返回消息与工具结果或新用户消息。
流式取 `MessageEvent.finish.messages`，不读取 `stream.result`；工具调用与对应结果放在同一批次。
请求的工具声明保持一致，Anthropic 独立 `system` 也检查一致。

这是本地请求结构的前缀一致率。没有供应商返回的真实 `cached_tokens`、计费、缓存阈值、
TTL 或命中结果，不能将 100% 解读为真实供应商缓存命中率。

## 验证

`pytest -q tests/integration/test_history_prefix.py`：**1224 passed**，包含缓存开关各 612 个用例。

全量 `pytest -q`：**4000 passed**。`ruff check src tests examples`、
`ruff format --check src tests examples`（203 个文件）、`git diff --check` 均通过。
