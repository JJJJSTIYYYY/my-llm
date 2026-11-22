# 2026-11-06 空推理字符串修复记录

本页迁录既有修复说明，描述当时的触发条件、实现意图与验证边界；不是本次文档整理新增的代码改动。
对应历史测试结果见 [2026-11 输入与推理回传测试结果](../testing/results/2026-11-input-e2e.md)。

当时的修复使 Chat 流式 encoder 保留上游显式返回的 `reasoning_content=""`，与回传数据的读取保持一致。
默认配置下，中间层得到 `ReasoningContent(reasoning="", feedback_item="")`，
不再因 `None != ""` 在严格输出转换时误报语义损失。字段缺失或为 `null` 时仍不创建推理块。

回归通过本地 HTTP transport 驱动官方 SDK，覆盖严格 `raise` 策略、Chat / Middleware 输出、
流式 / 汇总调用，以及只有空字符串和空片段与非空片段混合的响应；真实 API 复跑结果见前述测试结果链接。
真正不同的非空回传文本和无法表达的反馈对象仍受语义损失检查约束。
此前真实失败未保留原始 SSE；本次修复针对已稳定复现的空值路径。

推理回传测试还补齐了 SenseNova Chat 的 `feedback_reasoning_item` Profile 路径，使其与可见推理
一同读取服务返回的 `reasoning`。标准 Chat 结果再次读入时，如果自定义回传字段未提供，
encoder 保留结果已有的原生 `reasoning_content`；显式的自定义值或 `null` 仍按 Profile 读取。
这不会生成推理或用空字符串替代原文。Anthropic 开启思考的 Profile 声明
`budget_tokens=1024`，对应输出上限为 2048，满足官方 SDK 的参数结构要求。
