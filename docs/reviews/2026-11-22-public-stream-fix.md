# 公开流翻译与业务入口修复记录

修复基线：`fee286f34d05116509ca08d8c7d3bfaf97fa9101`。
修复提交：`a0d351ebb29689794c1e6dc7865c9b9427949e17`，2026-11-22。
问题来源：[PR #2](https://github.com/JJJJSTIYYYY/my-llm/pull/2)，感谢 @zhaibingye 提供问题和复现。
本次为独立实现，没有采用该 PR 的代码；原 PR 已关闭，未合并。

## 问题与根因

公开中间流保存的内容不足以让新的 `MyCodec` 独立恢复目标完整响应。
原生 Responses 的 item 身份、块索引和终态内容需要隐藏调用状态协助；
终态重建、实时事件和能力过滤使用不同的数据路径，容易产生缺失内容或索引不一致。
在下游扫描事件再拼装某种协议，只能补齐该次样例，不能修正公开翻译边界的契约。

问题跨越公开流交付与目标渲染，不能只按 Responses 一个完成事件处理。
目标协议过滤不支持块后需要使用自己的连续索引；同协议原生推理回传也需要保留来源，
避免被当作外来 opaque 数据误报损失。

## 最终修改

- `MyCodec.encode(chunks=...)` 接受完整原生流，累积并生成公开 `MessageEvent`；内部原生片段仍走私有路径。
- 成功 `finish` 同时携带完整 `messages`、索引化的 `output_blocks` 和 Encoder 明确提取的来源 `metadata`。
  `protocol` / `source_ref` 保留源关联，完整事件经 `to_wire()` 和 JSON 保存后可交给新的 `MyCodec.decode`。
- 公开 `decode` 检查完整流的入口结构，恢复源上下文，为本次翻译创建独立的目标索引映射。
  它依赖调用方保留事件和终态数据的一致性，不负责恢复外部主动删改后的数据。
- Responses 统一实时地址和终态地址映射；Anthropic 在目标内容过滤后分配索引。
  原生空 item 仍由来源 metadata 保留，与被过滤掉的非空 item 区分。
- 生成只以 `MyLLM.generate` 为业务入口，独立翻译以 `MyCodec.encode` / `decode` / `convert` 为入口。
  删除顶层 `generate` 包装和内部组件的便捷重导出；内部组件按完整流程的前置条件工作。

实现依据：[公开翻译入口](../../src/my_llm/codec/converter.py)、
[公开流模型](../../src/my_llm/middleware/stream.py)、[终态累积](../../src/my_llm/runtime/state.py)、
[Responses Decoder](../../src/my_llm/codec/openai/responses/decoding/decoder.py)及
[Anthropic Decoder](../../src/my_llm/codec/anthropic/decoding/decoder.py)。
当前完整设计见[中间协议](../design/protocol-middleware.md)和[流状态](../design/stream-state.md)。

## 验证与后续边界

相关契约测试见[公开翻译](../../tests/contract/test_public_translation.py)和
[原生流投影](../../tests/contract/test_native_stream_projection.py)，覆盖独立 codec、
typed / wire / JSON 流、同协议投影、跨协议及过滤后的地址。
修复后的常规验证和真实服务结果汇总见[测试结果目录](../testing/results/README.md)。

随后专项审查记录的 A01 空 item 顺序、A02 摘要展示分段，以及外部破坏数据和非业务非法值，
不属于本次修复对所有原生细节完全保真的承诺。
[后续审查](2026-11-22-codec-audit.md)已逐项说明证据和不修复决定；
其中 A02 的正式结果与下一轮实际 HTTP 回传保持一致，不能将展示分段差异当作缓存输入变化。
