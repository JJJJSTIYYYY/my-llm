# 流状态与块身份

复核日期：2026-10-17。依据当天读取的官方在线文档，重构
[`StreamState`](../../src/my_llm/stream/_state.py) 的块身份、累积和发布规则。
公开接口继续返回各协议的标准字典；下述操作仅属于内部 Codec 边界。

## 官方依据

| 协议 | 官方文档与本次使用的语义 |
| --- | --- |
| OpenAI Chat | [流事件](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)：工具片段按调用 index 累积；choice 的 finish_reason 与最终独立 usage 块分开处理。 |
| OpenAI Responses | [流事件](https://developers.openai.com/api/reference/resources/responses/streaming-events)：output_index 标识输出项，content_index / summary_index 标识文本或思考摘要部分；完成响应提供最终 output。 |
| Anthropic | [Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming)：内容块按 index 开启、增量更新和关闭；签名在块关闭前到达；display=omitted 时 thinking 文本可以为空。 |
| Gemini GenerateContent | [生成接口](https://ai.google.dev/api/generate-content)、[Thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures)：Part 上的签名应按原样保留，签名可能出现在空文本 Part；独立 Part 不按文本相等关系合并。 |
| Ollama | [Chat](https://docs.ollama.com/api/chat)、[Streaming](https://docs.ollama.com/capabilities/streaming)、[Tool calling](https://docs.ollama.com/capabilities/tool-calling)：分别累积 content、thinking 和工具调用，done 表示响应完成。 |

Gemini 的范围仍是 `generate_content_stream`。官方思考文档的部分旧链接目前指向
Interactions；本次查阅 GenerateContent 专属文档，没有把 Interactions 的 step/index
规则套到当前实现，也没有迁移 SDK 或接口。

## 操作边界

Encoder 无状态，只翻译本次传入的原生事件。流控制操作继续存放在内部保留命名空间，
不借用原生扩展的 type、index 或 events 字段。

| 操作 | 含义 |
| --- | --- |
| block_set | 按源引用创建或以完整内容替换一个块，例如 Anthropic 的开始块、Responses 的完成块。 |
| block_update | 按源引用更新块元数据，例如工具名称、ID 与函数扩展。 |
| block_delta | 向已引用的块追加 text、thinking、refusal、签名或 arguments 片段；是否发布文本由块的文本字段决定。 |
| block_annotation | 向引用块的列表字段追加注释，保留到最终内容；需要时在块生命周期内发布。 |
| start | 源协议显式开始响应；在增量前生成目标开始帧。 |
| block_append | 追加独立完整块，不携带源引用，例如 GenerateContent Part、Ollama 完整工具调用。 |
| block_stop | 源协议关闭指定块。 |
| snapshot | 保存最终消息，并使用与实时操作一致的源引用协调每个块。 |
| metadata / usage / reason / finish / opaque | 更新响应级信息、标记协议终态，或透传未知原生事件。 |

`BlockRef(scope, index, part)` 使用固定结构表示源地址；所有累积器映射键都归一为此类型。
NamedTuple 在中间字典中可直接 JSON 序列化，反序列化后的列表在累积器入口还原。
输出块位置来自请求内块列表，与源引用分离。内部发布事件携带 source_ref；
Responses Decoder 据此保持原生父 item 身份与内容/摘要索引，不把多个 part 改成多个加密 item。没有 `index=None` 约定，
也没有 `part_n`、`complete_n` 或 `final_n` 合成键。

`BlockState` 收拢内容、片段列表、工具参数列表和发布/关闭状态。
控制字段存放在原生内容字典之外；参数片段不会被同名扩展覆盖。
文本、签名和 JSON 参数使用列表累积，在需要快照或关闭时拼接。

每个块缓存当前内容版本的物化结果。替换、元数据更新、增量和注释会使缓存失效；
嵌套修改按需重建容器，已发布的旧版本保持稳定。物化结果在请求内部只读共享，
公开协议输出则在 Decoder 边界隔离其可变内容。

## 快照与生命周期

最终块按源引用匹配，而非按类型加文本内容搜索。相同文本的两个块可以保有不同引用信息。
每个摘要部分按原生 summary_index 匹配，最终消息仍保留完整的原生 reasoning item。

快照可以补齐尚未流出的块或文本后缀。若它否定已经发布的文本前缀，或遗漏已累积的块，抛出
`IncompleteStreamError`，避免对无法撤回的流事件报告成功。

文本块的开启不依赖非空增量；空 thinking 仍能收到签名并正常关闭。
不透明完整块按发现顺序发布。工具调用在收到协议终态且上游迭代结束、所有参数验证通过后
统一发布；其后出现的块一并暂存，以保留工具与后续内容的顺序。这是库的发布策略，
并非上游协议要求；代价是这些后续块不会即时输出。

EOF 本身不代表成功，必须收到源协议的终止操作。重复原生终态或终态后继续输入内容都会报错。
源 Encoder 可声明终态后允许的操作类型，保留 Chat 的用量尾块及 Gemini 的尾部元数据；
这项选择不在共享状态机中按厂商名判断。
`finished` 仅在验证及完成事件生成成功后设置。失败的部分消息不会解析未完成工具 JSON，
也不会把它当作可执行调用返回。原生 failed/incomplete 事件仍由 Encoder 抛出既有异常。

成功终态先物化并验证全部块，再发布工具与完成事件。同一版本的工具 JSON 只解析一次；
完成事件、最终消息转换和状态读取复用这份结果。`messages()` 返回独立副本，
`MyLLM` 内部直接使用完成事件已有的消息，避免再次复制中间终态。

## 验证

[`test_stream_state.py`](../../tests/contract/test_stream_state.py) 使用独立编写的原生事件覆盖：

- 五协议中间事件经 JSON 往返后，逐块或批量输入的结果一致。
- 改变同一文本和工具参数的分片方式，Unicode、嵌套 JSON 和结果保持一致。
- 重复文本的引用信息按输出项及内容部分隔离。
- 最终快照补齐多个缺失块和后缀；相互矛盾的已发布文本、遗漏已有块的快照报错。
- 空 thinking 签名、redacted_thinking、独立空签名 Part 保留。
- 工具先于文本时，延迟发布、原始顺序和目标原生结构同时成立。
- 多个 reasoning summary 部分不混用身份；工具验证失败不会设置完成状态。

既有协议 schema、25 对转换、扩展保留、官方 SDK 本地 transport、并发隔离和资源关闭
测试继续覆盖完整路径。测试不调用远端模型服务。

[`test_stream_materialization.py`](../../tests/contract/test_stream_materialization.py) 还验证
物化版本失效、工具参数单次解析，以及明确来源协议的 25 对公开输出不会反向修改缓存。

完整修复及最新 Schema 复核见[五协议验收记录](../audits/protocol-fixes-2026-10-17.md)。
