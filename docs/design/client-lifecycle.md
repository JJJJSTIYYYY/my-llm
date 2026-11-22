# SDK 客户端生命周期

每次 `generate` 独立拥有 SDK 客户端、响应流和累积状态。MyLLM 实例保存配置及可选的消息翻译缓存，不缓存客户端。
流式与收集式调用使用同一条执行路径；同实例的不同请求也分别创建客户端。

## 创建与关闭

构造 MyLLM 时解析、校验并复制 provider / model 配置，准备连接参数，不创建客户端。
调用级 api_key 由 Decoder 应用到本次调用的连接参数，不修改实例配置。
流式调用在真正开始消费响应时创建客户端；未消费就关闭的流不创建客户端。
收集式调用在返回完整结果前消费并关闭同一条上游流。

每次执行使用独立的 AsyncExitStack，创建客户端后立即登记 Transport.close_client，
再进入上游响应上下文。完整消费、提前关闭、异常和取消时，先关闭响应，再关闭客户端。
客户端构造失败时没有已创建的客户端需要关闭；后续调用仍可创建新客户端。
请求失败只影响本次响应和客户端。

同实例并发调用拥有独立客户端和流状态。关闭消息缓存时支持同实例并发；
启用 `enable_message_cache=True` 后，应先完成或关闭前一条流再调用下一次生成。
实例复用和跨会话缓存隔离由应用层负责；开始新会话前可调用 `clear_cache()`。

MyLLM 实例没有关闭状态，也不需要上下文管理器。消息缓存通过 `clear_cache()` 主动清理，或随实例释放。
`MyLLM.generate(stream=True)` 返回的流通过 `async with` 或 `aclose()` 管理；包根不提供独立 `generate` 函数。

逐请求客户端让调用级密钥、响应状态和关闭顺序具有明确所有者，避免共享连接身份与实例关闭状态互相约束。
其成本是每次创建客户端；应用复用的是 MyLLM 配置和可选历史，而不是底层 SDK 客户端。

## 协议边界

Encoder 验证输入，Decoder 准备 client / request / local 字典并负责请求参数、过滤和验证。
MyLLM 调用 Transport.create_client(client) 创建本次 SDK 客户端，
Transport.upstream(client, request) 提供响应上下文，Transport.close_client(client) 关闭客户端。
Transport 不接收 Profile 或原始生成控制项，不自行维护客户端池。

| 协议 | SDK 客户端 | 客户端关闭 | 响应关闭 |
| --- | --- | --- | --- |
| OpenAI Chat / Responses | AsyncOpenAI | await client.close() | SDK stream.close() |
| Anthropic | AsyncAnthropic | await client.close() | SDK stream.close() |

[生命周期测试](../../tests/unit/test_client_lifecycle.py)覆盖逐请求创建、正常关闭、未消费流、提前退出、失败和取消。
[真实 SDK 并发测试](../../tests/integration/test_concurrency.py)验证不同响应和客户端的隔离。
[依赖边界测试](../../tests/contract/test_codec_boundaries.py)验证 SDK 执行仍由 Transport 拥有。
