# SDK 客户端生命周期

MyLLM 实例拥有官方 SDK 客户端；请求只拥有响应流与累积状态。
同一实例的串行、并发、流式及收集式生成复用客户端，不在每次请求结束时重建连接池。
此管理停留在官方 SDK 层，不自行创建或配置 HTTPX 连接池。

## 创建、复用与关闭

构造 MyLLM 时解析、校验并复制 provider / model 配置，不创建 SDK 客户端。首次真正消费响应流时，
MyLLM 从 BaseCodec 统一组合的 Transport 获取 create_client，保存客户端到实例缓存，
并登记 Transport.close_client 到实例的 AsyncExitStack。
客户端创建和缓存登记均为同步操作，首次并发请求之间没有 await 切换点，
因此不需要用异步锁包住请求或 SDK 流。

生成参数错误在创建客户端前报错；未开始消费的流不会创建客户端。
客户端构造失败时不写缓存，后续可以重试。请求创建或读取失败时保留客户端，
关闭本次响应；SDK 的重试和后续请求继续使用相同实例。
不检测用户直接修改 MyLLM 内部属性引起的配置失效。

使用 `async with MyLLM(...) as llm` 或 `await llm.aclose()` 关闭实例。
关闭前先完成或关闭所有响应流；关闭操作幂等，关闭后拒绝新请求和未启动的流。
SDK 客户端按实例及其事件循环使用，不跨事件循环共享。
实例退出时按资源栈顺序关闭所有缓存客户端；单个关闭失败仍会尝试其余清理回调。

独立函数 generate 创建临时 MyLLM。收集式调用在返回或异常时关闭临时实例；
流式调用将清理回调交给 MessageStream，完整消费、提前关闭、取消或异常都会释放临时实例。
关闭未启动的独立流同样不会创建 SDK 客户端。

## 协议边界

MyLLM 调度 Transport 的客户端接口并管理缓存；客户端实现类、请求执行和关闭方式由 Transport 决定。
BaseCodec 组合 Encoder、Decoder 和 BaseTransport 三个独立组件。
Encoder / Decoder 负责已识别字段翻译；Decoder 映射生成选项并组织原生请求结构。
参数验证、SDK 字段过滤及原生结构校验统一由 Transport 处理。
Transport.create_client 创建客户端，upstream 显式借用 client 执行原生请求；
close_client 默认 await client.close()，client_key 默认 None。
SDK 的连接约束和协议配置由 Transport.validate_profile 校验；BaseCodec.validate_profile 委托 Transport。
Codec 和 Transport 均保持无状态，实际客户端仍由 MyLLM 缓存，
消息翻译仍经过 Encoder → 中间态 → Decoder。

| 协议 | SDK 客户端 | 实例关闭 | 单次请求关闭 |
| --- | --- | --- | --- |
| OpenAI Chat / Responses | AsyncOpenAI | await client.close() | SDK stream.close() |
| Anthropic | AsyncAnthropic | await client.close() | SDK stream.close() |
| Gemini | genai.Client | await client.aio.aclose()，finally client.close() | 生成器和本次响应体 |
| Ollama | AsyncClient | await client.close() | SDK 生成器 aclose() |

Ollama 的公开 chat 参数没有逐请求 headers 入口，headers 只能在 SDK 客户端创建时绑定。
因此其 client_key 使用 profile 与调用选项合并后的请求头：相同请求头复用同一 SDK 客户端，
不同请求头各自缓存，直到实例关闭。动态请求头组合越多，缓存客户端越多；
不通过临时改写共享 headers 或访问 SDK 私有 HTTPX 属性实现逐请求覆盖。
其他 SDK 将动态请求头放在请求参数中，不因生成参数或请求头变化重建客户端。

## Gemini 响应隔离

共享 SDK 客户端时，HTTP response hook 不能再捕获某一次请求的资源栈。
hook 通过 ContextVar 取得当前 SDK 读取对应的请求栈，登记本次响应体的 aclose。
重试产生的多个响应体也分别登记。

上下文只包围 SDK 流创建和每次 anext，不跨越向调用者 yield 的位置。
这样响应流可以在一个任务中读取，再从另一任务关闭或继续读取，
不会复位另一个上下文的 token，也不会清理相邻请求的响应体。
每个请求仍显式关闭 SDK 生成器及其底层响应体，保留提前退出时的即时清理保证。

## 验证与依据

[客户端生命周期测试](../../tests/unit/test_client_lifecycle.py)覆盖延迟创建、重复使用、实例隔离、
幂等关闭、构造失败重试、请求失败后复用、关闭后拒绝请求，以及独立流的资源所有权。
[组件契约测试](../../tests/unit/test_native_codecs.py)检查 BaseCodec 必须组合 Transport、
Decoder 不再提供 SDK 方法，以及配置校验的统一协调；
[依赖边界测试](../../tests/contract/test_codec_boundaries.py)防止转换实现与 Transport 相互导入。
[并发测试](../../tests/integration/test_concurrency.py)用真实 SDK 和本地 HTTP transport 验证：

- 五协议各有六路首次并发混合调用，响应在同步屏障汇合；SDK 客户端只创建一次，后续调用仍复用。
- 原有 25 种上游 / I/O 组合继续检查内容、用量、生成选项和请求头隔离。
- 一路提前退出、一路取消、一路失败时，其他响应和共享客户端保持打开，后续仍可生成。
- 请求在不同任务间读取与关闭时，只清理对应响应，直到 MyLLM 退出才关闭客户端。

本地测试验证客户端创建次数和资源隔离，不将模拟传输耗时当作真实网络性能基准。
2026-10-18 同时检查了安装的官方 SDK 实现及以下官方资料：

- [OpenAI 官方示例](https://developers.openai.com/cookbook/examples/agents_api/apps/data_analyst/readme)：应用生命周期结束时 await client.close()。
- [Anthropic Python SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python#managing-http-resources)：客户端的显式关闭和上下文管理。
- [Google Gen AI SDK](https://googleapis.github.io/python-genai/)：同一客户端多次调用，以及分别关闭同步与异步客户端。
- [Ollama 官方 SDK 源码](https://github.com/ollama/ollama-python/blob/main/ollama/_client.py)：AsyncClient.close 与 chat 的公开参数。
