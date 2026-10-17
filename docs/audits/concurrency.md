# MyLLM 并发调用验证

2026-10-17，使用真实官方 SDK 和本地可控 HTTP transport 验证同一 MyLLM 实例的异步并发。
[测试文件](../../tests/integration/test_concurrency.py)覆盖五种上游协议和五种 I/O 协议。

## 正常并发

25 组协议组合各发起 8 路并发请求，每组混合 4 路流式和 4 路非流式调用，共 200 次调用。
响应体在每个原生事件前通过 asyncio.Barrier 汇合，保证 8 路都已进入读取、事件片段交错进行；
不依赖 sleep 调度碰巧发生重叠。15 秒超时用于让意外串行执行产生明确失败。

每路使用不同的输入文本、请求 header、输出限制、返回文本、工具参数和 token 用量，
故意复用相同工具名称及原生调用 ID，验证请求状态不会相互污染。断言包括：

- 每路最终消息及逐块输出中的文本、工具名称和完整参数正确；逐块输出重新独立累积检查。
- token 计数与本路请求对应，Responses 序号从 0 开始且连续。
- 请求中的输入、header 与输出限制对应正确，共享工具声明和原始历史没有被修改。
- 未显式指定 Chat I/O 的实例能够在并发首次调用时绑定协议。
- 每次调用拥有独立 client，正常完成后 client 与 HTTP 响应体均关闭。

## 故障隔离

五种上游分别在同一实例上开启 5 路流，并在读到第一段文本后暂停上游：
一路提前退出、一路取消、一路抛出传输读取异常，其余两路继续等待。
先检查前三路已经退出、其余两路未被关闭，再释放正常请求并确认结果正确。
最后在同一实例追加一次非流式调用，确认故障未污染后续请求。此部分共 30 次调用。

提前退出、取消、失败的 stream.result 仍为 None；所有请求结束后，HTTP 响应体与 client 均已关闭。
原先仅覆盖 4 路 FakeClient 且通过 sleep(0) 交错的测试已由这些真实 SDK 集成测试替代。

## 发现与修复

本地 google-genai 1.75.0 的流由多层异步生成器包装。提前退出时关闭外层生成器及
SDK client，不能保证悬挂的 HTTPX 响应体在退出当下已经关闭；测试直接捕获了该问题。

[Gemini Decoder 传输实现](../../src/my_llm/codec/gemini/decoding/request/transport.py)通过 SDK 公开 HttpOptions.async_client_args
配置 HTTPX response hook，在每次请求自己的 AsyncExitStack 中登记响应体清理。
退出时显式关闭响应体，再关闭异步和同步 SDK client；清理不依赖垃圾回收，
多个响应或重试也分别登记。没有修改 SDK，也没有恢复公开的自定义 client 接口。
SDK HTTP 配置与资源关闭依据见[官方 SDK 文档](https://googleapis.github.io/python-genai/)。

## 运行与边界

```bash
.venv/bin/pytest -q tests/integration/test_concurrency.py
```

专项共 30 项测试、230 次调用。完整回归 714 项通过。
这些结果验证单个 asyncio 事件循环内的并发隔离及资源生命周期；未调用远端模型，
不用于推断供应商限流、真实网络吞吐、最高并发容量或跨线程使用情况。
