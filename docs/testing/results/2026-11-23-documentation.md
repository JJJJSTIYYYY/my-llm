# 2026-11-23 文档整理验证

本轮仅重组、核对文档，不修改实现与测试，不访问真实模型服务，也不重新运行性能测量。

## 基线与环境

- HEAD：`a0d351ebb29689794c1e6dc7865c9b9427949e17`。
- 运行时源码指纹：`c4ec8f16e9cdd69f7ea8eb487919dab539c8d756a5085b87cf31a74f3ca21dc5`，
  按现有 `tests.performance.fingerprint.source()` 计算，包含工作区已有的版本文件改动。
- Python 3.12.12；macOS 27.2 arm64；OpenAI 2.54.0；Anthropic 0.125.0；
  Pydantic 2.14.1；HTTPX 0.28.1；pytest 9.1.1。

## 常规回归与文档检查

```sh
.venv/bin/python -m pytest -q
.venv/bin/python docs/reference/schema/verify.py
.venv/bin/ruff check src tests examples
.venv/bin/ruff format --check src tests examples
git diff --check
```

完整既有测试 **2764 passed in 14.72s**，包含文档与发行包测试。
Schema 独立校验通过：16 个 Schema、38 个模板、14 个流文件、81 帧。
Ruff 检查通过，201 个文件符合格式；差异空白检查通过。
38 份 Markdown 的 314 处本地链接及锚点有效，43 个 Python 代码块编译通过；
所有文档均可从总索引到达。262 个既有非 Markdown 文件的任务前后哈希一致，
其中三份历史性能 JSON 仅移动路径；原有版本文件改动未被本轮修改。
仓库 CI 的旧导出断言不由本地 pytest 执行，其差异仍按末尾的审查链接保留。

## 离线 E2E 预检

```sh
.venv/bin/python examples/sensenova_e2e.py --validate --request-interval 0 --report /tmp/myllm-docs-sensenova-offline-2026-11-23.json
.venv/bin/python examples/deepseek_reasoning_e2e.py --validate --request-interval 0 --report /tmp/myllm-docs-deepseek-offline-2026-11-23.json
```

两份报告均为 `mode=offline`、**34/34 通过、0 次服务请求**。
只验证当前 Profile 与请求准备，不包含真实源推理、服务接受情况或回传一致性。
真实服务的既有结果见 [2026-11-22 SenseNova](2026-11-22-sensenova.md)，本轮未补跑。

## 性能文档核对

核对现有性能入口的参数帮助，以及 `tests.performance.__main__`、`runner`、
`request_chain` 和完整性测试的场景构造：当前主矩阵为 **native 6 + cross 18 = 24**。
Middleware stdio 由独立成对基准覆盖。性能指南已同步这一范围与逐请求 SDK 客户端计时。
这里只核对入口与计时实现，没有生成当前版本性能结果。

历史三份 JSON 已移至 `docs/testing/performance/data`，并逐个与 HEAD 原路径的文件字节比较，
全部相同：

| 文件 | SHA-256 |
| --- | --- |
| `2026-11-04-parameter-events.json` | `01c434e8b24b4bc3f64b8824654a975029b8e58474c4c78d24e6d09007baa28e` |
| `2026-11-04-reasoning-content.json` | `d133d195a7a27c57bc50fe625914abe8f08a788f442f42c351bbca5ea286aca4` |
| `2026-11-06-message-cache-copy.json` | `b16af56129de6dafd7c4ca668a419e7df1aa4db40bd40f55a20b3a0795c1e7e7` |

本轮另发现 CI 隔离安装仍引用已移除导出，已在 [对齐审查记录](../../reviews/2026-11-23-documentation-alignment.md)
记录，未据此改动 CI 或测试。当前公开接口的手工验证见 [发行包验证](../packaging.md)。
