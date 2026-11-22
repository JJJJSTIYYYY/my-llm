# 测试文档

本部分面向复现与结果核对，保存测试方法、判据和带日期的结果记录。当前设计意图见
[架构文档](../design/architecture.md)，接口使用见 [API 指南](../reference/api.md)。

| 验证任务 | 文档 | 是否访问真实服务 |
| --- | --- | --- |
| 常规回归、协议契约、文档与离线示例 | 本页 | 否 |
| 发行包内容、安装与导入 | [发行包验证](packaging.md) | 构建或安装可能下载依赖，不调用模型 |
| SenseNova 正文、工具与能力策略 | [SenseNova E2E](sensenova.md) | 真实模式会调用服务；`--validate` 离线 |
| 输入历史、工具图片与推理回传 | [输入 E2E](input-e2e.md) | 真实模式会调用服务；推理回传仅支持真实响应 |
| generate、准备阶段、缓存与并发性能 | [性能复测](../guides/performance.md) | 否，真实 SDK 配本地 HTTP transport |
| 历史测量与原始样本 | [性能记录](performance/README.md) | 已完成的历史测试 |

## 环境与常规验证

在仓库根目录使用 Python 3.12+。`dev` 提供测试和构建工具，默认 `sdks` 组安装三种协议所需 SDK。
锁文件安装与项目 CI 保持一致：

```sh
uv sync --locked --group dev
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests examples
uv run --no-sync ruff format --check src tests examples
uv run --no-sync python docs/reference/schema/verify.py
uv run --no-sync python examples/local_demo.py
uv run --no-sync python examples/translation.py
```

普通 pytest 不读取真实 API 密钥、不调用模型服务。SDK 集成测试用本地 HTTP transport
驱动官方 SDK，验证其实际序列化、SSE 解析和资源关闭；它证明本地协议链路，不能替代服务端 E2E。
固定 Schema 的独立校验不导入生产转换器，来源、快照日期和校验范围见
[协议模板说明](../reference/schema/README.md)。

只检查本轮相关范围时，可以运行已有测试分组：

```sh
uv run --no-sync pytest -q tests/contract
uv run --no-sync pytest -q tests/integration/test_documentation.py
uv run --no-sync pytest -q tests/performance
```

`tests/performance` 里的 pytest 使用小负载检查基准驱动的正确性，不是性能测量结论。
专门基准单独运行，避免与其他测试争用 CPU；不要对普通回归设置依赖机器速度的绝对耗时门槛。

## 测试代码的职责

| 目录 | 职责 |
| --- | --- |
| `tests/unit` | 配置、模型、编码解码和状态等局部行为 |
| `tests/contract` | 跨协议契约、公开翻译、流生命周期与原生投影 |
| `tests/integration` | SDK HTTP 边界、公开生成、缓存、并发和扩展协议 |
| `tests/packaging` | 在临时源码副本中构建并检查 wheel、源码包和元数据 |
| `tests/performance` | 独立性能驱动及驱动自身的完整性检查 |

`conftest.py` 只注册 fixture，测试模块之间不相互导入。独立预期值不通过被测转换器生成。
共用辅助按职责使用明确的 `tests.helpers` 模块路径：

| 辅助模块 | 职责 |
| --- | --- |
| `protocols.py`、`chat_history.py` | 独立原生消息、工具、历史和流样例 |
| `transport.py` | SDK 测试替身及 SSE 序列化 |
| `requests.py`、`profiles.py` | 请求与显式 Profile 构造 |
| `streaming.py` | 离线完整流累积与渲染 |
| `schema.py` | 固定来源 Schema 读取与验证 |
| `sdk.py` | 官方 SDK 类型与未消费字段检查 |
| `tool_results.py`、`tool_images.py` | 独立工具结果及图片样例 |

## 记录测试结果

结果文档应注明日期、源码提交或指纹、未提交改动、Python 与依赖版本、命令、场景范围、
策略和通过／失败数量。真实测试记录服务与模型；性能比较另记负载、预热、样本数、
计时边界、GC、原始样本和两版指纹。未保留的基线或原始证据要明确写出，不能补造。

补跑报告保留原始失败，区分本轮结果与合并结果。正文匹配、请求结构正确、回传相等和
真实缓存命中是不同的判据，按实际证据描述。结果只适用于记录的版本与环境；
历史通过不能写成当前持续保证。

密钥通过环境变量或隐藏输入读取，不进入文档、命令行参数、源码、报告或提交。
真实思维、签名和密文不作为结果附件，按现有脚本保留比较结果与摘要。
原始运行报告默认留在 `/tmp` 等本机目录；需要入库时只保存已脱敏且可复核的必要证据。

已记录结果见 [测试结果汇总](results/README.md) 和 [性能记录](performance/README.md)。
