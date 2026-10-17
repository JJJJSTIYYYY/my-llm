# 本地开发规范

## 目录与模块

运行时包位于 src/my_llm；应用调用方式见 [API 指南](../reference/api.md)，
完整职责树见[架构文档](../design/architecture.md)。

- codec/base 的 codec.py、encoder.py、decoder.py 定义抽象契约，模块名与五种原生实现一致。
- 五种协议各自维护 encoding 和 decoding；两者都按 messages、tools、stream 分组，
  decoding 另有 request 分组。SDK、字段名与协议特判留在对应目录。
- codec/shared 保存不选择厂商的公共机制；values.py 负责 SDK 值读取、路径读取及参数值展开。
- profile/configuration.py 保存配置类型、覆盖字段与字典合并；resolution.py 负责解析、快照及校验。
- stream 保存消息流生命周期与每次请求的状态；invoke.py 负责生成调用编排。
- __init__.py 只保留说明、导入与 __all__，不放业务代码。

新增模块应以职责命名；避免重新引入 common.py 或 utils 作为未分类代码的容器。
本次整理仅移动模块、更新导入和文档引用，没有调整参数、转换、终态或清理逻辑。

## 导入约定

运行时代码使用 my_llm 开头的绝对导入，依赖具体定义模块，
例如 `from my_llm.profile.resolution import ResolvedProfile`。
包初始化文件负责组合公开导出；应用、示例和公共接口测试继续使用
`from my_llm import MyLLM, ModelProfile, ProviderProfile`。

导入子模块时可以通过所属包引用模块本身，例如 `from ...encoding.stream import chunks`；
不通过包的重导出反向获取实现类。为避免定义阶段的循环依赖而已有的函数内延迟导入保留。
Ruff 统一导入顺序并禁止相对导入；结构契约检查实现代码不依赖包的符号重导出。

## 测试目录

unit、contract、integration、packaging 分别保存单元、协议契约、集成和发行包测试。
conftest.py 只注册 pytest fixture；通用测试辅助代码使用 tests.helpers 的明确模块路径。

| 模块 | 职责 |
| --- | --- |
| helpers/protocols.py | 独立编写的原生消息、工具和流样例。 |
| helpers/transport.py | SDK 测试替身和 SSE 序列化。 |
| helpers/requests.py | 请求构造辅助。 |
| helpers/streaming.py | 离线完整流的累积与渲染辅助。 |
| helpers/schema.py | 固定官方 Schema 的读取和校验。 |
| helpers/sdk.py | 原生 SDK 类型及未消费字段校验。 |

测试模块之间不相互导入，避免依赖收集顺序。只供本文件使用的小型辅助函数可留在测试文件中。
协议预期样例仍独立于生产转换代码，移动文件不改变验证依据。

## 验证

```sh
uv sync --locked --group dev
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests examples
uv run --no-sync ruff format --check src tests examples
uv run --no-sync python docs/reference/schema/verify.py
uv run --no-sync python examples/local_demo.py
```

目录移动后同步检查 Markdown 本地链接和发行包内容；完整测试会执行这些检查。
结构整理还应对比移动前后的代码 AST，核对除导入外的执行节点不变，
并核对顶层公开导出；不要以测试通过代替对纯结构变更的直接检查。
发行步骤见[构建指南](building.md)。

## 2026-10-17 首轮目录整理验收

对照整理前的工作区快照，148 个运行时模块除导入外的 AST 及导入绑定一致，
22 个既有包接口的导出保持一致；24 个迁移的测试辅助定义未改写，
82 个协议参考文件逐字节相同。依赖声明及锁文件保持原值，
构建配置仅将源码包的文档范围改为整个 docs 目录。

完整测试为 **915 passed**，包括新增的直接实现导入和测试收集独立性约束；
Ruff 检查与 182 个 Python 文件的格式检查、全部 Markdown 本地链接、
协议模板校验和离线示例均通过。wheel / sdist 构建与 Twine 严格检查通过，
并在隔离进程从 wheel 成功导入全部 148 个运行时模块。

## 2026-10-17 协议目录分层验收

五种协议的 encoding / decoding 已统一按职责分组，见[原生协议目录](../design/architecture.md#原生协议目录)。
82 个模块移入 messages、tools、stream，以及 decoding/request。
encoding 顶层仅保留 __init__.py、encoder.py、matching.py，
decoding 顶层仅保留 __init__.py、decoder.py；没有保留旧路径的转发模块。

与分层前快照相比，原有 148 个运行时模块除导入外的 AST 和导入绑定一致；
35 个新增分组初始化文件仅含说明。公开入口、构建配置、依赖和锁文件不变，
82 个协议参考文件逐字节相同。

完整测试仍为 **915 passed**；发行包测试新增完整运行时文件清单比对，
覆盖新增分组及其初始化文件。Ruff 与 217 个 Python 文件的格式检查、
Markdown 本地链接、协议模板校验、离线示例、wheel / sdist 构建、Twine 严格检查通过；
隔离进程从 wheel 成功导入全部 183 个运行时模块。
