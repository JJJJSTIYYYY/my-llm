# 本地开发规范

当前 codec 布局与接口以本页目录说明、[架构文档](../design/architecture.md)和
[接入协议指南](adding-protocol.md)为准。页末各轮验收保留对应时点的目录、接口与测试快照，
其中的旧路径和职责划分不作为当前实现约束。

## 目录与模块

运行时包位于 src/my_llm；应用调用方式见 [API 指南](../reference/api.md)，
完整职责树见[架构文档](../design/architecture.md)。

- codec/base 的 codec.py、encoder.py、decoder.py、transport.py 定义抽象契约，模块名与五种原生实现一致。
- 五种协议各自维护 encoding 和 decoding；转换方法直接落在 encoder.py / decoder.py。
  内容、字段及工具翻译辅助放在本方向的 content.py，双向共用的辅助放在本协议的 utils/；生成参数映射和请求结构放在 Decoder；请求验证、过滤及官方 SDK 执行放在协议根目录的 transport.py。
  BaseCodec 声明 ProtocolDefaults 并组合 encoder / decoder / transport，转换与执行组件互不导入。
  仅在存在对应职责时保留辅助模块，不为每个转换方法或短小函数建立分组包。
- codec/shared 保存不选择厂商的公共机制；conversion.py 负责 SDK 值读取及媒体 URI 转换，
  options.py 负责配置路径、选项准备与校验。Transport 返回请求独立的 PreparedOptions，
  Decoder 必须显式接收并复用，不重复解析或添加内部兜底路径。
- profile/configuration.py 保存配置类型、覆盖字段与字典合并；resolution.py 负责解析、快照及校验。
  协议默认值保存在各 Codec.defaults；profile/defaults.py 只从 registry.py 复制已注册声明。
- stream 保存消息流生命周期与每次请求的状态；invoke.py 负责生成调用编排与 SDK 客户端复用。
  实例通过异步上下文或 aclose 关闭客户端；响应流按请求清理，见[生命周期设计](../design/client-lifecycle.md)。
- __init__.py 只保留说明、导入与 __all__，不放业务代码。

新增模块应以职责命名；同协议双向共用辅助放在其 utils/ 中并按内容职责命名。
新增协议依据[接入协议指南](adding-protocol.md)实现四个基类的契约并调用 MyCodec.register(codec)。
Protocol 使用 str，新增协议不修改公共入口、枚举、默认值表或公共编排。
Decoder._messages / _tools 与公开 messages / tools 均返回普通 list，不添加恢复分组。

## 导入约定

运行时代码使用 my_llm 开头的绝对导入，依赖具体定义模块，
例如 `from my_llm.profile.resolution import ResolvedProfile`。
包初始化文件负责组合公开导出；应用、示例和公共接口测试继续使用
`from my_llm import MyLLM, ModelProfile, ProviderProfile`。

导入子模块时可以通过所属包引用模块本身，例如 `from my_llm.codec.gemini import transport`；
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
| helpers/ | 提供独立协议样例、SDK 本地响应及测试共用辅助；不通过隐藏恢复记录判定翻译结果。 |

测试模块之间不相互导入，避免依赖收集顺序。只供本文件使用的小型辅助函数可留在测试文件中。
协议预期样例仍独立于生产转换代码，移动文件不改变验证依据。
注册及完整请求行为由[独立第六协议测试](../../tests/integration/test_protocol_registration.py)验证；
该实现不复用五个内置协议的 Encoder / Decoder / Transport。

## 验证

性能基准通过 `python -m tests.performance` 单独运行；普通 pytest 只执行小规模完整性检查。
复杂负载、计时边界与参数见[性能测试指南](performance.md)。

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

## 历史验收：2026-10-17 首轮目录整理

对照整理前的工作区快照，148 个运行时模块除导入外的 AST 及导入绑定一致，
22 个既有包接口的导出保持一致；24 个迁移的测试辅助定义未改写，
82 个协议参考文件逐字节相同。依赖声明及锁文件保持原值，
构建配置仅将源码包的文档范围改为整个 docs 目录。

完整测试为 **915 passed**，包括新增的直接实现导入和测试收集独立性约束；
Ruff 检查与 182 个 Python 文件的格式检查、全部 Markdown 本地链接、
协议模板校验和离线示例均通过。wheel / sdist 构建与 Twine 严格检查通过，
并在隔离进程从 wheel 成功导入全部 148 个运行时模块。

## 历史验收：2026-10-17 协议目录分层

当时五种协议的 encoding / decoding 按职责分组；该布局后由 2026-10-18 的收敛调整替代。
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

## 历史验收：2026-10-18 codec 收敛与抽象契约

本轮记录早于请求校验统一移入 Transport 的调整；当前 Decoder 不再承担下述能力检查。
当前固定流程与组件边界见[架构文档](../design/architecture.md)和[接入协议指南](adding-protocol.md)。

codec 的 Python 文件由 171 个收敛为 63 个，其中五协议的 encoding / decoding
从 142 个文件收敛为 34 个。转换方法直接实现在类中，内容辅助与 SDK 资源管理按职责保留，
不再为单个方法建立 messages / tools / stream / request 分组包。

BaseCodec 的两个方向都成为必需的抽象属性；BaseEncoder 固定整批匹配，
BaseDecoder 固定能力检查、消息解码、工具解码与请求组装。
新协议仅实现对应原生接口并注册，无需改写公共编排或依赖另一协议的转换器。
自定义实现的接口调整和旧内部路径迁移见[接入协议指南](adding-protocol.md)。

与整理前快照逐项核对，115 个转换方法及辅助函数的执行体在更新调用引用后 AST 一致，
五个 SDK transport 文件逐字节一致；原有顶层公开导出与全部协议参考文件未变。
流状态、字段语义、原生参数和资源清理算法保持原有行为。

完整测试 **933 passed**，覆盖五协议任意组合、真实 SDK 的本地传输与并发隔离，
并新增缺失抽象接口、Decoder 直接接收中间态、只转换一次、None / 空工具列表区别和方向依赖检查。
Ruff、格式、Markdown 链接、24 个 Schema / 59 个模板 / 90 帧事件校验及离线示例通过。
