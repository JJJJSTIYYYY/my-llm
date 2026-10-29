# 本地开发规范

codec 布局与接口以本页目录说明、[架构文档](../design/architecture.md)和
[接入协议指南](adding-protocol.md)为准。

## 目录与模块

运行时包位于 src/my_llm；应用调用方式见 [API 指南](../reference/api.md)，
完整职责树见[架构文档](../design/architecture.md)。

- codec/base 的 codec.py、encoder.py、decoder.py、transport.py 定义抽象契约，模块名与五种原生实现一致。
- 五种协议各自维护 encoding 和 decoding；转换方法直接落在 encoder.py / decoder.py。
  内容、字段及工具翻译辅助放在本方向的 content.py，双向共用的辅助直接放在本协议根目录，以 content.py、schema.py 等职责命名；生成参数映射、请求结构、验证与过滤放在 Decoder；官方 SDK 执行及资源关闭放在协议根目录的 transport.py。
  BaseCodec 声明 ProtocolDefaults 并组合 encoder / decoder / transport，转换与执行组件互不导入。
  仅在存在对应职责时保留辅助模块，不为每个转换方法或短小函数建立分组包。
- middleware 只设 models.py、stream.py、protocol.py 三个实现模块，分别组织消息内容与工具、流事件与用量、完整输入容器与校验；相关模型和 wire 投影放在一起。
- basic 集中保存共用基础能力：data.py 负责数据捕获与 SDK 字段只读视图，exceptions.py 定义异常，
  semantic_loss.py 管理损失收集与策略，protocols.py 合并协议注册、工厂快照和可选 SDK 发现。
- codec/shared 保存不选择厂商的公共机制；conversion.py 负责消息、媒体、工具转换与历史关联验证，
  cache/compiled_cache.py 集中保存类型分析、编译计划、过滤与验证；响应关闭辅助随执行契约放在 codec/base/transport.py；
  options.py 负责配置路径、选项准备与校验。Decoder 每次解析一次 PreparedOptions，并输出 client / request / local 字典；连接参数在构造时准备。
- profile/configuration.py 保存配置类型、覆盖字段与字典合并；resolution.py 负责解析、快照及校验。
  协议默认值保存在各 Codec.defaults；profile/defaults.py 只从 basic/protocols.py 复制已注册声明。
- 包根目录的 my_llm.py 负责生成编排与 SDK 客户端复用，runtime/message_stream.py 和 state.py 保存响应流生命周期与请求状态。
  实例通过异步上下文或 aclose 关闭客户端；响应流按请求清理，见[生命周期设计](../design/client-lifecycle.md)。
- __init__.py 只保留说明、导入与 __all__，不放业务代码。

新增模块应以职责命名；同协议双向共用辅助放在协议根目录。只有职责确实形成多个模块时才增加目录，不为短小辅助函数单独建包。
新增协议依据[接入协议指南](adding-protocol.md)实现四个基类的契约并调用 `MyCodec.register(factory)` 注册零参数工厂。
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
| helpers/profiles.py | 按各协议默认声明构造显式 Profile 覆盖。 |
| helpers/tool_results.py | 独立编写的原生工具结果样例与期望文本提取。 |
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
