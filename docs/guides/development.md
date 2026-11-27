# 本地开发规范

本页面向维护 `my-llm` 的开发者，说明模块归属、依赖方向与变更约束。
应用接入见[API 使用指南](../reference/api.md)，设计取舍见[架构文档](../design/architecture.md)，
测试复现与结果入口见[测试文档](../testing/README.md)。

## 目录与模块

运行时包位于 `src/my_llm`，完整职责树见[架构文档](../design/architecture.md#目录与组件)。

- `my_llm.py` 编排生成；`codec/converter.py` 提供独立翻译并路由内部请求准备。
- `codec/base` 定义组件契约及共同流程。三种协议各自维护 `encoding`、`decoding`、`codec.py` 和 `transport.py`。
  字段映射、事件处理与协议限制由各自实现负责；双向共用辅助放在该协议根目录。
- `codec/shared` 保存可复用的转换、能力、输入输出策略、流操作及生成参数绑定。
  不把不同协议仅在外观上相似的行为强行抽象为同一规则。
- `middleware/models.py`、`stream.py`、`protocol.py` 定义具体模型、公开事件和容器；
  `validation.py`、`history.py`、`usage.py` 分别负责内部结构检查、工具历史归一化和计数累积。
- `basic` 保存异常、语义损失策略和协议注册；`profile` 保存配置声明、合并与解析。
- `utils` 只保存全项目通用的数据复制与读取、路径操作、JSON Schema 转换和 JSON 提取。
  不依赖 Profile、中间层消息模型、Codec、缓存实现或生成编排；领域策略留在所属模块。
- `cache/compiled_cache.py` 保存进程级类型规则；`message_cache.py` 保存实例会话的已翻译历史。
- `runtime/state.py` 保存每次响应的块状态，`message_stream.py` 管理公开异步流和关闭回调。
  SDK 客户端按请求创建并关闭，详见[生命周期设计](../design/client-lifecycle.md)。
- `__init__.py` 只保留包说明、导入与 `__all__`，不放业务逻辑。

新增模块按职责命名。只有职责确实形成多个模块时才增加目录，不为单个短小辅助另建包。
新增协议实现四个基类，并通过 `MyCodec.register(factory)` 注册零参数工厂，
不修改公共入口或中央协议枚举，详见[接入协议](adding-protocol.md)。

## 业务边界与设计约束

业务入口为 `MyLLM.generate` / `generate_json` 和 `MyCodec.encode` / `decode` / `convert`；
配置类型、模型及流管理方法服务于这些入口。组件方法、准备步骤及内部批次只按完整调用链的前置条件协作，
不承诺它们可作为独立业务接口使用。

Encoder 丢弃中间层未支持的原生字段；Decoder 对中间层支持、目标无法表达的语义记录损失。
新增支持应同时明确模型语义、两个方向的转换与真实原生输出，不能靠未知字段透传或下游猜测补齐。

自动协议识别可以比较候选协议的特征；内部连续 Responses assistant 项分组用于保持回合语义与推理回传判断。
这两项是当前接受的设计取舍，见[架构中的取舍说明](../design/architecture.md#设计取舍)。
不为隐藏这些内部规则在公共消息模型新增协议合并开关或兼容字段。

公共完整流转换接受有效的完整事件序列。应用主动改坏终态快照、删除生命周期事件后的行为不在保障范围；
不要为此增加重复校验、状态同步或猜测恢复。现有审查结论见[审查文档](../reviews/README.md)。

## 导入约定

运行时代码使用以 `my_llm` 开头的绝对导入，依赖具体定义模块，
例如 `from my_llm.profile.resolution import ResolvedProfile`。
应用、示例和公共接口测试使用 `from my_llm import MyLLM, ModelProfile, ProviderProfile`。

允许通过所属包引用子模块，例如 `from my_llm.codec.anthropic import transport`；
不通过包的重导出反向获取实现类。为避免定义阶段循环依赖而已有的函数内延迟导入保留。
Ruff 统一导入顺序并禁止相对导入。

## 变更验证

测试按 `unit`、`contract`、`integration`、`packaging` 分层；原生期望样例独立于生产转换器。
测试模块不互相导入，通用辅助放在明确的 `tests.helpers` 模块。
具体目录、命令与复现边界见[测试指南](../testing/README.md)。

纯结构调整应直接核对移动前后的执行节点和顶层公开导出，不以测试通过代替变更范围检查。
文档变更检查本地链接、示例与当前接口，历史结果保留原基线，不包装成当前验证。
性能比较见[性能测试指南](performance.md)，发行流程见[构建指南](building.md)。
