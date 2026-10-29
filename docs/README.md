# 项目文档

## 使用与配置

- [直接使用中间协议](guides/middleware-io.md)：显式 middleware stdio、具体模型历史、工具回合和公开流事件。
- [中间协议与语义损失迁移](guides/semantic-loss.md)：固定支持集合、默认损失错误、双向 hook、私有 usage 与工厂注册。
- [API 使用指南](reference/api.md)：调用、流式消费、工具回合和独立格式转换。
- [配置参考](reference/configuration.md)：ProviderProfile、ModelProfile 与覆盖规则。
- [SenseNova 接入与端到端测试](guides/sensenova.md)：三种上游、四种输入/输出格式的真实服务验证与失败说明。
- [五协议请求与返回模板](reference/schema/README.md)：固定来源的 Schema、请求、返回及流事件。

## 开发与设计

- [本地开发规范](guides/development.md)：目录职责、导入、测试辅助代码与结构验证。
- [generate 性能测试](guides/performance.md)：增量消息缓存、复杂协议负载、无网络 SDK 路径与复测方法。
- [接入新的协议](guides/adding-protocol.md)：Encoder / Decoder 必需接口、固定流程与注册验证。
- [构建与发行包验证](guides/building.md)：安装、构建和 CI。
- [完整中间协议](design/protocol-middleware.md)：具体模型、源位置、损失 draft、实例生命周期与缓存边界。
- [架构与职责边界](design/architecture.md)：当前代码目录、组件职责、请求和返回路径。
- [流状态设计](design/stream-state.md)：块身份、累积操作和终态约束。
- [Schema 过滤计划缓存](design/schema-filter-cache.md)：缓存的实际对象、类型计划图、两次请求的执行过程与可运行示例。
- [翻译字段边界](design/translation-boundaries.md)：已识别字段、开放字典与 Profile 兼容配置的范围。
- [SDK 客户端生命周期](design/client-lifecycle.md)：实例复用、请求资源隔离和显式关闭。

reference 保存接口、配置与协议模板；guides 保存操作及验证步骤；design 保存实现约束。
测试与性能结果保存在本机输出目录，复测方法见[开发指南](guides/development.md#验证)和
[性能指南](guides/performance.md)。
