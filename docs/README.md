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
- [generate 性能测试](guides/performance.md)：复杂协议负载、无网络 SDK 路径及完整返回耗时。
- [接入新的协议](guides/adding-protocol.md)：Encoder / Decoder 必需接口、固定流程与注册验证。
- [构建与发行包验证](guides/building.md)：安装、构建和 CI。
- [完整中间协议](design/protocol-middleware.md)：具体模型、源位置、损失 draft、实例生命周期与缓存边界。
- [架构与职责边界](design/architecture.md)：当前代码目录、组件职责、请求和返回路径。
- [流状态设计](design/stream-state.md)：块身份、累积操作和终态约束。
- [Schema 过滤计划缓存](design/schema-filter-cache.md)：缓存的实际对象、类型计划图、两次请求的执行过程与可运行示例。
- [翻译字段边界](design/translation-boundaries.md)：已识别字段、开放字典与 Profile 兼容配置的范围。
- [SDK 客户端生命周期](design/client-lifecycle.md)：实例复用、请求资源隔离和显式关闭。

## 审计与验证记录

- [Middleware 模型整理核对](audits/middleware-model-2026-10-25.md)：逐项区分实际缺陷与设计取舍，统一项身份和思维结构，记录验证与性能对照。
- [中间协议公开入口验收与性能](audits/middleware-io-2026-10-24.md)：显式 stdio、跳过两端转换、具体结果与事件、十组成对 SDK 基准及尾延迟限制。
- [Schema 缓存检查与优化试验](audits/schema-cache-2026-10-24.md)：命中与所有权复核、完整调用链对照及未采用的规则路由和批量校验实验。
- [需求全量验收复核](audits/requirements-acceptance-2026-10-24.md)：按最新确认口径核对全部需求与验收，补齐反例及最终源码证据。
- [重构代码清理与职责复核](audits/code-cleanup-2026-10-24.md)：重复复制、死路径、确定输入边界、原生差异归属及清理后复测。
- [中间协议重构验收与性能对照](audits/protocol-middleware-2026-10-24.md)：C1–C10、缓存隔离、60 场景、构造与原生 ABBA；明确记录性能回退。

reference 保存接口、配置与协议模板；guides 保存操作步骤；design 保存实现约束；
audits 保存本轮模型核对与最近五轮性能报告，原始 JSON 日志已清理。报告保留测量条件、结果汇总和复测方法。
