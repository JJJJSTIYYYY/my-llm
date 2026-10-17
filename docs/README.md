# 项目文档

## 使用与配置

- [API 使用指南](reference/api.md)：调用、流式消费、工具回合和独立格式转换。
- [配置参考](reference/configuration.md)：ProviderProfile、ModelProfile 与覆盖规则。
- [五协议请求与返回模板](reference/schema/README.md)：固定来源的 Schema、请求、返回及流事件。

## 开发与设计

- [本地开发规范](guides/development.md)：目录职责、导入、测试辅助代码与结构验证。
- [接入新的协议](guides/adding-protocol.md)：Encoder / Decoder 必需接口、固定流程与注册验证。
- [构建与发行包验证](guides/building.md)：安装、构建和 CI。
- [架构与职责边界](design/architecture.md)：代码目录、请求和返回路径、既有重构验收。
- [流状态设计](design/stream-state.md)：块身份、累积操作和终态约束。
- [SDK 客户端生命周期](design/client-lifecycle.md)：实例复用、请求资源隔离和显式关闭。

## 审计与验证记录

- [最初重构需求与当前实现对照](audits/refactor-requirements-2026-10-17.md)：逐项核对、验收范围与后续设计调整。
- [官方协议审计](audits/protocol-audit-2026-10-17.md)及[修复验收](audits/protocol-fixes-2026-10-17.md)。
- [协议定义与测试预期复核](audits/protocol-conformance.md)。
- [字段保留专项复核](audits/lossless.md)。
- [并发与资源生命周期验证](audits/concurrency.md)。

reference 保存接口、配置与协议模板；guides 保存操作步骤；design 保存实现约束；
audits 保存带有范围和验证依据的审计记录。审计中的测试数量对应当时快照，
当前状态以实际测试输出为准。
