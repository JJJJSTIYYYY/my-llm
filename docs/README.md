# 项目文档

## 使用与配置

- [API 使用指南](reference/api.md)：调用、流式消费、工具回合和独立格式转换。
- [配置参考](reference/configuration.md)：ProviderProfile、ModelProfile 与覆盖规则。
- [五协议请求与返回模板](reference/schema/README.md)：固定来源的 Schema、请求、返回及流事件。

## 开发与设计

- [本地开发规范](guides/development.md)：目录职责、导入、测试辅助代码与结构验证。
- [generate 性能测试](guides/performance.md)：复杂协议负载、无网络 SDK 路径及完整返回耗时。
- [接入新的协议](guides/adding-protocol.md)：Encoder / Decoder 必需接口、固定流程与注册验证。
- [构建与发行包验证](guides/building.md)：安装、构建和 CI。
- [架构与职责边界](design/architecture.md)：当前代码目录、组件职责、请求和返回路径。
- [流状态设计](design/stream-state.md)：块身份、累积操作和终态约束。
- [Schema 过滤计划缓存](design/schema-filter-cache.md)：缓存的实际对象、类型计划图、两次请求的执行过程与可运行示例。
- [翻译字段边界](design/translation-boundaries.md)：已识别字段、开放字典与 Profile 兼容配置的范围。
- [SDK 客户端生命周期](design/client-lifecycle.md)：实例复用、请求资源隔离和显式关闭。

## 审计与验证记录

- [新协议缓存通用性与策略审计](audits/cache-audit-2026-10-20.md)：独立协议验证、统一缓存键、有界 LRU、批量计划复用与前后基准。
- [Parameter / Config 重构性能复测](audits/parameter-config-performance-2026-10-20.md)：同源码基线重测、完整矩阵、原生交替确认、实例构造和 GC 诊断。
- [当前性能回退定位与修复](audits/performance-regression-2026-10-20.md)：删除前提交、本轮修复前后对照、实际冗余原因与原始样本。
- [Profile 重构后的性能复测](audits/performance-retest-2026-10-20.md)：此前清理的完整矩阵与原始样本；两份源码均已删除未知字段保留。
- [当前执行路径与抽象契约审计](audits/request-path-audit-2026-10-20.md)：共同调用链、必要差异、冗余清理和新增协议约束。

- [历史性能复测](audits/performance-retest-2026-10-19.md)：当时的五协议矩阵、同进程对照及 GC 耗时观察。
- [历史仓库清理与架构复核](audits/architecture-audit-2026-10-18.md)：当时的约束、清理范围和验证依据。
- [最初重构需求对照](audits/refactor-requirements-2026-10-17.md)：历史逐项核对与后续设计调整。
- [官方协议审计](audits/protocol-audit-2026-10-17.md)及[修复验收](audits/protocol-fixes-2026-10-17.md)。
- [协议定义与测试预期复核](audits/protocol-conformance.md)。
- [并发与资源生命周期验证](audits/concurrency.md)。

reference 保存接口、配置与协议模板；guides 保存操作步骤；design 保存实现约束；
audits 保存带有范围和验证依据的审计记录。审计和开发规范中的历史验收对应当时的
目录、接口和测试快照；当前结构与契约以 design 和 guides 的当前说明为准，
验证结果以实际测试输出为准。
