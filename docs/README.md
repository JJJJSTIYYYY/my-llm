# 项目文档

文档按读者和目的分为四部分。使用页说明接口怎么调用，开发页解释为什么这样设计，
测试页保存复现步骤和结果，审查页记录根因、修复与处理决定。

## 使用文档

面向 `my-llm-py` 包的使用者。从[项目快速开始](../README.md#快速开始)进入，
业务调用使用 `MyLLM.generate`，离线翻译使用 `MyCodec.encode` / `decode` / `convert`。

- [API 使用指南](reference/api.md)：调用参数、返回值、流式消费、工具回合、消息缓存和独立翻译。
- [配置参考](reference/configuration.md)：连接与模型 Profile、覆盖规则、能力声明和参数路径。
- [直接使用中间协议](guides/middleware-io.md)：类型化消息、工具结果、公开事件和保存后的流翻译。
- [语义损失处理](guides/semantic-loss.md)：支持边界、默认错误、忽略策略和双向 hook。
- [SenseNova 接入](guides/sensenova.md)：用现有 Profile 配置三种上游接口。
- [原生协议模板](reference/schema/README.md)：固定来源的请求、返回、流事件与 Schema；模板范围不等于本包的翻译支持范围。

## 开发文档

面向本项目开发者，解释当前设计、取舍、组件职责和扩展契约。

- [本地开发规范](guides/development.md)：模块划分、依赖方向与导入约定。
- [架构与职责边界](design/architecture.md)：公开业务入口、请求准备、返回转换与状态所有权。
- [中间协议设计](design/protocol-middleware.md)：消息身份、原生回放、内部流与公开事件的区别。
- [流状态设计](design/stream-state.md)：块身份、工具参数累积、成功终态与失败处理。
- [翻译字段边界](design/translation-boundaries.md)：Encoder 投影与 Decoder 语义损失的不同职责。
- [Schema 过滤计划缓存](design/schema-filter-cache.md)：可复用的类型计划与每次仍需执行的数据过滤。
- [SDK 客户端生命周期](design/client-lifecycle.md)：逐请求资源所有权和流关闭顺序。
- [接入新的协议](guides/adding-protocol.md)：四个抽象基类、默认声明、工厂注册与扩展要求。
- [构建与发行](guides/building.md)：版本来源、产物组成和发行流程。

## 测试文档

面向复现与验证。先看[测试入口](testing/README.md)，其中区分常规离线测试、真实服务 E2E、
性能基准和发行包验证。结果页记录对应日期与基线，不能替代当前版本复测。

- [输入修复与推理回传 E2E](testing/input-e2e.md)：独立原生输入、对照、工具图片和下一轮 HTTP 回传判据。
- [连续请求前缀矩阵](testing/history-prefix.md)：四输入 × 三上游、缓存开关、finish 消息回填与请求前缀一致性。
- [SenseNova E2E](testing/sensenova.md)：运行范围、离线检查、真实服务调用和补跑方法。
- [generate 性能测试](guides/performance.md)：负载、计时边界、复测和比较约束。
- [性能结果与原始样本](testing/performance/README.md)：保留日期、环境和原始数据的历史测量。
- [发行包验证](testing/packaging.md)：源码包、wheel、元数据、隔离安装与当前 CI 差异。
- [测试结果目录](testing/results/README.md)：按日期查看常规验证和供应商 E2E 汇总。

## 审查文档

面向审查者和维护者。根因、修复记录与明确不修复的决定统一从[审查目录](reviews/README.md)进入。

- [公开流翻译与业务入口修复](reviews/2026-11-22-public-stream-fix.md)：PR #2 报告的问题、独立实现及验证范围。
- [协议转换与公开接口审计](reviews/2026-11-22-codec-audit.md)：A01—A05 的证据、业务影响与不修复决定。
- [空推理字符串修复记录](reviews/2026-11-06-empty-reasoning.md)：原始反馈与空文本的处理边界。
- [文档对齐审查](reviews/2026-11-23-documentation-alignment.md)：本轮文档修正与保留的 CI 入口差异。
