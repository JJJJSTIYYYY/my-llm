# Profile 重构逻辑复核

本轮只复核逻辑、调用流程和职责边界，没有重新进行性能测量。
对照本轮开始前的源码快照和上轮重构前工作区快照，保留用户原有未提交修改。

## 发现并修复

1. `reasoning_enable=omit` 被当作模型不支持 reasoning。
   这会拒绝原先有效的配置：只关闭启用开关的自动路由，仍让 reasoning_effort 独立生效。
   现在只有 Parameter.supports=False 明确拒绝 reasoning；omit 继续只禁用对应自动路由。
   增加五协议测试，分别覆盖省略 enable 路由、同时省略两条 reasoning 路由。

2. Profile 各层只检查 Parameter/Config 类型，未核对参数 source/scope 是否一致。
   扩展 Model 可以把 max_tokens 从 input 改为 output，悄悄绕过 token 校验与请求注入。
   现在统一验证字段身份和层间 source/scope 一致性；覆盖自定义 __post_init__ 忘记绑定身份的情况。
   client 参数的 input 来源校验归并到声明校验，移除连接组装中的重复检查，以及单独维护的 scopes 字典。

3. 输出参数禁用只覆盖 delta，Anthropic content_block_start 和 Responses part.done / response.completed 仍可能回填可见内容。
   修正在各协议自己的编码器中：禁用的文本／可见思考保持为空，同时保留块引用、签名、加密内容、工具块和完成事件。
   Responses 的 part 事件与最终快照共用一个本协议的块处理函数，避免两套禁用规则。
   回归测试先复现这四种失效情况，再验证修复；同时检查原始输入未被修改。

此外，移除测试和 SenseNova 示例中 config 辅助函数已不再使用的 name 参数，迁移对应调用点。
将读取已删除 supports_reasoning 字段、永远得到默认 False 的断言改为检查实际 Parameter.supports。

## 职责与差异范围

本轮生产代码修正仅涉及四个文件：

- profile/resolution.py：通用配置身份、继承与声明校验。
- codec/shared/options.py：通用 reasoning 能力检查与自动路由语义。
- codec/anthropic/encoding/encoder.py：Anthropic 块起始事件。
- codec/openai/responses/encoding/encoder.py：Responses part 事件与最终快照。

逐文件对照确认：MyLLM、Transport、StreamState、MessageStream、Schema 过滤与缓存实现均未修改。
结合上一轮重构，生产代码差异仍只位于 profile 与 codec 目录。
Profile 没有引入协议名分支、SDK 客户端依赖或响应处理；公共 Codec 层没有增加协议名分派。
各协议的路径、事件结构、SDK 参数和特殊校验仍由自己的 Codec/Encoder/Decoder/Transport 实现负责。

保留既有的 None 继承、omit 路由禁用、普通默认值补缺、Config 字典合并、输入协议推断、
客户端缓存与关闭、结果所有权、流终止检查和工具历史校验。
自动生成参数的最高覆盖优先级继续生效。

## 验证

完整离线测试：2068 项通过，包含协议职责边界、动态协议注册、五协议 SDK 链路、
输出初始块与最终快照、身份继承、并发隔离与资源关闭等回归。
Ruff lint/format 与 git diff --check 通过。
在本轮重构范围内，未再发现未修复的死代码、职责越界或范围外行为变化。

上一轮性能报告中的源码 SHA256 对应复核前的采样快照；本轮修复后没有重新测量，不将旧数字视为新版本的实测结果。
