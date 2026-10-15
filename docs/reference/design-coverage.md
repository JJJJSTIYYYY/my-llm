# 设计要求与验收对应

实现以用户设计文档及其附带消息参考为依据。以下对应的是可执行测试，
追加需求中的四种原生协议均通过官方 SDK 实现。

| 设计要求 | 实现 | 本地验收 |
| --- | --- | --- |
| OpenAI Responses/Chat Completions 及 Anthropic/Gemini/Ollama 官方异步流 | `invoke.py`、两套 Codec | SDK 模拟传输测试中的三种 stream 模式与 Responses 工具测试 |
| 公开 generate；非流式由同一流拼接；仅实现 _stream | `generate`、`MyLLM.generate`、`MyLLM._stream` | 三种 stream 参数取值测试、流式累加测试 |
| 统一公开命名、完整 docstring、使用指南与独立配置指南 | 根导出的 `generate`、`MyLLM`；`api.md`、`configuration.md` | `test_public_api.py` 名称/参数/字段说明检查；`test_documentation.py` 示例语法、配置字段覆盖及两套 API 的实际 SDK 工具回合 |
| Codec 与 MyLLM 分层 | `codec/` 转换与解析；`invoke.py` 执行请求 | Codec 单元/契约测试和真实 SDK 模拟传输测试 |
| Profile 创建、无预设厂商模型；四种协议 | `ProviderProfile`、`ModelProfile` | `test_profiles.py` 默认值、原生协议支持与未知协议拒绝测试 |
| Model > Provider > 协议；Omit 禁用；字典按键覆盖 | `resolve_profiles`、`merge_dicts` | `test_model_overrides_provider_and_dict_keys`、`test_omit_disables_defaults_and_none_inherits` |
| 模型上下文窗口元数据与 Provider 模型列表管理 | `ModelProfile.max_context_window`、`ProviderProfile.models` 及管理方法 | `test_model_catalogue.py` 默认值、任意整数、增删列举、列表隔离和请求无行为变化测试 |
| 客户端 None 回退，调用参数优先 | `client_parameters` | `test_client_parameter_override_and_omission`、`test_owned_client_parameters_and_cleanup` |
| Token 名称按 API 映射，默认不限制 | `request_parameters` | `test_tokens_translate_and_upstream_always_streams` |
| reasoning、effort 能力、None 映射禁用与自定义开启/强度字段 | `request_parameters` | `test_encoding.py` 推理测试、`test_edge_cases.py` 自定义强度/summary/布尔/Omit 测试 |
| 自动生成推理参数优先；Responses 默认 detailed summary | 推理 binding 与 extra_body 合并 | 自动生成参数优先级、summary 测试 |
| 不支持上游 Stream 抛 UnsupportedProviderError | `resolve_profiles` | `test_upstream_stream_is_required` |
| 内容路径与推理路径；SDK 对象直接取属性，不实现完整 JSONPath | `path_value`、两个 decoder | `test_path_reads_sdk_attributes_without_dumping`、两个自定义路径测试 |
| 原生完成/failed/incomplete/error、无完成的截断流 | 两个 decoder 的 feed/finish | `test_decoding.py` 完成与失败测试、截断流与部分输出测试 |
| 完整 Function Calling，不向上输出参数碎片 | 两个 decoder 的工具聚合 | 并行交错 Chat 工具测试、Responses 快照不重复测试、不完整工具拒绝测试 |
| Responses/Chat 两种工具模板自动识别、双向转换、同格式透传；统一返回字典 | `codec/tools.py`、两套 encode/decoder | 两种输入格式与目标 API 组合测试、混合列表、SDK 请求与工具返回测试 |
| Responses/Chat messages 自动识别、双向转换、混合历史及原生项保留 | `codec/input.py`、`codec/openai/messages.py`、`codec/openai/content.py` | `test_message_conversion.py`、`test_message_inputs.py` 格式组合、并行调用、内容块、原生回放和实际 SDK 序列化测试 |
| Chat 推理历史按开关回传；Response 原生 reasoning/function items 或 ID 回放 | 两套 encode、Message 的原生元信息 | 推理回传开关、原生历史、previous_response_id 与 SDK 回合测试 |
| 工具结果正确关联调用 ID | `validate_messages` | `test_tool_result_association`、原生与统一调用一致性测试 |
| 固定四种角色；名称开关 | `ModelProfile.supported_roles`、Chat encode | 固定角色、角色校验和名称开关测试 |
| Usage none/native/stream_options；空 choices 用量事件 | 请求生成与两个 decoder | 默认不暴露用量、原生用量、空 choices SDK 测试 |
| Message/MessageChunk 与拼接参考 | `types/messages.py` | `test_messages.py` 累加、完成检查与深拷贝测试 |
| 消息在 generate 入口复制一次，内部只读复用；输出字典隔离 | `invoke.py`、`codec/openai/messages.py`、`codec/validation.py` | `test_ownership.py` 复制次数、延迟消费、原生历史、配置与结果快照测试 |
| 包内采用绝对导入 | `my_llm...` 导入、Ruff TID252 | `ruff check src tests examples` |
| 异步取消、失败、提前消费的生命周期 | `MessageStream`、`MyLLM._stream` | `test_invoke.py` 取消/提前关闭/失败、所有者与借用客户端测试 |
| PyPI 可构建与安装 | Hatchling、py.typed | wheel/sdist 内容、Twine strict、隔离安装与导入检查 |

目录中的集成测试使用四种真实官方 SDK 配合 httpx.MockTransport，均为本地测试。
没有以模拟测试结果宣称真实厂商模型调用已验证。
