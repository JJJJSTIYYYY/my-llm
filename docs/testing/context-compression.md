# 结构化上下文压缩 E2E

本页验证“外部历史消息 + 一条新用户压缩请求 → generate_json → fake tool 参数字典”的实际场景。
接口契约见 [API 指南](../reference/api.md)，通用接入见 [SenseNova 指南](../guides/sensenova.md)。

## 复现

安装开发环境和 SDK 后，准备项目外的 UTF-8 JSON 文件，格式为 Chat 消息数组，
或带 `messages` 数组的对象。消息必须有 `role` 和文本 `content`；不得把真实上下文放入脚本或仓库。
脚本保留这些历史消息，追加一条场景请求，通过公开 `generate_json` 发送；输出指令仍由接口注入系统消息。

```sh
uv run python examples/context_compression_e2e.py \
  --context /tmp/conversation.json \
  --report /tmp/conversation-compressed.json \
  --scenario compression compact handoff indexed_handoff
```

- `compression`：完整保留后续工作必需信息，每类最多六条。
- `compact`：极简摘要，每类最多两条。
- `handoff`：开发交接，优先最新状态和未完成工作，每类最多四条。
- `indexed_handoff`：在新用户请求内自动附加消息编号/角色/前100字符的索引，并约束待办归纳。
  历史消息保持不变；索引来自外部文件，不在脚本内写死。

默认上游为 `openai_chat`；`--upstream openai_responses anthropic` 可选择其他上游。
默认模型由现有 SenseNova 示例提供，默认输出预算为 4096 token；JSON 重试一次，SDK 自动重试关闭。
模型支持工具调用被明确开启。实际 HTTP 检查工具名称、Schema 引用已展开、系统指令和凭据不进入请求体。

密钥只从 `SENSENOVA_API_KEY` 或隐藏输入读取，不接受密钥命令行参数。
输入读取时屏蔽本次密钥和 `sk-` 形式的字符串；这不替代调用方对其他个人信息的预先脱敏。
输入和报告路径都拒绝仓库内位置，报告权限设为 `0600`。
报告包含压缩后的私有上下文，只能保存在项目外；终端仅输出指标，不输出正文、请求头、推理或签名。

## 接口验收标准

本测试只验证接口的结构化输出，内容与原会话大体相关即可，不评价模型压缩质量。

1. 原始历史来自外部文件，保持消息顺序和内容，再追加一条场景用户请求。
2. `supports_function_calling=True`，实际 HTTP 中只发送 `structured_output` 工具，
   工具 Schema 已展开引用，接口输出指令是系统消息。
3. `generate_json` 返回字典；通过严格 Pydantic 校验：summary 为字符串，
   user_goals、decisions、implemented、pending、constraints、caveats 为列表，
   每项是含 statement 字符串和 source_turns 整数列表的对象，拒绝额外字段。
4. 可读取 Usage、记录实际 HTTP 请求次数，SDK 客户端关闭。
5. 人工只确认返回内容与输入主题大体一致，不做精确复述断言。

来源编号是否准确、遗漏哪些细节、压缩比例、任务优先级及列表条数等模型质量问题，
不作为接口通过条件。脚本不包含这些评分或质量判定。

## 2026-11-28 实测

来源为本轮可见用户消息及最终答复的人工转录：28 条、4060 个 content 字符。
格式规范化，链接改为相对路径；密钥脱敏，省略内部指令、工具日志、进度消息及引用元数据。
它不是平台逐字完整导出，也不是长上下文压力测试。输入和完整模型输出均未入库。

模型为 `sensenova-6.8-flash-lite`，本次只对 Chat 上游实测四个场景，
不把脚本支持的其他上游算作已验证。基础三个场景与索引对照使用相同历史，输入摘要指纹一致。
输入以原始 Chat 消息列表进入公开接口，没有将上下文先通过 Codec 改造成其他输入协议。
本轮还包含此前待提交的系统消息/提示词常量改动，没有提交或推送本次新增脚本。

| 场景 | 结构/工具路径 | 结果字符 | 字符保留率 | HTTP 请求 | input / output / total |
| --- | --- | ---: | ---: | ---: | --- |
| compression | 通过 | 1047 | 25.79% | 1 | 3132 / 575 / 3707 |
| compact | 通过 | 1054 | 25.96% | 2 | 6268 / 966 / 7234 |
| handoff | 通过 | 1310 | 32.27% | 1 | 3135 / 3028 / 6163 |
| indexed_handoff | 通过 | 1409 | 34.70% | 1 | 4214 / 3932 / 8146 |

四个结果均为模型实际工具参数，返回字典及其嵌套字段均通过测试脚本的严格 Pydantic 校验，
内容大体围绕输入中的开发与测试主题，所有客户端关闭，接口场景结论为 **4/4 通过**。
compact 发生一次 JSON 重试，原始报告没有保存首次失败的具体类型，不能据此宣称是某一种 JSON 错误。
本轮没有单独对照每条原生 SSE Usage；表内值来自 `get_last_usage()`，不是独立计费审计。

表中字符比例仅为早期原始报告的观察值，不作为验收依据，也不是 token 或费用节省率；
当前脚本已移除质量评分、引用核对和预算检查，仅保留输入/输出长度等接口观测数据。
历史私有报告可能仍有这些早期观察字段；不据此将结构正确的结果判为接口失败。
本轮只实测 Chat 上游，不将脚本支持的其他上游算作已验证。

新增离线用例覆盖外部文件读取/脱敏、拒绝仓库内路径、私有报告权限与密钥拒写、
四种场景的真实 SDK 序列化、历史保持不变、用户请求与系统指令顺序以及 Usage，共 7 项通过。
本轮全量回归 4210 项通过；验收范围收口后的脚本变动另以这 7 项专项复核。

本机输入与报告：

- `/tmp/myllm-context-compression-input-20261128.json`
- `/tmp/myllm-context-compression-chat-20261128.json`
- `/tmp/myllm-context-compression-indexed-20261128.json`

它们是私有材料，未作为仓库附件；不要复制其中正文到公共报告。
