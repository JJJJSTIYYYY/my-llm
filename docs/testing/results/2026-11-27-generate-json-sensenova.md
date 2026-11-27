# 2026-11-27 SenseNova 结构化输出实测

本次使用真实服务，模型为 `sensenova-6.8-flash-lite`。源码为 `HEAD=916dfdc` 之上的
未提交工作区（源码版本 `2.1.0`），不能把该 HEAD 当作完整实现基线。
环境：Python 3.12.12、OpenAI SDK 2.54.0、Anthropic SDK 0.125.0、
Pydantic 2.14.1、HTTPX 0.28.1。

密钥通过隐藏输入进入进程内存，临时设置 `SENSENOVA_API_KEY`，退出时移除；
没有写入项目、脚本、报告或命令行参数。脚本和脱敏报告只保留在本机 `/tmp`，
没有保存认证头、原始推理或签名。以下先保留修复前证据，同日修复与复测另列于后。
接入参数复用 [SenseNova 示例](../../../examples/sensenova_e2e.py)。

## 主矩阵：含引用的嵌套 Schema

三种上游分别使用 Chat、Responses、Anthropic、Middleware 四种输入格式，
每种各测工具与提示词模式，共 24 个场景。返回值与独立预期对象精确比较，
并使用 Pydantic 验证字段类型。数据包含中文、整数、布尔值、数组、嵌套对象、花括号与引号。
Schema 来自 Pydantic 类型或同一类型导出的字典，嵌套字段使用 `$ref` / `$defs`。

提示词模式明确覆盖 `supports_function_calling=False`，并检查解析后的能力值为 False。
实际 HTTP 请求不得带工具，必须包含 JSON Schema 输出指令。
工具模式设置为 True，实际 HTTP 请求必须恰有一个 `structured_output` 工具。
所有调用都使用公开 `generate_json`，上游保持现有 SSE 执行路径，不执行工具。

| 上游 | 提示词兜底 | fake tool | 工具失败原因 |
| --- | ---: | ---: | --- |
| Chat Completions | 4/4 | 0/4 | `details` 应为对象，上游返回字符串 |
| Responses | 4/4 | 0/4 | 同上 |
| Anthropic Messages | 4/4 | 0/4 | 同上 |

不能把工具成功被调用当作 Schema 验证通过，也不能将这个矩阵描述成全通过。
提示词通过项均核对了原生 Usage 与 `get_last_usage()`，并检查响应和客户端关闭。
主矩阵的工具失败发生在内容断言，失败行未完整记录后续的 Usage 比较结果，
因此不据此声称这 12 个失败项的用量均已逐项核对。

## 内联 Schema 对照

保留同一对象、模型、Profile 和断言，只将 `details` 的引用替换为原定义，移除 `$defs`。
在 Middleware 输入下，三种上游 × 两种模式共 6 个场景全部通过。
实际工具请求中的 `details.type=object`、`details.properties.note.type=string` 已检查。
这 6 项还通过了独立原生 Usage 核对以及 HTTP 响应和 SDK 客户端关闭检查。

该对照只覆盖 Middleware 输入，没有重跑全部四种输入格式，不能扩大为内联 Schema 的完整 24 项矩阵。

## 引用与原生参数诊断

额外执行 Chat 上游、Middleware 输入、工具模式的一项引用 Schema 诊断：

- 实际 HTTP 工具参数完整保留 `details.$ref=#/$defs/Details` 及对应对象定义。
- 原生工具参数与接口交付字典相等，`details` 在原生参数里已经是字符串。
- 内容断言仍失败，未将它改记为成功。
- 实际执行一次请求，Usage 为 input=517、output=95、total=612、cached=0、reasoning=0，
  与原生计数一致，响应和客户端均关闭。

在该诊断中，没有发现 Codec 丢失引用定义或将对象改写为字符串；
当前服务/模型没有按该引用 Schema 返回正确的嵌套字段。内联对照支持问题与引用形式有关，
但不构成对服务内部实现或全部 JSON Schema 支持范围的判断。

`generate_json` 当前只要求得到 JSON 字典，不执行完整 Schema 验证。
因此这类合法 JSON 内的字段类型错误不会触发 JSON 重试；本次未修改该业务契约，
也没有加入协议特例或对有效 JSON 值进行重写。

## 字符串复制探针

最初的探索包含末尾反斜杠字符串，并产生内容断言失败；另做一次 Chat 提示词诊断予以保留。
目标字符串长度 34、含一个反斜杠，模型返回长度 35、含两个反斜杠。
原生 JSON 直接解析值与接口返回值一致，说明该诊断中的偏差已存在于模型输出。
主矩阵去掉此反斜杠，继续单独验证其他结构特征；不能因此将字符串探针算作通过。

## 低输出预算重试

Chat 上游、Middleware 输入、提示词模式开启消息缓存，将输出预算限制为 1 token，
JSON 重试预算为 1，SDK 自动重试仍为 0。两次真实输出均不是完整 JSON，最终按预期抛出 `ValueError`。

| 尝试 | input | output | total | cached | reasoning |
| --- | ---: | ---: | ---: | ---: | ---: |
| 首次 | 278 | 1 | 279 | 0 | 0 |
| 重试 | 278 | 1 | 279 | 0 | 0 |
| `get_last_usage()` 累计 | 556 | 2 | 558 | 0 | 0 |

累计值逐项等于原生 SSE 中两次 Usage 的和，重试耗尽后仍可读取；两次响应和客户端均关闭。
实际 HTTP 中均未发送工具。该用例验证失败重试及累计，不代表已验证真实服务中的重试后恢复成功。

## 同日修复：生成前展开本地引用

初轮只描述了服务对引用 Schema 的返回偏差，根因归属不完整：
在 fake tool 的 Schema 准备阶段，项目遗漏了本地引用展开，直接发送了 Pydantic 生成的引用。
这次在 `utils/json_schema.py` 统一处理，先展开本地 JSON Pointer，再构造 `FunctionTool`；
提示词模式也复用相同的归一化结果，没有为 SenseNova 或某种协议添加特例。

支持 `$defs`、旧 `definitions`、引用链、转义路径和引用旁的约束；
定义容器在输出中移除，默认值与示例保持普通数据。循环、外部、缺失引用及命名锚点
在准备阶段明确报错，不猜测递归结构、不联网下载 Schema。

修复后用原有含引用的 Pydantic 类型/Schema 字典重新执行三上游 × 四输入的 12 个工具场景。
测试脚本没有手动内联输入，实际 HTTP 工具参数全部检查为已展开的对象结构，无 Schema `$ref` / `$defs`。

| 上游 | 精确内容断言 | 原先的嵌套对象类型错误 |
| --- | ---: | --- |
| Chat Completions | 3/4 | 未再出现 |
| Responses | 4/4 | 未再出现 |
| Anthropic Messages | 4/4 | 未再出现 |

唯一内容失败为 Chat 上游、Anthropic 输入：`details` 已正确返回对象，
但 `details.note` 字符串长度应为 22，模型返回长度 66。
原生工具参数与接口字典一致；保留为模型内容断言失败，不改写为 12/12 全通过。
其他 11 项精确内容、原生 Usage 对照与资源关闭检查均通过。

本次复测未重跑三上游 × 四输入的提示词真实矩阵；之前 12/12 的结果保留为修复前证据。
离线新增引用展开及两模式请求验证：专项 179 项、全量 4197 项通过，Ruff、格式和 diff 检查通过。
结果仍不代表接口执行了完整的返回 Schema 校验。

## 本机材料与范围

- `/tmp/myllm-sensenova-generate-json-20261127.json`：主矩阵 24 项。
- `/tmp/myllm-sensenova-json-inline-control-20261127.json`：内联对照 6 项。
- `/tmp/myllm-sensenova-json-ref-wire-control-20261127.json`：引用与原生参数诊断。
- `/tmp/myllm-sensenova-json-escaping-probe-20261127.json`：字符串复制诊断。
- `/tmp/myllm-sensenova-json-retry-control-20261127.json`：真实 JSON 失败重试与逐次 Usage。
- `/tmp/myllm-sensenova-json-refs-fixed-20261127.json`：引用展开修复后的 12 个工具场景。
- 临时脚本通过隐藏输入获取密钥；不从这些材料重建密钥或认证头。

主矩阵之前还有探索失败和一次主动取消的在途请求，未完整保留所有请求统计。
不能把上述最终场景行数简单相加当作整个会话的真实 HTTP 请求总数。
本次结果只代表该日期、模型和配置；修复后全量离线回归结果另列于上，
未独立验证供应商的 HTTP 非流式响应。接口契约见 [API 使用指南](../../reference/api.md)。
