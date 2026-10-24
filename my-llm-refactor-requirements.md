# my-llm 协议转换重构需求

## 1. 范围

本文档只记录需要改动的部分；已实现且不需要改的行为不展开。

### 1.1 需求

1. 流式返回：思维链、最终回答、工具调用、usage。
2. 输入：工具、思维链、多模态用户消息、多模态工具结果。
3. 用量：流式可获取接口返回的 usage；非流式不提供（后续确认）。

多模态仅指图片和文本；audio、file 不支持。

### 1.2 使用模型

- 每个 `MyLLM` 固定绑定一个输入协议（`stdio_protocol`）和一个上游协议（`provider.protocol`）。
- 一次调用只涉及这一对协议，不存在运行时动态路由。
- 请求方向：输入协议 → MyLLM 支持的字段集合 → 上游协议。
- 响应方向：上游协议 → MyLLM 支持的字段集合 → 输入协议。

---

## 2. 判定规则

### 2.1 MyLLM 支持的字段集合（中间协议字段）

固定集合，由 MyLLM 定义：

- 文本；
- 图片（用户消息、工具结果；URL 或 base64）；
- 思维链（文本，以及同协议下的 signature / encrypted_content）；
- 工具调用（tool_calls）；
- 输入的工具（tools）；
- 工具结果（文本、图片）；
- usage（input、output、total、cached、reasoning）。

**中间协议需要定义具体的类`ProtocolMiddleWare`，不能直接声明为 dict、Any**

Profile 只负责现有的 Attribute、Parameter、Config 属性、能力申明及分支控制，不能通过继承扩展支持集合。

**兼容开关导致支持字段无法表达时，仍按支持字段损失处理。**

### 2.2 两条处理路径

| 阶段 | 情况 | 处理 |
|---|---|---|
| encoder（源协议 → 中间协议） | 中间协议不支持的字段 | 静默丢弃：不报错、不记录、不进入encoder/decoder翻译结果 |
| decoder（中间协议 → 目标协议） | 中间协议支持的字段，但目标协议无法表达 | `SemanticLossError`，进入损失策略 |

注意：源协议 与 目标协议 是相对的，用户输入 -> API时，用户输入协议（stdio_protocol）是源协议，API协议（provider.protocol）是目标协议，API返回 -> 用户时，API协议（provider.protocol）是源协议，用户输入协议（stdio_protocol）是目标协议。
对于 encoder 来说其始终是源协议 → 中间协议，对于 decoder 来说其始终是中间协议 → 目标协议。

**encoder/decoder**

### 2.3 损失策略

这里的 draft 指 encoder/decoder 完成后、尚未交给 transport 或返回给用户的完整中间结果。
实时流响应对 SDK client yield 的每个 chunk 各做一次翻译路径并逐 chunk 返回，此时 draft 指该 chunk 的完整翻译结果；流末尾另对最终完整结果整体回译一次，其 draft 指完整消息。

`SemanticLossError` 后续流入三种分支：

1. `raise`（默认）：抛给上层。
2. `ignore`：完全不记录，静默继续，draft 原样进入后续链路。
3. `hook`：进入 `BaseSemanticLossAdaptor` 的对应 hook。

`BaseSemanticLossAdaptor` 类与 `hook` 分支：

- 请求方向调用 `BaseSemanticLossAdaptor.input_semantic_loss_hook`；
- 响应方向调用 `BaseSemanticLossAdaptor.output_semantic_loss_hook`；
- `semantic_loss_protocol`、`input_semantic_loss_hook`、`output_semantic_loss_hook`统一由 MyLLM 构造时传入，`BaseSemanticLossAdaptor`由 MyLLM 自动构造，同时加入二次设置接口；
- `semantic_loss_protocol='raise'`时，hook 自动生成为内部直接raise；`semantic_loss_protocol='ignore'`时，hook 自动生成空函数，等价于过滤该损失；`semantic_loss_protocol='hook'`时，hook 赋值为用户传入的 hook 函数；
- hook 拿到完整 draft、损失集合、方向/位置上下文，单条损失需封装为 `SemanticLossItem`，损失集合封装为`SemanticLossGroup`；
- hook 约定为原地修改 draft，后续进入 transport/返回用户的是传入 hook 的同一个 draft；
- my-llm 不额外限制 hook 的执行权限，可以执行任何逻辑，新增 user 消息、搬移图片；

默认策略是 `semantic_loss_protocol='raise'`；

---

## 3. 目标架构

### 3.1 Codec 生命周期

- 注册表从“实例”改为“工厂”。
- 每个 `MyLLM` 实例动态创建自己的 codec 实例。
- 不使用全局共享 codec 实例。
- `BaseSemanticLossAdaptor` 与 encoder/decoder/tramsport 一样由 `BaseCodec` 统一管理。

### 3.2 支持字段的表示

- 中间协议是 MyLLM 支持字段集合的表示，不是“所有协议字段的并集”。
- encoder 负责“源协议 → 支持字段集合”。
- decoder 负责“支持字段集合 → 目标协议”。
- encoder 不需要知道目标协议；decoder 不需要知道源协议。
- Profile 只参与字段路径、默认值、兼容开关。

### 3.3 损失处理链路

请求方向：

```text
用户输入 → encoder → decoder → BaseCodec 捕获
  → raise / ignore / hook(input_semantic_loss_hook)
  → transport
```

响应方向：

```text
transport → encoder → decoder → BaseCodec 捕获
  → raise / ignore / hook(output_semantic_loss_hook)
  → 用户
```

两个方向默认都是 `raise`；hook 分支按方向调用不同 hook。

### 3.4 同协议翻译的语意损失

- 输入协议 == 目标协议时，由于中间协议的存在也可能会导致语意损失。
- 此损失是可接受的，当前不做特殊处理。

---

## 4. 变更项

### C1：固定支持字段集合，Profile 不再支持继承扩展（P0）

**当前实现**

- 中间协议是隐式 `list[dict]`，各协议 encoder/decoder 自行挑选字段。

**问题**

- 没有一份“MyLLM 支持什么”的显式清单。

**目标**

- 明确固定支持字段集合（见 2.1）。
- Profile 只提供路径、默认值、兼容开关，不支持通过继承扩展新的支持集合。
- 支持集合之外的字段统一静默丢弃。
- 各协议 encoder/decoder 按固定集合硬编码字段识别与丢弃规则。
- encoder/decoder 以该集合为共同契约。

ProtocolMiddleWare 最小实现层级关系（JSON表示, "类名": "变量名: 变量示例值"），需要根据具体情况适当补充：

```json
{
  "ProtocolMiddleWare": {
    "Messages": { // list
      "UserMessage": {
        "str": "role: user name",
        "UserMessageContent": { // list
          "UserMessageTextContent": {
            "str": "type: text",
            "str": "text: message content"
          },
          "UserMessageImageContent": {
            "str": "type: image",
            "str": "base64: image base64"
          }
        }
      },
      "AssistantMessage": {
        "str": "role: assistant name",
        "AssistantMessageContent": { // list
          "AssistantMessageReasoningContent": {
            "str": "type: text",
            "str": "reasoning: reasoning content",
            "str": "mark: encrypted/summary/raw" // 类型为str，变量名为mark，变量值为'encrypted'、'summary'或'raw'，'encrypted'时可用于存放加密思维链的签名等
          },
          "AssistantMessageTextContent": {
            "str": "type: text",
            "str": "text: message content"
          },
          "ToolCalls": { // list
            "ToolCall": {
              // 统一的 ToolCall 对象
            }
          },
        }
      },
      "ToolMessage": {
        "str": "call_id: call_123456",
        "ToolMessageContent": { // list
          "ToolMessageTextContent": {
            "str": "type: text", // 类型为str，变量名为type，变量值为'text'
            "str": "text: message content"
          },
          "ToolMessageImageContent": {
            "str": "type: image",
            "str": "base64: image base64"
          }
        }
      },
    }
  },
  "Tools": {
    // 统一的 function tool 结构（使用 openai chat_completion 现成的结构即可，但需要 my-llm 独立定义，不依赖 SDK）
  },
  "Stream": { // list, 某轮 generate 输出的 chunks 列表
    "AssistantMessageChunk": {
      // 统一的 AI消息 chunk 结构
    },
    "ToolCalls": {
      // list, 同 AssistantMessage.ToolCalls，my-llm 统一收集 chunk 后一次性 yield，因此不需要 ToolCallsChunk
    },
  },
  "Usage": {
    // 统一的 usage 结构
  }
}
```

---

### C2：Codec 改为工厂 + 每实例独立（P0）

**当前实现**

- `src/my_llm/registry.py` 使用全局 `_CODECS` 保存 codec 实例。
- `MyCodec` 从注册表拿到的是同一批实例。
- `MyCodec.register` 要求 protocol components 无状态（空 `__slots__`）。

**问题**

- 每个 `MyLLM` 需要注入自己的 `BaseSemanticLossAdaptor`；把 adaptor 挂到共享 codec 上会跨实例污染。
- 全局实例与“每实例独立 loss 策略”冲突。

**目标**

- 注册表保存工厂，而不是实例。
- 每个 `MyLLM` 实例创建独立 codec。
- adaptor 由 `MyLLM` 注入到每实例 codec，不跨实例共享。
- 允许 codec 实例持有 adaptor；现有“protocol components 必须无状态”的约束按实例生命周期调整，但不得跨实例共享。
- 不再依赖全局可变状态。

---

### C3：统一“丢弃 / 报错”边界（P0）

**当前实现**

- encoder 的 `select` / `encode_content` 会丢弃未识别字段。
- decoder 对无法表达的字段行为不一致：
  - 有的抛 `UnsupportedCapabilityError`；
  - 有的转成普通 text；
  - 有的 JSON 化；
  - 有的直接丢弃。
- 没有 `SemanticLossError`。

**问题**

- 不支持字段和“支持字段损失”混在一起。
- 部分支持字段被静默改变语义。
- 部分不支持字段导致整次调用失败。

**目标**

- encoder 阶段只做“不支持字段丢弃”，不产生 `SemanticLossError`。
- decoder 阶段对“支持字段、目标无法表达”产生 `SemanticLossError`。
- 不支持字段一律静默丢弃，不进入异常，也不进入损失集合。
- 不再存在“支持字段静默变 text / JSON”的路径，如 image_url 中 base64 自动转 text。

---

### C4：`SemanticLossError` 的载荷与抛出时机（P0）

**当前实现**

- 只有 `UnsupportedCapabilityError`，仅带一句描述。
- 没有完整翻译结果、损失集合、方向、阶段、路径。

**问题**

- adaptor 无法基于完整 draft 做原地修改。
- 调用方无法定位具体字段。
- 无法区分请求方向和响应方向。

**目标**

- `SemanticLossError` 表示**my-llm支持，但目标协议无法表达的字段**
- `SemanticLossError` 携带：
  * 完整 draft
  * 损失集合 `SemanticLossGroup`：SemanticLossItem 列表与损失发生方向 origin_source: Literal["input", "output"]
  * 单条损失对象 `SemanticLossItem`：
    - origin_value：原始值/原始对象
    - origin_path：原始值/原始对象在原输入的位置，结构类似于 Parameter.path 的 path 字符串
- decoder 记录损失后继续翻译；完整翻译结束后统一抛出（实时流为当前 chunk 的翻译结束后抛出）。
- 请求方向在 transport 之前抛出；响应方向在返回用户之前抛出。

---

### C5：BaseCodec 语意损失三分支 + `BaseSemanticLossAdaptor`（P0）

**当前实现**

- 没有统一策略层。
- 没有 adaptor，也没有 input/output 两个 hook。
- decoder 抛错后直接向上传播。

**问题**

- 没有位置让用户在不修改 my-llm 源码的情况下处理支持字段损失。
- 请求方向和响应方向无法使用不同的 hook。

**目标**

- `BaseCodec` 捕获 `SemanticLossError`，按策略处理：
  - `raise`：向上抛出；
  - `ignore`：静默继续，不记录；
  - `hook`：调用 `BaseSemanticLossAdaptor`。
- `BaseSemanticLossAdaptor` 提供两个 hook：
  - `input_semantic_loss_hook`：请求方向；
  - `output_semantic_loss_hook`：响应方向。
- hook 由最外层 `MyLLM` 注入，知道源协议和目标协议。
- hook 拿到完整 draft、损失集合、上下文；原地修改 draft。

---

### C6：跨协议思维链（P0）

**当前实现**

- 各协议 `foreign_block` 用 `text_fields={"thinking": "thinking"}` 把外来 thinking 变成普通 text。
- Chat 目标会把多个 text block 拼成一个字符串，思维链和最终回答混在一起。
- 跨协议场景下只有 Ollama 目标保留 `thinking`。

**问题**

- 思维链是支持字段，却被静默改变语义。
- 客户端无法区分思维链和最终回答。

**目标**

- 同协议：保留 thinking 文本、signature、encrypted_content。
- Chat 目标：Profile 支持 `reasoning_content` 时映射为 `reasoning_content`；否则 `SemanticLossError`。
- Responses 目标：能表达为 reasoning item 时映射；否则 `SemanticLossError`。
- Anthropic / Gemini 目标：没有签名时不能伪造签名；无法表达时 `SemanticLossError`。
- Ollama 目标：使用 `thinking`。
- 不允许把思维链和最终回答拼成一个字符串。
- 不允许静默把思维链当普通回答返回。
- hook 可以自行决定重排、丢弃或抛错。

---

### C7：多模态工具结果（P0）

**当前实现**

- Responses / Anthropic 目标可以原生保留工具结果中的图片。
- Chat 目标把工具结果图片整体 JSON 序列化成字符串。
- Gemini 目标把非对象内容包成 `{"result": ...}`。
- Ollama 目标把内容转成字符串。

**问题**

- 图片是支持字段，但在 Chat / Gemini / Ollama 目标被静默改写成文本。
- 工具结果图片无法被模型看到，也没有触发 `SemanticLossError`。

**目标**

- Responses / Anthropic：工具结果图片正常翻译。
- Chat / Gemini / Ollama：工具结果图片属于支持字段损失，抛 `SemanticLossError`。
- hook 分支下，用户可以自行把图片搬到后续 user 消息、丢弃或抛错；my-llm 不自动构造/添加内容。
- 工具结果文本在所有协议正常翻译；文本可以是字符串，或对象/数组序列化为文本。

---

### C8：工具定义 schema 保真（P0）

**当前实现**

- Ollama 在 decoder 阶段不符合 SDK schema 约束时会抛错（validate_schema）。

**问题**

- decoder 阶段没有采用抛 `SemanticLossError` 返回损失集合的方式。

**目标**

- 统一在有语意损失时才抛 `SemanticLossError`。
- `strict` 在 Chat/Responses/Anthropic 是支持的；转到 Gemini/Ollama 时继续保持静默丢弃。
- 工具 name、description、parameters 正常翻译。

---

### C9：不支持字段统一静默丢弃（P1）

**当前实现**

- audio：Chat 支持，Responses 输入校验拒绝，Anthropic / Ollama 不支持，Gemini 按 document 处理。
- file：Chat / Responses / Anthropic / Gemini 支持，Ollama 不支持。
- metadata 多数情况下丢弃。
- 不支持的字段有的报错、有的转换。

**问题**

- 违反“不支持字段静默丢弃”的原则。
- 同一份输入在不同协议下行为不同。

**目标**

- audio、file、metadata、等其他未列入支持集合的字段，统一静默丢弃。
- 不报错、不记录、不进入目标协议。
- 即使某个 provider 原生支持 audio/file，my-llm 也不翻译。
- 请求方向和响应方向都适用。

---

### C10：响应方向纳入统一损失链路（P0）

**当前实现**

- 请求方向有 `MyCodec.prepare` 和 transport。
- 响应方向通过 `MyCodec.result` / `_render` 直接回译给用户，没有损失捕获、策略或 adaptor。

**问题**

- 响应方向的支持字段损失仍然静默。
- 请求/响应两个方向行为不一致。

**目标**

- 响应方向同样经过：transport → encoder/decoder → BaseCodec 捕获 → 策略 → 用户。
- 默认 `raise`。
- hook 分支调用 `output_semantic_loss_hook`。

---

## 5. 验收

### 5.1 支持字段与丢弃

- 文本、图片、思维链、工具调用、工具定义、工具结果（文本+图片）、usage 属于支持集合；目标能表达时正常翻译，不能表达时按 5.2-5.6 处理。
- audio、file、metadata 静默丢弃，不产生异常；`strict` 在 Chat/Responses/Anthropic 保留，转到 Gemini/Ollama 时静默丢弃，不产生异常。
- 支持字段损失产生 `SemanticLossError`，不支持字段不产生。
- 不再存在支持字段静默变 text / JSON 的路径。

### 5.2 损失策略与 hook

- 默认 `raise`：支持字段损失直接抛给上层。
- `ignore`：静默继续，不记录。
- `hook`：请求方向调用 `input_semantic_loss_hook`，响应方向调用 `output_semantic_loss_hook`。
- hook 拿到完整 draft、损失集合、上下文；可原地修改 draft、新增 user 消息、搬移图片。实时流按 chunk 批次调用 hook，流末尾另以完整结果调用一次。
- hook 不修改 draft 时，draft 原样进入 transport 或返回用户。
- hook 内可以自行 raise / ignore / translate。

### 5.3 思维链

- 同协议：signature / encrypted_content 保留。
- Chat 目标：映射为 `reasoning_content`，或抛 `SemanticLossError`；不能混入正文。
- Responses 目标：映射为 reasoning item，或抛 `SemanticLossError`。
- Anthropic / Gemini 目标：有思维链，无签名时抛 `SemanticLossError`。
- Ollama 目标：`thinking` 正常。
- 请求方向和响应方向都验证。

### 5.4 多模态工具结果

- Responses / Anthropic：工具结果图片正常翻译。
- Chat / Gemini / Ollama：工具结果图片抛 `SemanticLossError`。
- 配置 hook 后，可以把图片搬到后续 user 消息；不配置时默认执行空函数体。
- 工具结果文本在所有协议正常翻译。

### 5.5 工具定义

- name、description、parameters 正常翻译。
- Ollama 在 decoder 阶段按统一路径在语意损失时抛 `SemanticLossError`。
- `strict` 在 Chat/Responses/Anthropic 保留；转到 Gemini/Ollama 时静默丢弃，不报错。

### 5.6 流式结构化数据与 usage

- 工具调用在流结束时一次性出现，不增量。
- 流结束后 `MyLLM.get_last_usage()` 返回完整上轮调用记录的 usage。
- 非流式 `generate` 返回 dict/list，不包含 usage。
- 流式过滤 usage 事件，不 yield，只记录。
- usage 缺失字段为空，不报错。

### 5.8 Codec 生命周期

- 注册表保存工厂。
- 每个 `MyLLM` 实例拥有独立 codec。
- adaptor 按实例注入，不跨实例共享。
- 不存在全局可变 codec 状态。

---

## 6. 非目标或禁止项

- audio / file 支持；
- 跨协议思维链签名伪造；
- 工具调用参数增量输出；
- 任意协议之间的动态路由；
- my-llm 自动把工具结果图片搬到 user 消息或自动转为工具 text 结果；
- **禁止在公共函数中添加某个协议的特判分支**

---

## 7. 迁移

### 7.1 行为变化

- 支持字段损失（跨协议思维链、工具结果图片、Gemini schema 约束等）默认抛 `SemanticLossError`。
- 不支持字段（audio、file、metadata）统一静默丢弃，不再有的报错、有的转换。
- `strict` 按目标能力处理：Chat/Responses/Anthropic 保留，Gemini/Ollama 静默丢弃。
- 支持字段不再静默变 text / JSON。
- 请求方向和响应方向使用同一套损失策略。
- 注册表从实例改为工厂，codec 每实例独立。

### 7.2 兼容策略

- 新增 `SemanticLossError` 公开异常；
- `UnsupportedCapabilityError` 的支持字段损失场景迁移；
- 不支持字段的报错路径改为静默丢弃；
- 新增 `input_semantic_loss_hook` / `output_semantic_loss_hook`；
- 不需要兼容旧版实现。