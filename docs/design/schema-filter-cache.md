# Schema 过滤计划：缓存中保存的对象与运行过程

本文对应当前 [编译缓存与 Schema 过滤实现](../../src/my_llm/cache/compiled_cache.py)。
这里的“预编译”是把 SDK 的 Python 类型注解转换成可直接调用的 Python 函数图，
在首次使用该类型时生成，随后重复执行。

**缓存保存类型对象、字段规则、分支规则和处理函数；每次请求的值只作为函数参数传入。**
它不是“第一次请求的结构样本”，也不是 Encoder 到 Decoder 的转换结果表。

## 1. 先区分三种不同的 Schema / 数据

| 对象 | 例子 | 本文缓存如何处理 |
| --- | --- | --- |
| SDK 类型定义 | `ChatCompletionMessageParam`、`ResponseInputItemParam`、`MessageParam`、`Content`、`Tool` | 分析其 Python 类型结构，缓存处理计划。 |
| 用户工具的 JSON Schema | 工具 `parameters` 或 `input_schema` 中的 `properties`、`$defs`、`additionalProperties` | 按 SDK 字段声明作为应用字典处理，保留其内容；不为每份工具 JSON Schema 编译本计划。 |
| 本次请求值 | 用户文本、图片数据、工具参数、工具结果、消息里的扩展值 | 每次重新处理；不作为计划缓存的键或值。 |

例如 `Literal["user"]` 中的 `"user"` 是类型定义的固定判别常量，会随类型规则保留。
用户本次发送的 `"北京天气怎么样？"` 则是实际请求值，不会写入计划。

本文只讨论 Schema 过滤缓存。项目的其他状态各有用途：

| 位置 | 保存内容 | 生命周期 |
| --- | --- | --- |
| Schema 过滤缓存 | 类型、字段与处理函数 | 当前进程内复用。 |
| MessageCache | 语义损失处理后的已翻译输入历史及指令字段 | 启用时按 MyLLM 的 uuid4 键隔离，aclose 时释放。 |
| Encoder 的 `_ModelView` 字段缓存 | 本次转换已读取的 SDK 字段值、缺失标记及嵌套包装值 | 当前转换，不跨请求复用。 |
| MyLLM 客户端缓存 | 官方 SDK 客户端 | 当前 MyLLM 实例。 |
| StreamState / BlockState 缓存 | 当前响应的物化块和终态消息，包含真实响应内容 | 当前请求。 |

因此，“Schema 缓存不保存实际请求值”不能理解为所有请求状态都不保存实际内容。

## 2. 五类编译缓存

[compiled_cache.py](../../src/my_llm/cache/compiled_cache.py) 集中保存类型分析、计划编译、
过滤与验证逻辑，`CompiledCache` 封装以下结果的有界 LRU、统计和清理操作。
每类最多 512 项，由同一进程内的不同 MyLLM / Codec 实例共享；Codec 通过接口复用这些规则。

| 缓存函数 | 键 | 返回并保留的值 |
| --- | --- | --- |
| `fields(schema)` | Python 类型定义 | 字段名到字段注解的字典。Pydantic 类型读取 `model_fields`；TypedDict 等类型读取 `get_type_hints`。 |
| `_cached_adapter(key)`，由 `adapter(schema)` 调用 | `SchemaKey` 对象 | 对应的 Pydantic `TypeAdapter`，用于实际验证和序列化检查。 |
| `parameter_types(method)` | SDK 方法对象 | 方法关键字参数到类型注解的字典。 |
| `_compiled_plan(key, delegated_paths)` | `SchemaKey` 对象与不可变路径元组 | 根 `_FilterPlan` 对象，由它引用整个子计划图。 |
| `_declared_path(key, keys)` | `SchemaKey` 对象与不可变字段路径 | SDK 类型是否声明该路径的布尔结果。 |

仅缓存字段名和 TypeAdapter，仍会让每个请求反复解析嵌套注解、列表元素和 Union；
完整计划把这些分析结果一起保存下来。缓存只接收类型与不可变规则键，消息值不进入 CompiledCache。

## 3. 缓存键具体保存什么

入口关系是：

```python
def _plan(schema, delegated_paths=()):
    return _compiled_plan(SchemaKey(schema), delegated_paths)
```

`SchemaKey` 位于 cache/compiled_cache.py，实际只有三个槽位：

| 属性 | 保存的值 | 用途 |
| --- | --- | --- |
| `schema` | 原始 Python 类型对象或类型注解对象的引用 | 确定要编译的 SDK 类型。不是类型名字符串。 |
| `_hash` | 类型的哈希值；不可哈希注解使用对象身份值 | 支持缓存字典查找。 |
| `_hashable` | 布尔值 | 区分按类型比较和按对象身份比较的情况。 |

完整缓存键还包含 `delegated_paths`，例如 `(("tools",),)`。它是元组组成的不可变元组，
由 Decoder 的 `message_path` / `tool_path` 声明生成，表示哪些叶节点交给后续过滤钩子处理。
同一类型在普通过滤和委派过滤中分别编译、缓存；同一类型位于不同嵌套位置时也按剩余路径区分，
因此委派规则不会污染普通 `supported` / `projected` 调用。

键不包含协议请求内容、消息字段集合、model 名称、ProviderProfile 或任何响应值。
同一类型可以处理许多字段形态不同的请求。

### 为什么不直接以 schema 的普通相等比较作为缓存键

过滤器处理 Union 时，同分候选按声明顺序选择。Python 的 Union 相等比较可能忽略候选顺序：

```python
Left | Right
Right | Left
```

这两个注解需要在本计划缓存中区分，否则同分值可能复用错误顺序的计划。
`SchemaKey.__eq__` 使用 `_same_schema` 递归比较泛型参数的实际顺序，
因此 `list[Left | Right]` 和 `list[Right | Left]` 也会分开。

等价的泛型注解仍可复用计划。`Annotated` 若携带不可哈希的 metadata，则按注解对象身份缓存。
该 metadata 属于类型声明，不是传入 `supported` / `projected` 的请求值。
验证器和路径声明缓存同样使用 `SchemaKey`，合法的不可哈希 metadata 不再只在过滤入口可用、
却在实际输入验证时失败。验证器也保留 Union 的声明顺序，避免不同顺序共用错误的验证器。

## 4. 缓存值具体保存什么

真实的 `_FilterPlan` 没有 `fields`、`messages`、`output` 等属性，只有四个槽位：

```python
class _FilterPlan:
    __slots__ = ("apply", "project", "score", "literals")
```

前三个槽位保存可调用对象，`literals` 保存结构节点已声明的 Literal 判别字段及其常量。
字段表、子计划和候选列表由函数闭包引用。
闭包是函数随身保留的外部变量；本实现保留的是编译期间确定的类型规则。

| 槽位 | 调用形式 | 做什么 |
| --- | --- | --- |
| `apply` | `plan.apply(value)` | 过滤协议字段，为可变应用 payload 建立独立副本，内置不可变标量直接使用。 |
| `project` | `plan.project(value)` | 过滤协议字段，借用应用 payload，供只读验证。 |
| `score` | `plan.score(value)` | 根据当前值计算某个 Union 候选的匹配分数。 |
| `literals` | `plan.literals` | 结构类型的 Literal 字段规则；其他节点为空字典，供编译 Union 候选索引使用。 |

### 不同节点的闭包中有什么

| 类型节点 | 编译后保留的规则或引用 |
| --- | --- |
| TypedDict / Pydantic 结构 | `children`：字段名到子 `_FilterPlan` 的字典；评分函数共用 `children` 并保留 `literal_fields`。 |
| 有元素类型的列表 | `item_plan`：元素类型的子计划。 |
| Union | `choices`：按声明顺序排列的候选计划元组；`routes`：Literal 字段名、未限制该字段的候选位图及常量到候选位图的索引。 |
| Literal | 声明中的常量参数，用于评分。 |
| 普通类型 | 判断所需的类型对象。 |
| 开放字典、Any 等应用 payload | `apply` 使用 `_copy_payload`；`project` 使用 `_borrow_payload`。 |

`value` 始终是上述函数的调用参数。函数执行结束后，不会把这个参数追加到计划中。
`children`、`choices` 等也不会因某次请求增加了字段而改变。

### 开放字典与协议结构的区别

处理方式取决于传入的类型声明，而不是运行时的值是否为 dict：

| 类型声明 | 当前值中未声明的字段 |
| --- | --- |
| `dict[str, Any]`、Any 等已识别开放应用数据 | 保留内部全部键值。 |
| TypedDict / Pydantic 协议结构 | 只保留声明字段；字段内部仍按各自类型处理。 |

SDK 模型的 extra 策略不提供未知协议字段的透传能力。项目只允许在已识别开放字典内部保留任意键，
不把模型的 `extra="allow"` 等同于协议翻译规则。非标准差异通过 Profile 明确支持的兼容能力处理。

缓存规则也不会读取首条请求中的额外字段名。新的已声明可选字段可在后续请求出现，开放应用字典
内部的新键随当前值处理；未知协议结构字段仍会过滤。完整边界见[翻译字段边界](translation-boundaries.md)。

协议结构的输入若为 SDK 的 Pydantic 对象，按类型计划中的已声明字段逐个读取属性，不调用全对象 model_dump；
不能通过模型实例绕过字典输入的字段规则。SDK 的标量序列化对象仍按其原生标量接口处理。

## 5. 用一个缩小类型把缓存展开

下面的类型用于说明算法和观察内存对象，实际生产过滤仍读取五家官方 SDK 类型。

```python
from typing import Any, Literal, NotRequired
from typing_extensions import TypedDict


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image"]
    url: str


class Message(TypedDict):
    role: Literal["user"]
    content: str | list[TextPart | ImagePart]
    metadata: NotRequired[dict[str, Any]]
```

首次请求 `_plan(Message)` 后，缓存中有一项“Message 类型 → 根计划”。
根计划通过函数闭包引用下面这些节点：

```mermaid
flowchart TD
    K["缓存键: SchemaKey(Message), delegated_paths=()"] --> M["Message 根计划"]
    M -->|"children: role"| R["Literal user 计划"]
    M -->|"children: content"| C["Union 计划: str 或 list"]
    M -->|"children: metadata"| D["开放 dict 计划"]
    C -->|"choices 第1项"| S["str 计划"]
    C -->|"choices 第2项"| L["list 计划"]
    L -->|item_plan| U["Union 计划: TextPart 或 ImagePart"]
    U -->|choices| T["TextPart 结构计划"]
    U -->|choices| I["ImagePart 结构计划"]
```

下列是对真实对象的逻辑展开，便于阅读；它不是缓存实际保存的 JSON：

```text
cache entry:
  key:
    schema: Message 类型对象
    _hash: hash(Message)
    _hashable: True
  delegated_paths: ()

  value: Message 的 _FilterPlan
    apply: structured_copy 函数
      closure.children:
        role     -> Literal["user"] 的计划
        content  -> str | list[TextPart | ImagePart] 的计划
        metadata -> dict[str, Any] 的计划

    project: structured_projection 函数
      closure.children: 同一份字段子计划字典

    score: structured_score 函数
      closure.children: role / content / metadata 的子计划映射
      closure.literal_fields: [("role", ("user",))]

    literals: {"role": ("user",)}
```

可以看到，缓存中有 `"role"`、`"content"`、`"metadata"`、`"user"` 这些来自类型声明的规则，
但没有任何一条请求的完整字典，也没有某次过滤后的结果字典。

一条根缓存项会引用许多子节点。`cache_info().currsize` 统计的是根缓存项，
不是整张图的节点数。编译期间使用的局部 `plans` 字典按类型身份和剩余委派路径复用节点，
没有把每个子节点分别注册为全局根缓存项。

## 6. 两次不同结构的数据如何经过同一计划

第一次调用：

```python
first = {
    "role": "user",
    "content": "北京天气怎么样？",
    "metadata": {"request_id": "R1"},
    "vendor_extra": "本类型未声明的字段",
}
supported(first, Message)
```

运行顺序：

1. `SchemaKey(Message)` 查找失败，开始编译上述完整计划，包括这次尚未使用的列表和图片分支。
2. 根计划遍历类型声明中的字段名，定点读取 `role`、`content`、`metadata`，不读取或枚举 `vendor_extra`。
3. `content` 的 Union 根据当前值选择 `str` 分支，返回本次文本。
4. `metadata` 是开放应用字典，`apply` 为它建立独立副本。
5. 返回本次结果；计划缓存仍只有类型规则。

第二次调用换成：

```python
second = {
    "role": "user",
    "content": [
        {"type": "text", "text": "上海天气怎么样？"},
        {"type": "image", "url": "https://example.invalid/map.png"},
    ],
    "metadata": {"request_id": "R2", "new_key": {"items": [1, 2]}},
}
supported(second, Message)
```

运行顺序：

1. 命中同一个 Message 根计划。
2. `content` 根据当前值选择列表分支；第一条请求选择过字符串分支，不影响本次选择。
3. 对列表中的每个元素，先用已编译的 Literal 索引排除不匹配候选，再评分：
   `type="text"` 选 TextPart，`type="image"` 选 ImagePart。
4. 开放 `metadata` 的 `request_id`、`new_key`、`items` 都保留；计划不记录这些键和值。
5. 返回第二条请求自己的结果。

缓存复用的是候选规则，实际分支选择和数据遍历每次仍执行。

### Union 运行时具体怎么选

编译时先根据各候选声明的 Literal 字段建立索引。运行时只读取这些已声明的字段，
排除明确不匹配的候选；没有限制该字段的候选继续保留。多个候选允许同一常量时也全部保留，
不会把判别值固定映射成单个分支。字段缺失时不按该字段剪枝；不能用于索引的值仍走评分路径。

剩余结构类型候选仍执行原有评分：

1. 检查当前值里已出现的 Literal 判别字段；与该候选声明不一致时，得分为 -1。
2. 其余候选按“当前字典与声明字段的交集大小 + 1”评分。
3. 取最高分；同分保留声明在前的候选。全部候选不适配时抛出 UnsupportedCapabilityError。

剪枝保留候选的原声明顺序，最高分和同分取首项的语义不变；获胜候选也不再额外评分一次。
计划没有保存“上一次选中了哪个候选”。None、单候选等分支按当前实现直接处理，
最终合法性仍由 TypeAdapter 验证。

## 7. apply 与 project 为什么共用计划却有不同复制行为

对于第二条数据的 `metadata`：

```text
plan.apply(second)
  -> 新的协议结构
  -> metadata 是独立副本

plan.project(second)
  -> 新的协议结构
  -> metadata 引用 second["metadata"]，只读借用
```

结构字段的筛选规则相同，区别在于应用 payload 是否需要独立修改。
`project` 的调用方必须只读使用借用值；当前调用方是输入验证路径。
测试覆盖了五家官方 SDK 的 `validate_python` 和 `dump_json` 不修改这些投影输入。

执行过滤后仍需要校验：字段缺失、字段类型、字面量及惰性 Iterable 的元素合法性不能只靠
过滤器的分支评分证明。TypeAdapter 仍逐次运行，`dump_json` 用来触发惰性元素验证。
缓存命中不会把这条请求直接判定为合法。

### Decoder 内部的叶节点委派

根请求过滤需要检查 SDK 参数、复制容器和普通配置，但消息、工具仍须经过可覆盖的
`filter_messages` / `filter_tools`。`delegated_paths` 让根计划在指定消息或工具叶节点处借用原值，
把投影和独立副本的建立交给后续钩子一次完成。委派只用于 Decoder 的请求组装过程；
普通 `supported(value, schema)` 没有委派路径，继续返回隔离结果。

Anthropic 的开放 `input_schema` 同样委派后一次捕获，避免先按 SDK TypedDict 复制再覆盖完整 JSON Schema。

例如 Gemini 的 `config` 信封仍按 `GenerateContentConfig` 过滤、复制，
`config.tools` 暂时借用，随后 `filter_tools` 再按工具 Schema 投影一次。
没有跳过协议覆盖钩子，也没有把整个 config 当作开放字典。其余应用数据仍按各自计划捕获，
不因工具叶节点的委派变成借用结果。整份叶节点被委派时可直接借入，无需先编译该叶节点计划。

## 8. 在真实 generate 路径中的位置

以输入输出和上游都为 OpenAI Chat、且显式设置 stdio_protocol 为例：

| 顺序 | 本次运行的操作 | Schema 计划缓存参与情况 |
| --- | --- | --- |
| 1 | generate 打包完整输入，Encoder 只读规范化消息与工具。 | 不在这里查询计划。 |
| 2 | 输入 Encoder 调用 projected，随后通过 TypeAdapter 验证。 | 首次遇到对应 SDK 类型时编译；后续复用。 |
| 3 | Encoder 编码当前消息与工具，生成本次中间态。 | 不跳过 Encoder。 |
| 4 | 校验生成选项，Decoder 构造当前原生请求。 | 不跳过 Decoder。 |
| 5 | 上游 Decoder 过滤根信封，委派消息/工具叶给 filter_messages / filter_tools 投影并验证。 | 普通消息/工具计划可与步骤2复用；含委派路径的信封计划单独缓存。输入输出协议不同时分别使用对应类型计划。 |
| 6 | BaseCodec.translate 完成语义损失处理；启用 MessageCache 时更新当前已翻译批，再读取最新历史快照。 | 只处理当前增量；缓存历史不重新翻译或校验工具关联。 |
| 7 | 官方 SDK 执行请求、解析返回事件。 | 仍执行 SDK 自己的处理。 |
| 8 | 每个事件经过 Encoder、StreamState、Decoder，再转换完整结果。 | 本计划缓存不替代返回路径，也不缓存响应结果。 |

第一次请求也可能在步骤2编译、步骤5就命中缓存；“冷请求”不代表它内部所有计划查询都未命中。
Encoder / Decoder 对每批消息和工具只解析一次计划并复用其 apply / project 函数，
不会为同一批中的每条数据重复创建缓存键。每条消息仍独立过滤、验证和执行必要的所有权复制。

源码入口见 [generate](../../src/my_llm/my_llm.py)、
[输入验证](../../src/my_llm/codec/base/encoder.py)、
[请求边界](../../src/my_llm/codec/base/decoder.py)和
[Codec 固定编排](../../src/my_llm/codec/base/codec.py)。

## 9. 生命周期、兼容变体与限制

### 进程内共享和内存代价

字段、验证器、SDK 参数注解、根过滤计划、路径声明这五类进程级缓存均使用
`CompiledCache` 封装的 `functools.lru_cache(maxsize=512)`。
每类最多保留 512 个最近使用的入口，超出时淘汰最久未使用项；
被淘汰的规则下次使用时重新编译，不改变输出。没有 TTL，同一进程内不同实例共享，不同进程各自缓存。
限制的是根入口数，不是所有子节点数或总字节数；一个根计划仍可能引用较大的递归类型图。
缓存增长取决于类型声明及其计划，而不是处理过多少条消息。

新增协议不按协议名加入缓存白名单。其 Encoder / Decoder 声明受支持的 TypedDict / Pydantic 类型及
标准容器、Union 等注解后，会自动复用同一套缓存；同名而不同对象的类型不会串用规则。
新协议仍须提供自己的 Encoder / Decoder / Transport，不会从首条请求推断协议。
建议稳定复用声明对象，避免每次请求动态构造新类或新的不可哈希 Annotated 对象。

缓存强引用键与计划，计划又引用类型和子计划。因此，若运行中持续创建新的动态类型，
这些类型在其入口被淘汰前也可能随缓存保留。注册的协议和其他外部对象仍可持有类型引用，
因此 LRU 淘汰不承诺立即释放整个类型。普通新协议与五个内置协议使用相同策略。

### 兼容服务的输入结构变化

- 可选字段增减、字符串与列表切换、不同合法内容块：按当前值执行已有计划。
- 开放应用字典中的任意业务键：按该字段声明整体处理，不按首条数据裁剪。
- SDK 未声明的新协议字段或新块类型：可能被过滤或拒绝，需要在协议所属 Decoder / Codec 明确支持。
- 同一个类型对象在运行中被修改：当前没有自动失效机制，字段缓存、适配器缓存及计划缓存都需要重新建立。

此缓存不根据某次请求自动“学习”新字段。新增字段是否支持，仍由 SDK 类型、协议实现与
供应商配置约束共同决定。当前没有对外提供统一的动态失效接口；常规升级 SDK 或类型声明后，
通过重新启动进程建立相应类型规则。

### 并发与递归类型

编译时先把占位节点登记到局部表，再编译子字段，递归引用会指回同一个节点。
整个根计划编译完成后才从 `_compiled_plan` 返回并进入全局缓存。
过滤计划执行时只读类型规则，请求值保存在每次函数调用的局部变量中。

编译过程没有 await，同一事件循环中不会在编译中途切换请求。
`functools.lru_cache` 保持多线程缓存表的一致性，但并发首次未命中时可能重复执行编译，
不保证所有线程合计只编译一次。这是 [Python LRU 缓存的既有语义](https://docs.python.org/3.12/library/functools.html#functools.lru_cache)。

### `_ModelView` 的转换局部缓存

Encoder 读取 SDK 模型时，`_ModelView` 只按请求的字段名读取属性。
每个视图记录读取值或缺失标记；存在性检查不创建子视图，首次取值才包装嵌套列表或模型，
后续读取复用该包装值。未知且未配置字段不会被枚举、读取或序列化，仍不调用整模型 `model_dump()`。
未设置的 Optional 默认 None 视为缺失，显式 None 仍保留；每次 `get` 的默认实参分别生效。

这是保存本次真实数据的短期视图，与上面的进程级类型计划缓存不同。
下一次公开转换会重新建立视图，同一个 SDK 对象在两次转换之间被修改后，后一次读取新值。
最终 Encoder / Decoder 公开出口的深拷贝继续保证结果隔离。

## 10. 可运行的小例子

在项目根目录执行下面的 Python 代码，可观察同一计划处理两份不同数据。
此段使用私有计划入口作实现诊断；业务代码继续通过 MyLLM / MyCodec 调用。

<!-- runnable-schema-cache-example -->
```python
from typing import Any, Literal, NotRequired
from typing_extensions import TypedDict

from my_llm.cache import compiled_cache


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image"]
    url: str


class Message(TypedDict):
    role: Literal["user"]
    content: str | list[TextPart | ImagePart]
    metadata: NotRequired[dict[str, Any]]


first = {
    "role": "user",
    "content": "北京天气怎么样？",
    "metadata": {"request_id": "R1"},
    "vendor_extra": "drop",
}
second = {
    "role": "user",
    "content": [
        {"type": "text", "text": "上海天气怎么样？"},
        {"type": "image", "url": "https://example.invalid/map.png"},
    ],
    "metadata": {"request_id": "R2", "new_key": {"items": [1, 2]}},
}

first_output = compiled_cache.supported(first, Message)
first_info = compiled_cache._compiled_plan.cache_info()
second_output = compiled_cache.supported(second, Message)
second_info = compiled_cache._compiled_plan.cache_info()
readonly = compiled_cache.projected(second, Message)
plan = compiled_cache._plan(Message)

assert "vendor_extra" not in first_output
assert first_output["content"] == "北京天气怎么样？"
assert second_output["content"][1]["type"] == "image"
assert second_output["metadata"] == second["metadata"]
assert second_output["metadata"] is not second["metadata"]
assert readonly["metadata"] is second["metadata"]
assert compiled_cache._plan(Message) is plan

print("first:", first_output)
print("second:", second_output)
print("after first:", first_info)
print("after second:", second_info)
print("same plan:", compiled_cache._plan(Message) is plan)
print("root callbacks:", plan.apply.__name__, plan.project.__name__, plan.score.__name__)
```

在单独的 Python 进程中运行时，关键输出如下：

```text
after first: CacheInfo(hits=0, misses=1, maxsize=512, currsize=1)
after second: CacheInfo(hits=1, misses=1, maxsize=512, currsize=1)
same plan: True
root callbacks: structured_copy structured_projection structured_score
```

两次输出包含各自文本与 metadata；缓存项数没有因 R1 / R2 或字符串 / 列表变化增加。

## 11. 代码与测试对应

| 要确认的事实 | 实现或证据 |
| --- | --- |
| 缓存键、容量与淘汰 | [compiled_cache.py](../../src/my_llm/cache/compiled_cache.py) 的 `SchemaKey`、`CompiledCache`。 |
| 编译计划中的处理规则 | [compiled_cache.py](../../src/my_llm/cache/compiled_cache.py) 的 `_FilterPlan`、`_compiled_plan`。 |
| 命中后不重复分析字段类型 | [计划测试](../../tests/unit/test_schema_filter_plans.py) 把类型分析函数替换为失败函数，继续执行已缓存计划。 |
| Union 顺序与递归类型保持正确 | 同一测试文件覆盖直接/嵌套 Union 同分顺序、Annotated 与递归模型。 |
| Literal 索引只剪枝，不改变候选选择 | 同一测试文件覆盖共享判别值、无限制候选、多判别字段、SDK 模型及不合法判别值。 |
| 委派路径不会污染其他计划 | 同一测试文件覆盖普通/委派计划隔离、相同类型的不同嵌套位置及整叶直接借入。 |
| 协议结构与开放应用字典 | 同一测试文件覆盖结构字段过滤、递归模型、缓存命中后的可选字段、开放字典新键及复制隔离。 |
| 请求值不会被计划持有 | 同一测试文件用弱引用验证 supported / projected 调用后 payload 可回收。 |
| 只读投影不复制应用数据 | 同一测试文件用禁止 deepcopy 的 payload 及引用身份断言检查。 |
| 编码、解码及结果仍隔离 | [所有权测试](../../tests/unit/test_copy_ownership.py)和[终态物化测试](../../tests/contract/test_stream_materialization.py)。 |
| SDK 字段只在本次转换中缓存 | 所有权测试覆盖重复取值、缺失与显式 None、存在性检查不包装子树，以及下一次转换读取更新值。 |
| 消息/工具只投影一次且钩子仍执行 | [原生请求边界测试](../../tests/unit/test_native_request_boundary.py)覆盖嵌套路径、覆盖钩子、未知字段不读取及副本隔离。 |

缓存策略与同负载试验见[Schema 缓存检查记录](../audits/schema-cache-2026-10-24.md)，
复制清理与性能对照见[代码清理报告](../audits/code-cleanup-2026-10-24.md)，
职责边界见[项目架构](architecture.md#数据所有权与缓存)。
