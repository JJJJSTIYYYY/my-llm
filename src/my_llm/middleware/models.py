"""SDK-independent message, content and function-tool models with wire projections."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Literal

from my_llm.basic.data import capture
from my_llm.basic.semantic_loss import originated

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


@dataclass(slots=True)
class Origin:
    path: str
    value: object = dataclass_field(compare=False)

    @classmethod
    def from_value(cls, value: object, path: str) -> Origin:
        return cls(
            getattr(value, "origin_path", path),
            getattr(value, "origin_value", value),
        )


class ProtocolValue(Mapping[str, object]):
    """Read-only mapping projection; model attributes remain directly editable."""

    __slots__ = ()

    def to_wire(self) -> dict:
        raise NotImplementedError

    def __getitem__(self, key: str):
        return self.to_wire()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_wire())

    def __len__(self) -> int:
        return len(self.to_wire())


def _located(value: dict, origin: Origin | None) -> dict:
    return originated(value, origin.path, origin.value) if origin is not None else value


@dataclass(slots=True)
class ImageSource:
    type: Literal["url", "base64"]
    url: str | None = None
    data: str | None = None
    media_type: str | None = None

    def to_wire(self) -> dict:
        return {
            "type": self.type,
            **{
                key: value
                for key, value in (
                    ("url", self.url),
                    ("data", self.data),
                    ("media_type", self.media_type),
                )
                if value is not None
            },
        }


@dataclass(slots=True)
class TextContent(ProtocolValue):
    text: str
    origin: Origin | None = dataclass_field(default=None, repr=False)
    type: Literal["text"] = dataclass_field(default="text", init=False)
    item_id: str | None = None
    item_status: str | None = None

    def to_wire(self) -> dict:
        value = {"type": "text", "text": self.text}
        for key in ("item_id", "item_status"):
            if (item := getattr(self, key)) is not None:
                value[key] = item
        return _located(value, self.origin)


@dataclass(slots=True)
class ImageContent(ProtocolValue):
    source: ImageSource
    origin: Origin | None = dataclass_field(default=None, repr=False)
    type: Literal["image"] = dataclass_field(default="image", init=False)

    def to_wire(self) -> dict:
        value = {"type": "image", "source": self.source.to_wire()}
        return _located(value, self.origin)


@dataclass(slots=True)
class ReasoningContent(ProtocolValue):
    reasoning: str = ""
    mark: Literal["raw", "summary", "encrypted"] = "raw"
    signature: str | None = None
    encrypted_content: str | None = None
    opaque_protocol: str | None = None
    id: str | None = None
    origin: Origin | None = dataclass_field(default=None, repr=False)
    type: Literal["thinking"] = dataclass_field(default="thinking", init=False)
    summary: list[str] | None = None
    redacted: bool = False
    status: str | None = None

    def to_wire(self) -> dict:
        if self.redacted:
            value = {"type": "redacted_thinking", "data": self.encrypted_content or ""}
        else:
            value = {"type": "thinking", "thinking": self.reasoning}
            if self.encrypted_content is not None:
                value["encrypted_content"] = self.encrypted_content
            if self.signature is not None:
                value["signature"] = self.signature
            if self.summary is not None:
                value["summary"] = [{"type": "summary_text", "text": s} for s in self.summary]
        if self.id is not None:
            value["id"] = self.id
        if self.status is not None:
            value["status"] = self.status
        if self.opaque_protocol is not None:
            value["protocol"] = self.opaque_protocol
        value["mark"] = self.mark
        return _located(value, self.origin)


@dataclass(slots=True)
class ToolCall(ProtocolValue):
    call_id: str | None
    name: str
    arguments: JsonObject = dataclass_field(default_factory=dict)
    signature: str | None = None
    opaque_protocol: str | None = None
    origin: Origin | None = dataclass_field(default=None, repr=False)
    type: Literal["tool_use"] = dataclass_field(default="tool_use", init=False)
    item_id: str | None = None
    status: str | None = None

    def to_wire(self) -> dict:
        value = {"type": "tool_use", "id": self.call_id, "name": self.name, "input": self.arguments}
        if self.signature is not None:
            value["signature"] = self.signature
        if self.opaque_protocol is not None:
            value["protocol"] = self.opaque_protocol
        for key in ("item_id", "status"):
            if (item := getattr(self, key)) is not None:
                value[key] = item
        return _located(value, self.origin)


type ResultContent = TextContent | ImageContent


@dataclass(slots=True)
class ToolResult(ProtocolValue):
    call_id: str | None
    content: list[ResultContent]
    name: str | None = None
    origin: Origin | None = dataclass_field(default=None, repr=False)
    type: Literal["tool_result"] = dataclass_field(default="tool_result", init=False)

    def to_wire(self) -> dict:
        value = {
            "type": "tool_result",
            "tool_use_id": self.call_id,
            "content": [part.to_wire() for part in self.content],
        }
        if self.name is not None:
            value["name"] = self.name
        return _located(value, self.origin)


type MessageContent = TextContent | ImageContent | ReasoningContent | ToolCall | ToolResult


def _result_content(value: object, path: str, protocol: str | None) -> list[ResultContent]:
    if value is None:
        return []
    if isinstance(value, str):
        return [TextContent(value, Origin(path, value))]
    if isinstance(value, list) and all(isinstance(v, dict) and "type" in v for v in value):
        return [
            part
            for index, raw in enumerate(value)
            if isinstance(
                part := content_from_wire(raw, f"{path}[{index}]", protocol),
                (TextContent, ImageContent),
            )
        ]
    # Objects and arrays of application values are supported text results.
    return [TextContent(json.dumps(value, ensure_ascii=False), Origin(path, value))]


def content_from_wire(
    value: dict, path: str = "$", protocol: str | None = None
) -> MessageContent | None:
    kind = value.get("type")
    origin = Origin.from_value(value, path)
    if kind == "text":
        return TextContent(
            value.get("text", ""), origin, value.get("item_id"), value.get("item_status")
        )
    if kind == "image":
        source = value.get("source", {})
        if source.get("type") not in ("url", "base64"):
            return None
        return ImageContent(
            ImageSource(
                source["type"], source.get("url"), source.get("data"), source.get("media_type")
            ),
            origin,
        )
    if kind in ("thinking", "redacted_thinking"):
        summary = value.get("summary")
        reasoning = value.get("thinking", "")
        encrypted = (
            value.get("data") if kind == "redacted_thinking" else value.get("encrypted_content")
        )
        summary_text = (
            [item.get("text", "") for item in summary if item.get("type") == "summary_text"]
            if isinstance(summary, list)
            else None
        )
        return ReasoningContent(
            reasoning=reasoning,
            mark=value["mark"]
            if "mark" in value
            else (
                "encrypted"
                if kind == "redacted_thinking" or encrypted is not None and not reasoning
                else ("summary" if summary is not None else "raw")
            ),
            signature=value.get("signature"),
            encrypted_content=encrypted,
            opaque_protocol=value.get("protocol", protocol),
            id=value.get("id"),
            origin=origin,
            summary=summary_text,
            redacted=kind == "redacted_thinking",
            status=value.get("status"),
        )
    if kind == "tool_use":
        return ToolCall(
            value.get("id"),
            value.get("name", ""),
            capture(value.get("input", {})),
            value.get("signature"),
            value.get("protocol", protocol),
            origin,
            value.get("item_id"),
            value.get("status"),
        )
    if kind == "tool_result":
        return ToolResult(
            value.get("tool_use_id"),
            _result_content(value.get("content"), f"{origin.path}.content", protocol),
            value.get("name"),
            origin,
        )
    return None


@dataclass(slots=True)
class Message(ProtocolValue):
    role: str
    content: list[MessageContent] = dataclass_field(default_factory=list)
    name: str | None = None
    protocol: str | None = None
    id: str | None = None
    status: str | None = None
    coalesce: bool = False
    origin: Origin | None = dataclass_field(default=None, repr=False)

    def to_wire(self) -> dict:
        result = {"role": self.role, "content": [part.to_wire() for part in self.content]}
        for key in ("name", "protocol", "id", "status"):
            if (value := getattr(self, key)) is not None:
                result[key] = value
        if self.coalesce:
            result["coalesce"] = True
        return _located(result, self.origin)


class UserMessage(Message):
    """User text and image content."""

    __slots__ = ()


class AssistantMessage(Message):
    """Assistant text, separate reasoning, and complete tool calls."""

    __slots__ = ()


class InstructionMessage(Message):
    """System or developer instructions."""

    __slots__ = ()


class ToolMessage(Message):
    """A tool result associated with a prior call."""

    __slots__ = ()

    @property
    def call_id(self) -> str | None:
        return next((p.call_id for p in self.content if isinstance(p, ToolResult)), None)


def messages_from_wire(values: list) -> list[Message]:
    result = []
    for index, raw in enumerate(values):
        role = raw["role"]
        protocol = raw.get("protocol")
        origin = Origin.from_value(raw, f"$.messages[{index}]")
        content = [
            part
            for offset, block in enumerate(raw["content"])
            if (part := content_from_wire(block, f"{origin.path}.content[{offset}]", protocol))
            is not None
        ]
        model = (
            AssistantMessage
            if role == "assistant"
            else InstructionMessage
            if role in ("system", "developer")
            else ToolMessage
            if any(isinstance(p, ToolResult) for p in content)
            else UserMessage
        )
        result.append(
            model(
                role,
                content,
                raw.get("name"),
                protocol,
                raw.get("id"),
                raw.get("status"),
                bool(raw.get("coalesce")),
                origin,
            )
        )
    return result


@dataclass(slots=True)
class FunctionTool(ProtocolValue):
    name: str
    description: str | None = None
    parameters: JsonObject | None = None
    strict: bool | None = None
    origin: Origin | None = dataclass_field(default=None, repr=False)

    def to_wire(self) -> dict:
        function = {"name": self.name}
        for key in ("description", "parameters", "strict"):
            if (value := getattr(self, key)) is not None:
                function[key] = value
        return _located({"functions": [_located(function, self.origin)]}, self.origin)


def tools_from_wire(values: list) -> list[FunctionTool]:
    result = []
    for index, tool in enumerate(values):
        for fn in tool["functions"]:
            result.append(
                FunctionTool(
                    fn.get("name", ""),
                    fn.get("description"),
                    capture(fn.get("parameters")),
                    fn.get("strict"),
                    Origin.from_value(fn, f"$.tools[{index}]"),
                )
            )
    return result
