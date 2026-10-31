"""SDK-independent message, content and function-tool models with wire projections."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Literal

from my_llm.basic.data import capture
from my_llm.basic.semantic_loss import originated
from my_llm.middleware.validation import validate_content, validate_messages, validate_tools

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


@dataclass(slots=True)
class Origin:
    path: str
    value: object = dataclass_field(compare=False)

    @classmethod
    def from_value(cls, value: object, path: str) -> Origin:
        """Preserve existing provenance or locate an unannotated canonical value."""
        return cls(
            getattr(value, "origin_path", path),
            getattr(value, "origin_value", value),
        )


class ProtocolValue(Mapping[str, object]):
    """Read-only mapping projection; model attributes remain directly editable."""

    __slots__ = ()

    def to_wire(self) -> dict:
        """Return the canonical dictionary projection implemented by each model."""
        raise NotImplementedError

    def __getitem__(self, key: str):
        """Read a field from this model's canonical mapping projection."""
        return self.to_wire()[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate field names in this model's canonical mapping projection."""
        return iter(self.to_wire())

    def __len__(self) -> int:
        """Count fields in this model's canonical mapping projection."""
        return len(self.to_wire())


def _located(value: dict, origin: Origin | None) -> dict:
    """Attach optional source provenance outside the wire payload keys."""
    return originated(value, origin.path, origin.value) if origin is not None else value


@dataclass(slots=True)
class ItemIdentity:
    """Identity of a source output item, distinct from a tool call's call_id."""

    id: str | None = None
    status: str | None = None

    def to_wire(self) -> dict:
        """Project present item identity fields, omitting None values."""
        return {key: value for key in ("id", "status") if (value := getattr(self, key)) is not None}


def _item(value: dict) -> ItemIdentity | None:
    """Read optional native item identity from a canonical dictionary."""
    item = value.get("item")
    return ItemIdentity(item.get("id"), item.get("status")) if item is not None else None


def _with_item(value: dict, item: ItemIdentity | None) -> dict:
    """Attach an item identity projection to a newly built wire envelope."""
    if item is not None:
        value["item"] = item.to_wire()
    return value


@dataclass(slots=True)
class ImageSource:
    type: Literal["url", "base64"]
    url: str | None = None
    data: str | None = None
    media_type: str | None = None

    def to_wire(self) -> dict:
        """Project the source tag and present URL/base64 metadata."""
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
    item: ItemIdentity | None = None

    def to_wire(self) -> dict:
        """Project text with optional item identity and source provenance."""
        return _located(_with_item({"type": "text", "text": self.text}, self.item), self.origin)


@dataclass(slots=True)
class ImageContent(ProtocolValue):
    source: ImageSource
    origin: Origin | None = dataclass_field(default=None, repr=False)
    type: Literal["image"] = dataclass_field(default="image", init=False)
    item: ItemIdentity | None = None

    def to_wire(self) -> dict:
        """Project image source with optional item identity and provenance."""
        value = {"type": "image", "source": self.source.to_wire()}
        return _located(_with_item(value, self.item), self.origin)


@dataclass(slots=True)
class ReasoningContent(ProtocolValue):
    reasoning: str = ""
    mark: Literal["raw", "summary", "encrypted"] = "raw"
    signature: str | None = None
    encrypted_content: str | None = None
    opaque_protocol: str | None = None
    item: ItemIdentity | None = None
    origin: Origin | None = dataclass_field(default=None, repr=False)
    type: Literal["thinking"] = dataclass_field(default="thinking", init=False)
    summary: list[str] | None = None
    redacted: bool = False

    def to_wire(self) -> dict:
        """Project recognized reasoning text, opaque data, signatures and native ownership."""
        value = {"type": "thinking", "thinking": self.reasoning, "mark": self.mark}
        for key in ("encrypted_content", "signature", "summary"):
            if (child := getattr(self, key)) is not None:
                value[key] = child
        if self.redacted:
            value["redacted"] = True
        if self.opaque_protocol is not None:
            value["protocol"] = self.opaque_protocol
        return _located(_with_item(value, self.item), self.origin)


@dataclass(slots=True)
class ToolCall(ProtocolValue):
    call_id: str | None
    name: str
    arguments: JsonObject = dataclass_field(default_factory=dict)
    signature: str | None = None
    opaque_protocol: str | None = None
    origin: Origin | None = dataclass_field(default=None, repr=False)
    type: Literal["tool_use"] = dataclass_field(default="tool_use", init=False)
    item: ItemIdentity | None = None

    def to_wire(self) -> dict:
        """Project tool identity/name while borrowing the argument object inside the pipeline."""
        value = {"type": "tool_use", "id": self.call_id, "name": self.name, "input": self.arguments}
        if self.signature is not None:
            value["signature"] = self.signature
        if self.opaque_protocol is not None:
            value["protocol"] = self.opaque_protocol
        return _located(_with_item(value, self.item), self.origin)


type ResultContent = TextContent | ImageContent


@dataclass(slots=True)
class ToolResult(ProtocolValue):
    call_id: str | None
    content: list[ResultContent]
    name: str | None = None
    origin: Origin | None = dataclass_field(default=None, repr=False)
    type: Literal["tool_result"] = dataclass_field(default="tool_result", init=False)
    item: ItemIdentity | None = None

    def to_wire(self) -> dict:
        """Project the associated call ID and canonical text/image result parts."""
        value = {
            "type": "tool_result",
            "tool_use_id": self.call_id,
            "content": [part.to_wire() for part in self.content],
        }
        if self.name is not None:
            value["name"] = self.name
        return _located(_with_item(value, self.item), self.origin)


type MessageContent = TextContent | ImageContent | ReasoningContent | ToolCall | ToolResult


def _result_content(value: object, path: str, protocol: str | None) -> list[ResultContent]:
    """Normalize tool results to typed parts, JSON-serializing untagged application data."""
    if value is None:
        return []
    if isinstance(value, str):
        return [TextContent(value, Origin(path, value))]
    if isinstance(value, list) and all(isinstance(v, dict) and "type" in v for v in value):
        return [
            content_from_validated_wire(raw, f"{path}[{index}]", protocol)
            for index, raw in enumerate(value)
        ]
    # Objects and arrays of application values are supported text results.
    return [TextContent(json.dumps(value, ensure_ascii=False), Origin(path, value))]


def content_from_wire(value: dict, path: str = "$", protocol: str | None = None) -> MessageContent:
    """Read canonical content; invalid structure raises MessageValidationError."""
    validate_content(value, path)
    return content_from_validated_wire(value, path, protocol)


def content_from_validated_wire(
    value: dict, path: str = "$", protocol: str | None = None
) -> MessageContent:
    """Project content already checked by validate_content, without revalidation."""
    kind = value["type"]
    origin = Origin.from_value(value, path)
    if kind == "text":
        return TextContent(value["text"], origin, _item(value))
    if kind == "image":
        source = value["source"]
        return ImageContent(
            ImageSource(
                source["type"], source.get("url"), source.get("data"), source.get("media_type")
            ),
            origin,
            _item(value),
        )
    if kind == "thinking":
        return ReasoningContent(
            reasoning=value["thinking"],
            mark=value["mark"],
            signature=value.get("signature"),
            encrypted_content=value.get("encrypted_content"),
            opaque_protocol=value.get("protocol", protocol),
            item=_item(value),
            origin=origin,
            summary=capture(value.get("summary")),
            redacted=value.get("redacted", False),
        )
    if kind == "tool_use":
        return ToolCall(
            value["id"],
            value["name"],
            capture(value["input"]),
            value.get("signature"),
            value.get("protocol", protocol),
            origin,
            _item(value),
        )
    return ToolResult(
        value["tool_use_id"],
        _result_content(value["content"], f"{origin.path}.content", protocol),
        value.get("name"),
        origin,
        _item(value),
    )


@dataclass(slots=True)
class Message(ProtocolValue):
    role: str
    content: list[MessageContent] = dataclass_field(default_factory=list)
    name: str | None = None
    protocol: str | None = None
    item: ItemIdentity | None = None
    coalesce: bool = False
    origin: Origin | None = dataclass_field(default=None, repr=False)

    def to_wire(self) -> dict:
        """Project the message envelope and typed content while retaining source metadata."""
        result = {"role": self.role, "content": [part.to_wire() for part in self.content]}
        for key in ("name", "protocol"):
            if (value := getattr(self, key)) is not None:
                result[key] = value
        if self.coalesce:
            result["coalesce"] = True
        return _located(_with_item(result, self.item), self.origin)


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
        """Read the first tool-result call ID, or None when no result exists."""
        return next((p.call_id for p in self.content if isinstance(p, ToolResult)), None)


def messages_from_wire(values: list) -> list[Message]:
    """Validate canonical message structure and build typed middleware models.

    Preserve source paths and protocol/item identity. Model readers capture
    mutable argument/schema data where required; untagged tool-result values
    become text parts. This boundary checks wire structure, while generation
    history association checks remain in BaseDecoder.request.
    """
    validate_messages(values)
    return messages_from_validated_wire(values)


def messages_from_validated_wire(values: list) -> list[Message]:
    """Project history already checked by validate_messages, without revalidating it."""
    result = []
    for index, raw in enumerate(values):
        role = raw["role"]
        protocol = raw.get("protocol")
        origin = Origin.from_value(raw, f"$.messages[{index}]")
        content = [
            content_from_validated_wire(block, f"{origin.path}.content[{offset}]", protocol)
            for offset, block in enumerate(raw["content"])
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
                _item(raw),
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
        """Project a function as a canonical function group with its source provenance."""
        function = {"name": self.name}
        for key in ("description", "parameters", "strict"):
            if (value := getattr(self, key)) is not None:
                function[key] = value
        return _located({"functions": [_located(function, self.origin)]}, self.origin)


def tools_from_wire(values: list) -> list[FunctionTool]:
    """Read canonical function declarations without swallowing application errors."""
    validate_tools(values)
    result = []
    for index, tool in enumerate(values):
        for fn in tool["functions"]:
            result.append(
                FunctionTool(
                    fn["name"],
                    fn.get("description"),
                    capture(fn.get("parameters")),
                    fn.get("strict"),
                    Origin.from_value(fn, f"$.tools[{index}]"),
                )
            )
    return result
