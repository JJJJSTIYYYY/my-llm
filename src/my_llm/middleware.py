"""SDK-independent, finite protocol models exchanged by encoders and decoders.

Native parsers and renderers use temporary dictionary projections. The public
attributes below are the canonical representation; mapping access supplies those
projections for low-level conversion helpers and offline inspection.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import Literal

from my_llm.codec.shared.conversion import capture
from my_llm.exceptions import MessageValidationError
from my_llm.profile.types import Omit
from my_llm.semantic_loss import originated

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
            mark="encrypted"
            if kind == "redacted_thinking" or encrypted is not None and not reasoning
            else ("summary" if summary is not None else "raw"),
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


@dataclass(slots=True)
class Usage:
    input: int | None = None
    output: int | None = None
    total: int | None = None
    cached: int | None = None
    reasoning: int | None = None

    @classmethod
    def from_record(cls, value: dict | Usage | None) -> Usage | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return replace(value)

        def count(name):
            group = value.get(name)
            if isinstance(group, int):
                return group
            if not isinstance(group, dict) or not isinstance(group.get("base"), int):
                return None
            return group["base"] + sum(group.get("add", {}).values())

        incoming, outgoing = count("input"), count("output")
        total = value.get("total")
        return cls(incoming, outgoing, total, value.get("cached"), value.get("reasoning"))

    def to_record(self) -> dict:
        result = {}
        for name in ("input", "output"):
            if (value := getattr(self, name)) is not None:
                result[name] = {"base": value}
        for name in ("total", "cached", "reasoning"):
            if (value := getattr(self, name)) is not None:
                result[name] = value
        return result


@dataclass(slots=True)
class ItemIdentity:
    id: str | None = None
    type: str | None = None
    status: str | None = None
    name: str | None = None
    call_id: str | None = None

    @classmethod
    def from_wire(cls, value: dict) -> ItemIdentity:
        return cls(*(value.get(key) for key in ("id", "type", "status", "name", "call_id")))

    def to_wire(self) -> dict:
        return {
            key: value
            for key in ("id", "type", "status", "name", "call_id")
            if (value := getattr(self, key)) is not None
        }


@dataclass(slots=True)
class AssistantMessageChunk(ProtocolValue):
    """Finite internal operation; application tool arguments never leave in fragments."""

    type: str
    ref: tuple[str, int, int] | None = None
    block: MessageContent | None = None
    field: str | None = None
    value: str | None = None
    kind: str | None = None
    reason: str | None = None
    allow_tail: tuple[str, ...] = ()
    usage: Usage | None = None
    response_id: str | None = None
    model: str | None = None
    created: int | float | None = None
    blocks: list[tuple[tuple[str, int, int], MessageContent]] = dataclass_field(
        default_factory=list
    )
    path: tuple[str, ...] = ()
    origin: Origin | None = dataclass_field(default=None, repr=False)
    update_fields: tuple[str, ...] = ()
    has_value: bool = False
    created_at: int | float | None = None
    native_stop_reason: str | None = None
    native_stop_sequence: str | None = None
    item_headers: dict[str, ItemIdentity] = dataclass_field(default_factory=dict)

    @classmethod
    def from_wire(cls, raw: dict) -> AssistantMessageChunk | None:
        kind = raw.get("type")
        if kind not in {
            "start",
            "metadata",
            "usage",
            "snapshot",
            "reason",
            "finish",
            "block_set",
            "block_append",
            "block_update",
            "block_delta",
            "block_stop",
        }:
            return None
        if kind == "block_delta" and raw.get("field") not in (
            "text",
            "thinking",
            "arguments",
            "signature",
        ):
            return None
        block = content_from_wire(raw["block"]) if "block" in raw else None
        values = raw.get("values", {})
        path = tuple(raw.get("path", ()))
        headers = {}
        if path and path[0] == "response_items":
            if len(path) == 2:
                headers[path[1]] = ItemIdentity.from_wire(values)
            else:
                headers = {index: ItemIdentity.from_wire(item) for index, item in values.items()}
        elif not path and "response_items" in values:
            headers = {
                index: ItemIdentity.from_wire(item)
                for index, item in values["response_items"].items()
            }
        if (
            kind == "metadata"
            and not headers
            and (
                path
                or not any(
                    key in values
                    for key in (
                        "response_id",
                        "model",
                        "created",
                        "created_at",
                        "native_stop_reason",
                        "native_stop_sequence",
                    )
                )
            )
        ):
            return None
        snapshots = []
        for entry in raw.get("blocks", []):
            if (item := content_from_wire(entry["block"])) is not None:
                snapshots.append((tuple(entry["ref"]), item))
        return cls(
            type=kind,
            ref=tuple(raw["ref"]) if raw.get("ref") is not None else None,
            block=block,
            field=raw.get("field"),
            value=raw.get("value"),
            kind=raw.get("kind"),
            reason=raw.get("reason"),
            allow_tail=tuple(raw.get("allow_tail", ())),
            usage=Usage.from_record(raw.get("usage")),
            response_id=values.get("response_id"),
            model=values.get("model"),
            created=values.get("created"),
            blocks=snapshots,
            path=tuple(raw.get("path", ())),
            origin=Origin.from_value(raw, "$.chunks"),
            update_fields=tuple(raw.get("block", {})),
            has_value="value" in raw,
            created_at=values.get("created_at"),
            native_stop_reason=values.get("native_stop_reason"),
            native_stop_sequence=values.get("native_stop_sequence"),
            item_headers=headers,
        )

    def to_wire(self) -> dict:
        result = {"type": self.type}
        for name in ("ref", "field", "value", "kind", "reason"):
            if (value := getattr(self, name)) is not None:
                result[name] = value
        if self.has_value:
            result["value"] = self.value
        if self.block is not None:
            result["block"] = self.block.to_wire()
            if self.type == "block_update":
                result["block"] = {
                    key: value
                    for key, value in result["block"].items()
                    if key in self.update_fields
                }
        if self.type == "metadata":
            result["values"] = {
                name: value
                for name in (
                    "response_id",
                    "model",
                    "created",
                    "created_at",
                    "native_stop_reason",
                    "native_stop_sequence",
                )
                if (value := getattr(self, name)) is not None
            }
            result["path"] = self.path
            if self.item_headers:
                if len(self.path) == 2:
                    result["values"] = self.item_headers[self.path[1]].to_wire()
                elif self.path:
                    result["values"] = {
                        index: item.to_wire() for index, item in self.item_headers.items()
                    }
                else:
                    result["values"]["response_items"] = {
                        index: item.to_wire() for index, item in self.item_headers.items()
                    }
        if self.type == "usage":
            result["usage"] = self.usage.to_record() if self.usage is not None else None
        if self.type == "snapshot":
            result["blocks"] = [
                {"ref": ref, "block": block.to_wire()} for ref, block in self.blocks
            ]
        if self.allow_tail:
            result["allow_tail"] = list(self.allow_tail)
        return _located(result, self.origin)


@dataclass(slots=True)
class StreamBatch(ProtocolValue):
    protocol: str
    events: list[AssistantMessageChunk]

    def to_wire(self) -> dict:
        return {
            "type": "batch",
            "protocol": self.protocol,
            "events": [e.to_wire() for e in self.events],
        }


@dataclass(slots=True)
class MessageEvent(ProtocolValue):
    """Public middleware stream event after tool assembly and EOF validation.

    Text/reasoning deltas use ``field`` and ``value``. A ``tool`` event holds a
    complete ToolCall in ``block`` and arrives only after source EOF. ``finish``
    contains the completed assistant history in ``messages``. Usage is available
    separately through MyLLM.get_last_usage(). Each event owns its mutable values.
    """

    type: Literal["start", "block_start", "delta", "block_stop", "block", "tool", "finish"]
    index: int | None = None
    source_ref: tuple[str, int, int] | None = None
    block: MessageContent | None = None
    tool_index: int | None = None
    field: Literal["text", "thinking"] | None = None
    value: str | None = None
    reason: str | None = None
    messages: list[Message] | None = None

    @classmethod
    def from_wire(cls, event: dict) -> MessageEvent:
        """Project a normalized StreamState event; raw encoder operations differ."""
        return cls(
            type=event["type"],
            index=event.get("index"),
            source_ref=event.get("source_ref"),
            block=content_from_wire(event["block"]) if "block" in event else None,
            tool_index=event.get("tool_index"),
            field=event.get("field"),
            value=event.get("value"),
            reason=event.get("reason"),
            messages=messages_from_wire(event["messages"]) if "messages" in event else None,
        )

    def to_wire(self) -> dict:
        result = {"type": self.type}
        for name in ("index", "source_ref", "tool_index", "field", "value", "reason"):
            if (value := getattr(self, name)) is not None:
                result[name] = value
        if self.block is not None:
            result["block"] = self.block.to_wire()
        if self.messages is not None:
            result["messages"] = [message.to_wire() for message in self.messages]
        return result


def stream_from_wire(values: list) -> list[StreamBatch]:
    result = []
    for value in values:
        events = [
            event
            for raw in value["events"]
            if (event := AssistantMessageChunk.from_wire(raw)) is not None
        ]
        result.append(StreamBatch(value["protocol"], events))
    return result


@dataclass(slots=True)
class ProtocolMiddleWare(ProtocolValue):
    messages: list[Message] | None = None
    tools: list[FunctionTool] | None = None
    stream: list[StreamBatch] | None = None
    usage: Usage | None = None

    def __iter__(self):
        for key, value in (
            ("messages", self.messages),
            ("tools", self.tools),
            ("chunks", self.stream),
            ("usage", self.usage),
        ):
            if value is not None:
                yield key

    def __len__(self):
        return sum(
            value is not None for value in (self.messages, self.tools, self.stream, self.usage)
        )

    def __getitem__(self, key):
        if key == "messages" and self.messages is not None:
            return [value.to_wire() for value in self.messages]
        if key == "tools" and self.tools is not None:
            return [value.to_wire() for value in self.tools]
        if key == "chunks" and self.stream is not None:
            return [value.to_wire() for value in self.stream]
        if key == "usage" and self.usage is not None:
            return self.usage.to_record()
        raise KeyError(key)

    def to_wire(self) -> dict:
        return {key: self[key] for key in self}


def wire_values(values: list) -> list:
    return [v.to_wire() if isinstance(v, ProtocolValue) else v for v in values]


def _valid_content(part: MessageContent, role: str) -> bool:
    if isinstance(part, TextContent):
        return isinstance(part.text, str)
    if isinstance(part, ImageContent):
        source = part.source
        return (
            role == "user"
            and isinstance(source, ImageSource)
            and (
                source.type == "url"
                and isinstance(source.url, str)
                and bool(source.url)
                or source.type == "base64"
                and isinstance(source.data, str)
                and isinstance(source.media_type, str)
                and bool(source.media_type)
            )
        )
    if isinstance(part, ReasoningContent):
        return (
            role == "assistant"
            and isinstance(part.reasoning, str)
            and part.mark in ("raw", "summary", "encrypted")
            and all(
                value is None or isinstance(value, str)
                for value in (part.signature, part.encrypted_content, part.opaque_protocol, part.id)
            )
            and (
                part.summary is None
                or isinstance(part.summary, list)
                and all(isinstance(value, str) for value in part.summary)
            )
        )
    if isinstance(part, ToolCall):
        return (
            role == "assistant"
            and isinstance(part.name, str)
            and bool(part.name)
            and isinstance(part.arguments, dict)
        )
    if isinstance(part, ToolResult):
        return (
            role == "user"
            and isinstance(part.content, list)
            and all(
                isinstance(value, (TextContent, ImageContent)) and _valid_content(value, "user")
                for value in part.content
            )
        )
    return False


def _input_origin(part: MessageContent, path: str) -> MessageContent:
    content = {}
    if isinstance(part, ToolResult):
        children = [
            _input_origin(value, f"{path}.content[{index}]")
            for index, value in enumerate(part.content)
        ]
        if any(new is not old for new, old in zip(children, part.content, strict=True)):
            content["content"] = children
    if part.origin is None:
        content["origin"] = Origin(path, part)
    return replace(part, **content) if content else part


def generation_input(messages, tools) -> ProtocolMiddleWare:
    """Validate typed generation input and attach request-local provenance.

    Shallow model envelopes record caller objects without mutating them. JSON
    leaves are borrowed until the decoder captures the independently owned SDK
    request; no additional application-data copy is needed here.
    """
    if not isinstance(messages, (list, tuple)) or not messages:
        raise MessageValidationError("messages must be a non-empty list or tuple")
    history = []
    for index, message in enumerate(messages):
        if not isinstance(message, Message) or message.role not in (
            "user",
            "assistant",
            "system",
            "developer",
        ):
            raise MessageValidationError(
                "middleware messages require Message models and valid roles"
            )
        if not isinstance(message.content, list) or not all(
            _valid_content(part, message.role) for part in message.content
        ):
            raise MessageValidationError("invalid middleware content or content role")
        path = f"$.messages[{index}]"
        parts = [
            _input_origin(part, f"{path}.content[{offset}]")
            for offset, part in enumerate(message.content)
        ]
        history.append(
            replace(
                message,
                content=parts,
                origin=message.origin if message.origin is not None else Origin(path, message),
            )
            if message.origin is None
            or any(new is not old for new, old in zip(parts, message.content, strict=True))
            else message
        )
    values = None
    if tools is not None and not isinstance(tools, Omit):
        if not isinstance(tools, (list, tuple)) or any(
            not isinstance(tool, FunctionTool)
            or not isinstance(tool.name, str)
            or not tool.name
            or tool.description is not None
            and not isinstance(tool.description, str)
            or tool.parameters is not None
            and not isinstance(tool.parameters, dict)
            or tool.strict is not None
            and not isinstance(tool.strict, bool)
            for tool in tools
        ):
            raise MessageValidationError("middleware tools require FunctionTool models")
        values = [
            tool
            if tool.origin is not None
            else replace(tool, origin=Origin(f"$.tools[{index}]", tool))
            for index, tool in enumerate(tools)
        ]
    return ProtocolMiddleWare(messages=history, tools=values)
