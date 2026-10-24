"""Typed stream operations, public events, item identities and usage counters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import Literal

from my_llm.middleware.models import (
    Message,
    MessageContent,
    Origin,
    ProtocolValue,
    _located,
    content_from_wire,
    messages_from_wire,
)


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
