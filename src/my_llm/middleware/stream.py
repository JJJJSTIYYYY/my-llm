"""Typed stream operations, public events, item identities and usage counters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from my_llm.middleware.models import (
    Message,
    MessageContent,
    ProtocolValue,
    content_from_validated_wire,
    messages_from_validated_wire,
)
from my_llm.middleware.usage import normalize_usage
from my_llm.middleware.validation import validate_events, validate_operations


@dataclass(slots=True)
class Usage:
    input: int | None = None
    output: int | None = None
    total: int | None = None
    cached: int | None = None
    reasoning: int | None = None

    @classmethod
    def from_record(cls, value: dict | Usage | None) -> Usage | None:
        """Flatten reported usage groups into a new Usage model, preserving absent counters."""
        if value is None:
            return None
        if isinstance(value, cls):
            value = value.to_record()
        if isinstance(value, dict) and not value:
            return cls()
        value = normalize_usage(value)
        if value is None:
            return None

        def count(name):
            """Read one flat or grouped usage counter, including additive counts."""
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
        """Project reported usage counters into canonical base-count groups."""
        result = {}
        for name in ("input", "output"):
            if (value := getattr(self, name)) is not None:
                result[name] = {"base": value}
        for name in ("total", "cached", "reasoning"):
            if (value := getattr(self, name)) is not None:
                result[name] = value
        return result


@dataclass(slots=True)
class StreamBatch(ProtocolValue):
    """Encoder operations for the state machine, separate from public MessageEvent.

    Operations are owned by the encoder and carried without reinterpreting their
    metadata or constructing models for incomplete block patches. Wire projections
    borrow these operations within the conversion pipeline.
    """

    protocol: str
    events: list[dict]

    def to_wire(self) -> dict:
        """Expose the batch envelope while borrowing its already-owned operations."""
        return {"type": "batch", "protocol": self.protocol, "events": self.events}


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
        """Read a public event, raising MessageValidationError for malformed wire data."""
        validate_events((event,))
        return cls.from_validated_wire(event)

    @classmethod
    def from_validated_wire(cls, event: dict) -> MessageEvent:
        """Project an event already checked by validate_events without revalidating it."""
        return cls(
            type=event["type"],
            index=event.get("index"),
            source_ref=event.get("source_ref"),
            block=content_from_validated_wire(event["block"]) if "block" in event else None,
            tool_index=event.get("tool_index"),
            field=event.get("field"),
            value=event.get("value"),
            reason=event.get("reason"),
            messages=messages_from_validated_wire(event["messages"])
            if "messages" in event
            else None,
        )

    def to_wire(self) -> dict:
        """Project present public event fields and nested typed content/messages."""
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
    """Check core operations; optional accounting/metadata is parsed by runtime handlers."""
    validate_operations(values, tolerate_auxiliary=True)
    return [StreamBatch(value["protocol"], value["events"]) for value in values]
