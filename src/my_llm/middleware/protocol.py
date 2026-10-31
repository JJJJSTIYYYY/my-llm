"""Complete middleware envelopes and validation of typed generation input."""

from __future__ import annotations

from dataclasses import dataclass, replace

from my_llm.basic.exceptions import MessageValidationError
from my_llm.middleware.models import (
    FunctionTool,
    ImageContent,
    ImageSource,
    ItemIdentity,
    Message,
    MessageContent,
    Origin,
    ProtocolValue,
    ReasoningContent,
    TextContent,
    ToolCall,
    ToolResult,
)
from my_llm.middleware.stream import StreamBatch, Usage
from my_llm.middleware.validation import REASONING_MARKS, RESULT_CONTENT_TYPES
from my_llm.profile.types import Omit


@dataclass(slots=True)
class ProtocolMiddleWare(ProtocolValue):
    messages: list[Message] | None = None
    tools: list[FunctionTool] | None = None
    stream: list[StreamBatch] | None = None
    usage: Usage | None = None

    def __iter__(self):
        """Yield the wire keys of the collections present in this envelope."""
        for key, value in (
            ("messages", self.messages),
            ("tools", self.tools),
            ("chunks", self.stream),
            ("usage", self.usage),
        ):
            if value is not None:
                yield key

    def __len__(self):
        """Count the collections present in this envelope."""
        return sum(
            value is not None for value in (self.messages, self.tools, self.stream, self.usage)
        )

    def __getitem__(self, key):
        """Project one present collection to wire values, or raise KeyError."""
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
        """Project all present collections to their canonical wire representation."""
        return {key: self[key] for key in self}


def wire_values(values: list) -> list:
    """Project typed or raw middleware collections into a list of dictionaries.

    Accept a list or tuple containing ProtocolValue models or dictionaries.
    Models create wire envelopes; raw dictionaries and nested application
    leaves remain borrowed. Reject invalid containers/items with
    MessageValidationError. Callers still need the appropriate structural
    validator before interpreting the projected values.
    """
    if not isinstance(values, (list, tuple)):
        raise MessageValidationError("$: expected a middleware list or tuple")
    result = []
    for index, value in enumerate(values):
        projected = value.to_wire() if isinstance(value, ProtocolValue) else value
        if not isinstance(projected, dict):
            raise MessageValidationError(f"$[{index}]: expected a middleware object")
        result.append(projected)
    return result


def _valid_content(part: MessageContent, role: str) -> bool:
    """Check a typed content model and whether its role permits that content."""
    if not isinstance(part, (TextContent, ImageContent, ReasoningContent, ToolCall, ToolResult)):
        return False
    if not _valid_item(part.item):
        return False
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
            and isinstance(part.redacted, bool)
            and part.mark in REASONING_MARKS
            and all(
                value is None or isinstance(value, str)
                for value in (part.signature, part.encrypted_content, part.opaque_protocol)
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
                getattr(value, "type", None) in RESULT_CONTENT_TYPES
                and _valid_content(value, "user")
                for value in part.content
            )
        )
    return False


def _valid_item(item: ItemIdentity | None) -> bool:
    """Accept absent item identity or string-valued identity fields."""
    return (
        item is None
        or isinstance(item, ItemIdentity)
        and all(value is None or isinstance(value, str) for value in (item.id, item.status))
    )


def _input_origin(part: MessageContent, path: str) -> MessageContent:
    """Attach missing provenance through shallow replacements, including result parts."""
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

    Require a nonempty list/tuple of Message models with valid roles and
    role-appropriate content. Tools must be FunctionTool models; None or Omit
    leaves tools absent, while an empty collection stays explicitly empty.
    Malformed models or content raise MessageValidationError.

    Shallow model envelopes record caller objects without mutating them. JSON
    leaves are borrowed until the decoder captures the independently owned SDK
    request. This function checks typed shapes and roles; call IDs, duplicate
    results and missing results are checked later by validate_history.
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
        if (
            not _valid_item(message.item)
            or not isinstance(message.content, list)
            or not all(_valid_content(part, message.role) for part in message.content)
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
