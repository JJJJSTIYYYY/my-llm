"""Complete middleware envelopes and validation of typed generation input."""

from __future__ import annotations

from dataclasses import dataclass, replace

from my_llm.basic.exceptions import MessageValidationError
from my_llm.middleware.models import (
    FunctionTool,
    ImageContent,
    ImageSource,
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
from my_llm.profile.types import Omit


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
