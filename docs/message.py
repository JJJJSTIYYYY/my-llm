"""Provider-independent messages used by the my-llm codec.

The public message format follows OpenAI Chat Completions. Codecs translate
between this format and each provider's request and response protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, NotRequired, TypedDict


MessageRole = Literal["system", "user", "assistant", "tool"]
MessageContent = str | list[dict[str, Any]] | None


class FunctionCall(TypedDict):
    """A complete function call; arguments remain a JSON string."""

    name: str
    arguments: str


class ToolCall(TypedDict):
    """A function call in the OpenAI Chat Completions format."""

    id: str
    type: Literal["function"]
    function: FunctionCall


class Message(TypedDict):
    """A provider-independent message in the Chat Completions format.

    The codec validates role-specific fields and encodes the message for the
    selected provider. Optional keys are omitted when they are not needed.
    """

    role: MessageRole
    content: NotRequired[MessageContent]
    name: NotRequired[str]
    tool_call_id: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]
    reasoning_content: NotRequired[str]
    finish_reason: NotRequired[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageChunk:
    """An assistant-message fragment that supports lossless concatenation.

    The codec emits complete tool calls, not partial JSON argument fragments.
    Adding chunks produces a new chunk without changing either input.
    """

    role: ClassVar[Literal["assistant"]] = "assistant"

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None

    def __add__(self, other: MessageChunk) -> MessageChunk:
        if not isinstance(other, MessageChunk):
            return NotImplemented

        if (
            self.finish_reason is not None
            and other.finish_reason is not None
            and self.finish_reason != other.finish_reason
        ):
            raise ValueError("conflicting finish reasons")

        return MessageChunk(
            content=self.content + other.content,
            reasoning_content=self.reasoning_content + other.reasoning_content,
            tool_calls=[*self.tool_calls, *other.tool_calls],
            finish_reason=other.finish_reason or self.finish_reason,
        )

    @property
    def has_delta(self) -> bool:
        return bool(self.content or self.reasoning_content or self.tool_calls)

    @property
    def is_finished(self) -> bool:
        return self.finish_reason is not None

    def to_message(self, *, require_finished: bool = False) -> Message:
        """Build the complete assistant message from accumulated fragments."""
        if require_finished and not self.is_finished:
            raise ValueError("cannot finalize an unfinished message stream")

        message: Message = {"role": "assistant", "content": self.content or None}
        if self.reasoning_content:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = list(self.tool_calls)
        if self.finish_reason is not None:
            message["finish_reason"] = self.finish_reason
        return message
