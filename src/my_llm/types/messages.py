"""Provider-independent messages used by the my-llm codec.

The public message format follows OpenAI Chat Completions. Codecs translate
between this format and each provider's request and response protocol.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, NotRequired, TypedDict

MessageRole = Literal["system", "user", "assistant", "tool"]
MessageContent = str | list[dict[str, Any]] | None


class FunctionCall(TypedDict):
    """A function name and complete JSON-object arguments encoded as a string.

    Parse arguments with json.loads() before invoking a caller-selected tool.
    """

    name: str
    arguments: str


class ToolCall(TypedDict):
    """A complete function call with id, type="function" and function details.

    Use id as the tool_call_id of the corresponding tool-result message.
    """

    id: str
    type: Literal["function"]
    function: FunctionCall


class Message(TypedDict):
    """A provider-independent message in the Chat Completions format.

    Required role is system/user/assistant/tool. content is text, content-block
    dictionaries, or None. Optional name identifies a message sender;
    tool_calls contains complete assistant function calls and tool_call_id
    associates a tool result with its call. reasoning_content contains visible
    reasoning. finish_reason marks completion and usage preserves native usage
    fields. Responses results may also include response_id, response_status and
    response_output for conversation continuation and lossless history replay.

    Optional refusal, audio and legacy function_call fields support native Chat
    input where applicable. A TypedDict describes ordinary dictionaries; it does
    not validate them on construction. Pass returned messages unchanged when
    replaying history so native Responses items are preserved.
    """

    role: MessageRole
    content: NotRequired[MessageContent]
    name: NotRequired[str]
    tool_call_id: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]
    reasoning_content: NotRequired[str]
    finish_reason: NotRequired[str]
    usage: NotRequired[dict[str, Any]]
    response_id: NotRequired[str]
    response_output: NotRequired[list[dict[str, Any]]]
    response_status: NotRequired[str]
    refusal: NotRequired[str | None]
    audio: NotRequired[dict[str, Any] | None]
    function_call: NotRequired[dict[str, Any] | None]


type MessageInput = Message | dict[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageChunk:
    """An assistant-message fragment that supports lossless concatenation.

    Args:
        content: Text delta, default "".
        reasoning_content: Visible reasoning delta, default "".
        tool_calls: Complete function calls, emitted in the final chunk; default [].
        finish_reason: Completion reason, or None before completion.
        usage: Native usage dictionary when configured, otherwise None.
        response_id: Native Responses ID, otherwise None.
        response_output: Native Responses output items for replay, default [].
        response_status: Native Responses status, otherwise None.

    role is always "assistant". Adding chunks produces a new chunk without
    changing either input. Nested payloads are shared read-only; to_message()
    creates an independent mutable snapshot for callers that need to edit it.
    """

    role: ClassVar[Literal["assistant"]] = "assistant"

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    response_id: str | None = None
    response_output: list[dict[str, Any]] = field(default_factory=list)
    response_status: str | None = None

    def __add__(self, other: MessageChunk) -> MessageChunk:
        """Concatenate deltas; raise ValueError for conflicting finish reasons.

        Tool lists are concatenated. Later non-empty response metadata and
        non-None usage take precedence. Nested payloads remain read-only.
        """
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
            usage=other.usage if other.usage is not None else self.usage,
            response_id=other.response_id or self.response_id,
            response_output=other.response_output or self.response_output,
            response_status=other.response_status or self.response_status,
        )

    @property
    def has_delta(self) -> bool:
        """Whether this chunk contains text, reasoning or complete tool calls."""
        return bool(self.content or self.reasoning_content or self.tool_calls)

    @property
    def is_finished(self) -> bool:
        """Whether finish_reason is present, including provider-specific values."""
        return self.finish_reason is not None

    def to_message(self, *, require_finished: bool = False) -> Message:
        """Return an independent mutable Message dictionary for this chunk.

        Args:
            require_finished: Raise ValueError when no finish_reason is present;
                default False permits snapshots of an unfinished stream.

        Accumulate all chunks with + before converting a complete stream.
        Empty content becomes None; unused optional keys are omitted.
        """
        if require_finished and not self.is_finished:
            raise ValueError("cannot finalize an unfinished message stream")

        message: Message = {"role": "assistant", "content": self.content or None}
        if self.reasoning_content:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        if self.finish_reason is not None:
            message["finish_reason"] = self.finish_reason
        if self.usage is not None:
            message["usage"] = self.usage
        if self.response_id is not None:
            message["response_id"] = self.response_id
        if self.response_output:
            message["response_output"] = self.response_output
        if self.response_status is not None:
            message["response_status"] = self.response_status
        return (
            deepcopy(message)
            if self.tool_calls or self.usage is not None or self.response_output
            else message
        )
