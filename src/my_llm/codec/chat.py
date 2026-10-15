"""OpenAI-compatible Chat Completions request and stream codec."""

from __future__ import annotations

from typing import Any

from openai import Omit

from my_llm.codec.common import (
    path_value,
    plain,
    read,
    request_parameters,
    text_delta,
)
from my_llm.codec.messages import encode_messages
from my_llm.codec.tools import encode_tools
from my_llm.codec.validation import validate_tool_call
from my_llm.exceptions import IncompleteStreamError, ResponseFailedError
from my_llm.profiles import ResolvedProfile
from my_llm.types import MessageChunk


class ChatCompletionsCodec:
    """Advanced adapter for encoding requests and decoding Chat Completions.

    Most applications should use MyLLM.generate(). Direct callers provide a
    resolved profile and request-owned message dictionaries; encode() does not
    isolate mutable input. Create a fresh decoder for each response stream.
    """

    def __init__(self, profile: ResolvedProfile) -> None:
        """Use a validated profile returned by resolve_profiles()."""
        self.profile = profile

    def encode(
        self, messages: Any, *, tools: Any, previous_response_id: Any = None, **parameters: Any
    ) -> dict[str, Any]:
        """Encode messages/tools in either input format as SDK request keywords.

        parameters accepts the generation reasoning, token and extra-field
        options. previous_response_id must be None/omit for this API.
        Invalid configuration or input raises ConfigurationError subclasses.
        """
        from my_llm.exceptions import ConfigurationError

        if previous_response_id is not None and not isinstance(previous_response_id, Omit):
            raise ConfigurationError("previous_response_id requires Responses API")
        history = encode_messages(messages, self.profile)
        encoded_tools = encode_tools(tools, self.profile)
        kwargs = request_parameters(self.profile, **parameters)
        kwargs.update(model=self.profile.model, messages=history, stream=True)
        if not isinstance(encoded_tools, Omit):
            kwargs["tools"] = encoded_tools
        return kwargs

    def decoder(self) -> ChatDecoder:
        """Return a fresh stateful decoder for one Chat Completions stream."""
        return ChatDecoder(self.profile)


class ChatDecoder:
    """Incrementally decode one stream; call finish() once after exhaustion."""

    def __init__(self, profile: ResolvedProfile) -> None:
        """Set delta paths and usage settings from the resolved profile."""
        self.profile = profile
        self.calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.closed = False

    def feed(self, event: Any) -> list[MessageChunk]:
        """Read one SDK event/dictionary and return any text/reasoning chunks.

        Tool argument fragments are buffered until finish(). StreamError
        subclasses report invalid events or upstream error payloads.
        """
        error = read(event, "error")
        if error is not None:
            raise ResponseFailedError("Chat Completions error", details=plain(error))
        if (
            read(event, "usage") is not None
            and self.profile.get("usage_mode", "none") == "stream_options"
        ):
            self.usage = plain(read(event, "usage"))
        choices = read(event, "choices", []) or []
        if not choices:
            return []
        choice = choices[0]
        delta = read(choice, "delta")
        content = text_delta(path_value(event, self.profile.values["content_path"]))
        reasoning = text_delta(path_value(event, self.profile.values["reasoning_content_path"]))
        calls = read(delta, "tool_calls", []) or []
        if self.finish_reason is not None and (content or reasoning or calls):
            raise IncompleteStreamError("received message deltas after finish_reason")
        for call in calls:
            index = read(call, "index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise IncompleteStreamError("tool call delta is missing its index")
            current = self.calls.setdefault(
                index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            call_id = read(call, "id")
            if call_id:
                if current["id"] and current["id"] != call_id:
                    raise IncompleteStreamError("conflicting tool call IDs")
                current["id"] = call_id
            if read(call, "type"):
                current["type"] = read(call, "type")
            function = read(call, "function")
            if read(function, "name"):
                current["function"]["name"] = read(function, "name")
            if read(function, "arguments"):
                current["function"]["arguments"] += read(function, "arguments")
        finish = read(choice, "finish_reason")
        if finish is not None:
            if self.finish_reason is not None and finish != self.finish_reason:
                raise IncompleteStreamError("conflicting finish reasons")
            self.finish_reason = finish
        return (
            [MessageChunk(content=content, reasoning_content=reasoning)]
            if content or reasoning
            else []
        )

    def finish(self) -> MessageChunk:
        """Validate completion and return final tools, usage and finish reason.

        Raises IncompleteStreamError for missing completion, invalid tool calls
        or a repeated finish() call.
        """
        if self.closed:
            raise IncompleteStreamError("decoder was already finalized")
        self.closed = True
        if self.finish_reason is None:
            raise IncompleteStreamError("Chat Completions stream ended without finish_reason")
        calls = [
            validate_tool_call(self.calls[index], streamed=True) for index in sorted(self.calls)
        ]
        if calls and self.finish_reason != "tool_calls":
            raise IncompleteStreamError("tool calls did not finish with tool_calls status")
        if len({call["id"] for call in calls}) != len(calls):
            raise IncompleteStreamError("duplicate tool call IDs")
        if self.finish_reason == "tool_calls" and not calls:
            raise IncompleteStreamError("tool_calls status without a complete function call")
        return MessageChunk(tool_calls=calls, finish_reason=self.finish_reason, usage=self.usage)
