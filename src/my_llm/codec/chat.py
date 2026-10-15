"""OpenAI-compatible Chat Completions request and stream codec."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from openai import Omit

from ..exceptions import IncompleteStreamError, ResponseFailedError
from ..profiles import ResolvedProfile
from ..types import MessageChunk
from .common import (
    path_value,
    plain,
    read,
    request_parameters,
    text_delta,
    validate_messages,
    validate_tool_call,
    validate_tools,
)


class ChatCompletionsCodec:
    def __init__(self, profile: ResolvedProfile) -> None:
        self.profile = profile

    def encode(
        self, messages: Any, *, tools: Any, previous_response_id: Any = None, **parameters: Any
    ) -> dict[str, Any]:
        from ..exceptions import ConfigurationError

        if previous_response_id is not None and not isinstance(previous_response_id, Omit):
            raise ConfigurationError("previous_response_id requires Responses API")
        history = []
        for message in validate_messages(messages):
            encoded = {
                key: deepcopy(message[key])
                for key in ("role", "content", "tool_call_id", "tool_calls")
                if key in message
            }
            if self.profile.get("supports_message_name", False) and "name" in message:
                encoded["name"] = message["name"]
            if (
                message.get("tool_calls")
                and self.profile.get("replay_reasoning_content_when_tool_calls", False)
                and "reasoning_content" in message
            ):
                encoded["reasoning_content"] = message["reasoning_content"]
            history.append(encoded)
        encoded_tools = validate_tools(tools, self.profile)
        kwargs = request_parameters(self.profile, **parameters)
        kwargs.update(model=self.profile.model, messages=history, stream=True)
        if not isinstance(encoded_tools, Omit):
            kwargs["tools"] = encoded_tools
        return kwargs

    def decoder(self) -> ChatDecoder:
        return ChatDecoder(self.profile)


class ChatDecoder:
    def __init__(self, profile: ResolvedProfile) -> None:
        self.profile = profile
        self.calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.closed = False

    def feed(self, event: Any) -> list[MessageChunk]:
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
