"""OpenAI request orchestration and per-stream decoder factories."""

from __future__ import annotations

from typing import Any

from openai import Omit

from my_llm.codec.openai.decoding.chat import ChatDecoder
from my_llm.codec.openai.decoding.responses import ResponsesDecoder
from my_llm.codec.openai.encoding.messages import encode_messages
from my_llm.codec.openai.encoding.parameters import request_parameters
from my_llm.codec.openai.encoding.tools import encode_tools
from my_llm.profile.utils.resolution import ResolvedProfile
from my_llm.types.exceptions import MessageValidationError


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
        from my_llm.types.exceptions import ConfigurationError

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


class ResponsesCodec:
    """Advanced adapter for encoding requests and decoding Responses streams.

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

        parameters accepts generation reasoning, token and extra-field options.
        previous_response_id permits tool results whose calls are in server-side
        history. Invalid input raises ConfigurationError subclasses.
        """
        has_previous = previous_response_id is not None and not isinstance(
            previous_response_id, Omit
        )
        if has_previous and (not isinstance(previous_response_id, str) or not previous_response_id):
            raise MessageValidationError("previous_response_id must be a non-empty string")
        history = encode_messages(messages, self.profile, allow_external_calls=has_previous)
        encoded_tools = encode_tools(tools, self.profile)
        kwargs = request_parameters(self.profile, **parameters)
        kwargs.update(model=self.profile.model, input=history, stream=True)
        if not isinstance(encoded_tools, Omit):
            kwargs["tools"] = encoded_tools
        if has_previous:
            kwargs["previous_response_id"] = previous_response_id
        return kwargs

    def decoder(self) -> ResponsesDecoder:
        """Return a fresh stateful decoder for one Responses stream."""
        return ResponsesDecoder(self.profile)
