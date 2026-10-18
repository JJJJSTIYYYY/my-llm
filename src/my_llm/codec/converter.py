"""Reusable, profile-independent conversion among five standard protocols."""

from typing import Any

from my_llm.codec.anthropic.codec import AnthropicCodec
from my_llm.codec.base.codec import BaseCodec
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.base.transport import BaseTransport
from my_llm.codec.gemini.codec import GeminiCodec
from my_llm.codec.ollama.codec import OllamaCodec
from my_llm.codec.openai.chat.codec import ChatCompletionsCodec
from my_llm.codec.openai.responses.codec import ResponsesCodec
from my_llm.codec.shared.replay import retain
from my_llm.codec.shared.stream import is_batch
from my_llm.exceptions import UnsupportedProviderError
from my_llm.profile.configuration import Protocol
from my_llm.profile.resolution import ResolvedProfile


class MyCodec:
    """Convert standard input to open intermediate dictionaries and back.

    No profile, format guess or stream fragments are retained between calls.
    ``profile=None`` means the standard protocol without provider overrides.
    """

    __slots__ = ("codecs",)

    def __init__(self) -> None:
        self.codecs: dict[str, BaseCodec] = {
            "openai_chat": ChatCompletionsCodec(),
            "openai_responses": ResponsesCodec(),
            "anthropic": AnthropicCodec(),
            "gemini": GeminiCodec(),
            "ollama": OllamaCodec(),
        }

    def _codec(self, protocol: str) -> BaseCodec:
        try:
            return self.codecs[protocol]
        except (KeyError, TypeError) as exc:
            raise UnsupportedProviderError(f"unsupported protocol: {protocol!r}") from exc

    def match_protocol(self, messages_or_tools: list[Any]) -> str | None:
        """Return the unique protocol matching the whole list, or None."""
        matches = [
            protocol
            for protocol, codec in self.codecs.items()
            if codec.match_protocol(messages_or_tools) is not None
        ]
        return matches[0] if len(matches) == 1 else None

    def decoder(self, protocol: str) -> BaseDecoder:
        """Return the stateless intermediate-to-native converter."""
        return self._codec(protocol).decoder

    def transport(self, protocol: str) -> BaseTransport:
        """Return the SDK execution component composed by this protocol's codec."""
        return self._codec(protocol).transport

    def validate_profile(self, profile: ResolvedProfile) -> None:
        self._codec(profile.protocol).validate_profile(profile)

    def validate_input(self, protocol: str, messages: list, tools: list | None = None) -> None:
        """Validate declared input through its codec's transport."""
        self._codec(protocol).validate_input(messages, tools)

    def upstream(self, client: Any, request: dict, profile: ResolvedProfile, **parameters: Any):
        return self.transport(profile.protocol).upstream(client, request, profile, **parameters)

    def result(self, messages: list, *, target_format: Protocol):
        converted = self.decode(messages=messages, target_format=target_format)["messages"]
        return self.decoder(target_format).result(converted)

    def encode(
        self,
        messages: Any = None,
        tools: Any = None,
        *,
        input_format: Protocol,
        profile: ResolvedProfile | None = None,
        chunks: Any = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        codec = self._codec(input_format)
        result = {}
        for kind, values in (("messages", messages), ("tools", tools), ("chunks", chunks)):
            if values is not None:
                result[kind] = getattr(codec, f"encode_{kind}")(values, profile, **parameters)
        return result

    def decode(
        self,
        messages: Any = None,
        tools: Any = None,
        *,
        target_format: Protocol,
        profile: ResolvedProfile | None = None,
        chunks: Any = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        codec = self._codec(target_format)
        replay_chunks = None
        if chunks and all(is_batch(chunk) for chunk in chunks):
            from my_llm.codec.shared.conversion import source_protocol

            if any(source_protocol(chunk) != target_format for chunk in chunks):
                replay_chunks = chunks
                # A complete batch is a pure per-call conversion. Live streams
                # use MyLLM's request-owned state instead of retaining it here.
                from my_llm.stream._state import StreamState

                state = StreamState(parameters.get("model", profile.model if profile else ""))
                converted_chunks = state.feed(chunks)
                converted_chunks.extend(state.finish())
                chunks = converted_chunks
                parameters = {**state.metadata, **parameters}
        result = {}
        for kind, values in (("messages", messages), ("tools", tools), ("chunks", chunks)):
            if values is not None:
                result[kind] = getattr(codec, f"decode_{kind}")(values, profile, **parameters)
                if kind == "chunks" and replay_chunks is not None:
                    result[kind] = retain(
                        replay_chunks, result[kind], target_format, kind, force=True
                    )
        return result

    def convert(
        self,
        messages: Any = None,
        tools: Any = None,
        *,
        input_format: Protocol,
        target_format: Protocol,
        profile: ResolvedProfile | None = None,
        chunks: Any = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Convert standard input directly to a target protocol."""
        intermediate = self.encode(messages, tools, input_format=input_format, chunks=chunks)
        return self.decode(
            **intermediate, target_format=target_format, profile=profile, **parameters
        )

    def request(self, intermediate: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        return self._codec(profile.protocol).request(
            intermediate["messages"], intermediate.get("tools"), profile, **parameters
        )
