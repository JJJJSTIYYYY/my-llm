"""Stateless bidirectional codecs and protocol implementation contracts."""

from my_llm._optional import __getattr__ as __getattr__
from my_llm.codec.base import BaseCodec, BaseDecoder, BaseEncoder, BaseTransport
from my_llm.codec.converter import MyCodec
from my_llm.codec.shared.options import PreparedOptions

__all__ = [
    "BaseCodec",
    "BaseDecoder",
    "BaseEncoder",
    "BaseTransport",
    "MyCodec",
    "PreparedOptions",
    "AnthropicCodec",
    "GeminiCodec",
    "OllamaCodec",
    "ChatCompletionsCodec",
    "ResponsesCodec",
]
