"""Profile-driven asynchronous OpenAI, Anthropic, Gemini and Ollama adapter."""

from openai import Omit, omit

from my_llm._version import __version__
from my_llm.invoke import MessageStream, MyLLM, generate
from my_llm.profile import APIType, ModelProfile, Protocol, ProviderProfile, ReasoningEffort
from my_llm.types import FunctionCall, Message, MessageChunk, MessageInput, ToolCall
from my_llm.types.exceptions import (
    ConfigurationError,
    IncompleteStreamError,
    MessageValidationError,
    MyLLMError,
    ResponseFailedError,
    ResponseIncompleteError,
    StreamError,
    UnsupportedCapabilityError,
    UnsupportedProviderError,
)

__all__ = [
    "APIType",
    "ConfigurationError",
    "FunctionCall",
    "IncompleteStreamError",
    "MyLLM",
    "Message",
    "MessageChunk",
    "MessageInput",
    "MessageStream",
    "MessageValidationError",
    "ModelProfile",
    "MyLLMError",
    "Omit",
    "Protocol",
    "ProviderProfile",
    "ReasoningEffort",
    "ResponseFailedError",
    "ResponseIncompleteError",
    "StreamError",
    "ToolCall",
    "UnsupportedCapabilityError",
    "UnsupportedProviderError",
    "__version__",
    "generate",
    "omit",
]
