"""Profile-driven asynchronous OpenAI Responses/Chat Completions adapter."""

from openai import Omit, omit

from my_llm._version import __version__
from my_llm.exceptions import (
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
from my_llm.invoke import MessageStream, MyLLM, generate
from my_llm.profiles import APIType, ModelProfile, Protocol, ProviderProfile, ReasoningEffort
from my_llm.types import FunctionCall, Message, MessageChunk, MessageInput, ToolCall

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
