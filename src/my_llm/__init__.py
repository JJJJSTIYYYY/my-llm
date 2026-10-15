"""Profile-driven asynchronous OpenAI Responses/Chat Completions adapter."""

from openai import Omit, omit

from .exceptions import (
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
from .invoke import Invoke, MessageStream, create
from .profiles import APIType, ModelProfile, Protocol, ProviderProfile, ReasoningEffort
from .types import FunctionCall, Message, MessageChunk, ToolCall

__all__ = [
    "APIType",
    "ConfigurationError",
    "FunctionCall",
    "IncompleteStreamError",
    "Invoke",
    "Message",
    "MessageChunk",
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
    "create",
    "omit",
]
