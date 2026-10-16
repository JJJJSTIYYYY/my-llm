"""Errors from configuration, protocol conversion and streamed responses.

Official SDK transport and HTTP errors retain their original types.
"""

from typing import Any


class MyLLMError(Exception):
    """Base error raised by my-llm-py."""


class ConfigurationError(MyLLMError, ValueError):
    """A profile or request parameter is invalid."""


class UnsupportedProviderError(ConfigurationError):
    """The selected protocol or provider cannot be invoked."""


class UnsupportedCapabilityError(ConfigurationError):
    """A model does not support the requested capability."""


class MessageValidationError(ConfigurationError):
    """A message/tool is invalid or cannot be converted to the selected API."""


class StreamError(MyLLMError):
    """A response failed in streaming or collected-message mode.

    Attributes:
        details: Native error/status payload when available, otherwise None.
        partial_message: Standard message/items containing output received before the
            failure, when available. It is not a completed successful result.
    """

    def __init__(self, message: str, *, details: Any = None, partial_message: Any = None) -> None:
        """Set the error message and optional native details/partial output."""
        super().__init__(message)
        self.details = details
        self.partial_message = partial_message


class IncompleteStreamError(StreamError):
    """The stream ended without a complete, valid result."""


class ResponseFailedError(StreamError):
    """A Responses failure/error event was received."""


class ResponseIncompleteError(IncompleteStreamError):
    """The Responses API reported an incomplete response."""
