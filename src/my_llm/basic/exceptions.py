"""Errors from configuration, protocol conversion and streamed responses.

Official SDK transport and HTTP errors retain their original types.
"""

from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from my_llm.basic.semantic_loss import SemanticLossContext, SemanticLossGroup


class MyLLMError(Exception):
    """Base error raised by my-llm."""


class ConfigurationError(MyLLMError, ValueError):
    """A profile or request parameter is invalid."""


class UnsupportedProviderError(ConfigurationError):
    """The selected protocol or provider cannot be invoked."""


class MessageValidationError(ConfigurationError):
    """A message/tool is invalid or cannot be converted to the selected API."""


class SemanticLossError(MyLLMError):
    """Supported values cannot be expressed; draft is complete and mutable."""

    def __init__(
        self,
        draft: object,
        losses: "SemanticLossGroup",
        context: "SemanticLossContext | None" = None,
    ) -> None:
        super().__init__("; ".join(f"{item.origin_path}: {item.reason}" for item in losses.items))
        self.draft = draft
        self.losses = losses
        self.context = context


class StreamError(MyLLMError):
    """A response failed in streaming or collected-message mode.

    Attributes:
        details: Native error/status payload when available, otherwise None.
        partial_message: Standard message/items containing output received before the
            failure, when available. It is not a completed successful result.
    """

    def __init__(self, message: str, *, details: Any = None, partial_message: Any = None) -> None:
        """Create a response error with optional native details and partial output.

        Args:
            message: Human-readable description.
            details: Native error/status payload, or None.
            partial_message: Incomplete output in the input/output protocol, or None.
                It must not be treated as a successful result.
        """
        super().__init__(message)
        self.details = details
        self.partial_message = partial_message


class IncompleteStreamError(StreamError):
    """The stream ended without a complete, valid result."""


class ResponseFailedError(StreamError):
    """A Responses failure/error event was received."""


class ResponseIncompleteError(IncompleteStreamError):
    """The Responses API reported an incomplete response."""


class JSONGenerationError(MyLLMError, ValueError):
    """JSON generation exhausted its budget; failures contains every failed attempt.

    Each record contains a one-based attempt, stage and errors list. Schema errors
    include instance/schema JSON Pointers, validator and message. Combined-schema
    branch errors are included individually. Records contain diagnostic messages,
    which may include model-generated values; callers own logging and redaction.
    """

    def __init__(self, failures: list[dict[str, Any]]) -> None:
        self.failures = deepcopy(failures)
        super().__init__(
            f"JSON generation failed after {len(failures)} attempt(s): "
            + "; ".join(
                f"attempt {failure['attempt']} [{failure['stage']}] "
                f"{issue.get('instance_path') or '<root>'}: {issue['message']}"
                for failure in failures
                for issue in failure["errors"]
            )
        )
