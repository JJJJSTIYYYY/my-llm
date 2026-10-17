"""Stateless intermediate to standard-protocol conversion contract."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any, final

from my_llm.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile


class BaseDecoder(ABC):
    """Intermediate values to native output, independently of the input protocol.

    Implement conversion, options, assembly and SDK hooks in the native owner.
    Profiles are call arguments; implementations must not mutate inputs or
    retain stream state. Request conversion follows the fixed request method.
    """

    __slots__ = ()

    @final
    def request(
        self, messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any
    ) -> dict:
        """Convert intermediate history/tools, then assemble a native SDK request.

        This is the single request pipeline for every protocol. Native values
        enter only ``assemble_request``; callers always pass intermediate values.
        ``None`` tools remain absent, while an empty list remains explicit.
        """
        if tools and not profile.get("supports_function_calling", False):
            raise UnsupportedCapabilityError("model does not support function calling")
        native_messages = self.messages(messages, profile, **parameters)
        native_tools = self.tools(tools, profile, **parameters) if tools is not None else None
        return self.assemble_request(native_messages, native_tools, profile, **parameters)

    @abstractmethod
    def assemble_request(
        self, messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any
    ) -> dict:
        """Place already decoded native messages/tools into the SDK request.

        Implement native history placement, option mapping and server history
        validation here. Do not re-encode inputs or consult another protocol.
        """
        ...

    @abstractmethod
    def request_parameters(self, profile: ResolvedProfile, **parameters: Any) -> dict:
        """Map generation options to native fields for ``assemble_request``."""
        ...

    @abstractmethod
    def upstream(
        self, request: dict, profile: ResolvedProfile, **parameters: Any
    ) -> AbstractAsyncContextManager[AsyncIterator[Any]]:
        """Open a request-owned native stream as an async context manager."""
        ...

    @abstractmethod
    def profile_defaults(self) -> dict:
        """Return native defaults without retaining mutable configuration."""
        ...

    def validate_profile(self, profile: ResolvedProfile) -> None:
        """Validate native capability and transport settings."""
        if profile.get("usage_mode", "none") not in self.usage_modes:
            raise ConfigurationError("usage_mode is incompatible with the selected API")

    @property
    @abstractmethod
    def usage_modes(self) -> tuple[str, ...]:
        """Native policies declared by each decoder, interpreted generically."""
        ...

    @abstractmethod
    def usage(self, value: dict | None) -> dict | None:
        """Render normalized token accounting for this protocol."""
        ...

    def result(self, messages: list) -> dict | list:
        """Return the collected value in this protocol's public shape."""
        return messages[0]

    def allows_external_calls(self, previous_response_id: Any) -> bool:
        """Whether native server history can supply missing local calls."""
        return False

    @abstractmethod
    def messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Render messages in the target protocol; unsupported extensions may be omitted."""
        ...

    @abstractmethod
    def tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Convert tool declarations."""
        ...

    @abstractmethod
    def chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Convert stream events without accumulating fragments on this object."""
        ...
