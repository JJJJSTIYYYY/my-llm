"""Stateless intermediate to standard-protocol conversion contract."""

from abc import ABC, abstractmethod
from typing import Any

from my_llm.exceptions import ConfigurationError
from my_llm.profile.resolution import ResolvedProfile


class BaseDecoder(ABC):
    """Convert independently on each call; request state belongs to MyLLM."""

    __slots__ = ()

    @abstractmethod
    def request(
        self, messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any
    ) -> dict:
        """Build a native SDK request after converting history and tools."""
        raise NotImplementedError

    @abstractmethod
    def upstream(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> Any:
        """Open a request-owned native stream as an async context manager."""
        raise NotImplementedError

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
