"""Stateless standard-protocol to intermediate conversion contract."""

from abc import ABC, abstractmethod
from typing import Any

from my_llm.profile.resolution import ResolvedProfile


class BaseEncoder(ABC):
    """Convert independently on each call; request state belongs to MyLLM."""

    __slots__ = ()

    @abstractmethod
    def matches(self, values: list) -> bool:
        """Recognize this protocol's native message and tool shapes."""
        ...

    @abstractmethod
    def usage(self, value: dict | None) -> dict | None:
        """Normalize native token accounting while retaining native counters."""
        ...

    @abstractmethod
    def messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict[str, Any]]:
        """Convert messages, preserving unknown fields and nested dictionaries."""
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
