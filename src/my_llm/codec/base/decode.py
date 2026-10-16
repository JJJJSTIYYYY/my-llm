"""Stateless intermediate to standard-protocol conversion contract."""

from abc import ABC, abstractmethod
from typing import Any

from my_llm.profile import ResolvedProfile


class BaseDecoder(ABC):
    """Convert independently on each call; request state belongs to MyLLM."""

    __slots__ = ()

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
