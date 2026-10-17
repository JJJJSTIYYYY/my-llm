"""Stateless standard-protocol to intermediate conversion contract."""

from abc import ABC, abstractmethod
from typing import Any, final

from my_llm.codec.shared.matching import match_all
from my_llm.profile.resolution import ResolvedProfile


class BaseEncoder(ABC):
    """Native input to intermediate values, without knowing the output protocol.

    Implement the five abstract hooks directly in the protocol's encoder.py.
    Keep content helpers in the same direction and mutable state request-owned.
    Profiles are call arguments; implementations must not mutate inputs.
    """

    __slots__ = ()

    @final
    def matches(self, values: list) -> bool:
        """Recognize the whole native list; malformed inputs never match."""
        return match_all(values, self.match_item)

    @abstractmethod
    def match_item(self, item: dict) -> bool:
        """Recognize one native message/tool; list handling is fixed in matches."""
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
