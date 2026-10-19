"""Stateless standard-protocol-to-intermediate translation."""

from abc import ABC, abstractmethod
from typing import Any, final

from my_llm.codec.shared.conversion import dictionaries, match_all
from my_llm.profile.resolution import ResolvedProfile


class BaseEncoder(ABC):
    """Capture inputs once, then translate only explicitly recognized fields."""

    __slots__ = ()

    @property
    @abstractmethod
    def protocol(self) -> str: ...

    @final
    def matches(self, values: list) -> bool:
        return match_all(values, self.match_item)

    @final
    def messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return self._messages(dictionaries(values), profile, **parameters)

    @final
    def tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return self._tools(dictionaries(values), profile, **parameters)

    @final
    def chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return self._chunks(dictionaries(values), profile, **parameters)

    @abstractmethod
    def match_item(self, item: dict) -> bool: ...

    @abstractmethod
    def _messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]: ...

    @abstractmethod
    def _tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]: ...

    @abstractmethod
    def _chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]: ...

    @abstractmethod
    def usage(self, value: dict | None) -> dict | None: ...
