"""Stateless standard-protocol-to-intermediate translation."""

from abc import ABC, abstractmethod
from typing import Any, final

from my_llm.codec.shared.conversion import match_all
from my_llm.codec.shared.replay import encode_retained, replay_inputs
from my_llm.exceptions import MessageValidationError
from my_llm.profile.resolution import ResolvedProfile


class BaseEncoder(ABC):
    """Translate values and recover unchanged converted histories."""

    __slots__ = ()

    @property
    @abstractmethod
    def protocol(self) -> str: ...

    @final
    def matches(self, values: list) -> bool:
        try:
            values, _ = replay_inputs(values, self.protocol)
        except MessageValidationError:
            return False
        return match_all(values, self.match_item)

    @final
    def messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return encode_retained(
            self._messages, values, self.protocol, "messages", profile, **parameters
        )

    @final
    def tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return encode_retained(self._tools, values, self.protocol, "tools", profile, **parameters)

    @final
    def chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return encode_retained(self._chunks, values, self.protocol, "chunks", profile, **parameters)

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
