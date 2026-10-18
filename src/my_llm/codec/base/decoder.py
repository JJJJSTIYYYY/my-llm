"""Stateless, loss-preserving intermediate-to-protocol translation."""

from abc import ABC, abstractmethod
from typing import Any, final

from my_llm.codec.shared.options import PreparedOptions
from my_llm.codec.shared.replay import retain
from my_llm.profile.resolution import ResolvedProfile


class BaseDecoder(ABC):
    """Translate values and request options; Transport validates the result."""

    __slots__ = ()

    @property
    @abstractmethod
    def protocol(self) -> str: ...

    @final
    def request(
        self,
        messages: list,
        tools: list | None,
        profile: ResolvedProfile,
        *,
        options: PreparedOptions,
        **parameters: Any,
    ) -> dict:
        """Translate one request, preserving extensions until the SDK boundary."""
        native_messages = self.messages(messages, profile, **parameters)
        native_tools = self.tools(tools, profile, **parameters) if tools is not None else None
        return self.assemble_request(
            native_messages, native_tools, profile, options=options, **parameters
        )

    @abstractmethod
    def request_parameters(self, profile: ResolvedProfile, options: PreparedOptions) -> dict:
        """Translate validated generation controls to native request fields."""
        ...

    @abstractmethod
    def assemble_request(
        self,
        messages: list,
        tools: list | None,
        profile: ResolvedProfile,
        *,
        options: PreparedOptions,
        **parameters: Any,
    ) -> dict:
        """Place translated messages and options in the native request envelope."""
        ...

    @final
    def messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return retain(
            values, self._messages(values, profile, **parameters), self.protocol, "messages"
        )

    @final
    def tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return retain(values, self._tools(values, profile, **parameters), self.protocol, "tools")

    @final
    def chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return self._chunks(values, profile, **parameters)

    def result(self, messages: list) -> dict | list:
        return messages[0]

    @abstractmethod
    def _messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Translate messages into this protocol's native representation."""
        ...

    @abstractmethod
    def _tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Translate tool definitions."""
        ...

    @abstractmethod
    def _chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Translate events without retaining stream state."""
        ...

    @abstractmethod
    def usage(self, value: dict | None) -> dict | None:
        """Translate token accounting."""
        ...
