"""Stateless standard-protocol-to-intermediate translation."""

from abc import ABC, abstractmethod
from typing import Any, final

from my_llm.codec.shared.conversion import capture, dictionaries, match_all
from my_llm.codec.shared.options import path_value
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Omit


class BaseEncoder(ABC):
    """Read recognized input fields, then own only the translated values."""

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
        return capture(self._messages(dictionaries(values), profile, **parameters))

    @final
    def tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return capture(self._tools(dictionaries(values), profile, **parameters))

    @final
    def chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return self._chunks(dictionaries(values), profile, **parameters)

    @final
    def parameter_value(
        self,
        value: Any,
        profile: ResolvedProfile | None,
        name: str,
        default_path: str,
        default: Any = None,
        *,
        use_default: bool = True,
    ) -> Any:
        """Read one semantic field, using only its encode declaration.

        Native paths remain the protocol hook's responsibility. Neither display
        names nor decode declarations change how upstream values are read.
        """
        if profile is not None:
            declaration = profile.parameters.get(name)
            if isinstance(declaration, Omit):
                return default
            if declaration is not None and declaration.phase == "encode":
                default_path = declaration.path
                if use_default and not isinstance(declaration.default_value, Omit):
                    default = declaration.default_value
        return path_value(value, default_path, default)

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
    ) -> list[dict]:
        """Build owned operations, capturing borrowed mutable leaves after projection."""
        ...

    @abstractmethod
    def usage(self, value: dict | None) -> dict | None: ...
