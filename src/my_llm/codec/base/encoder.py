"""Stateless standard-protocol-to-intermediate translation."""

from abc import ABC, abstractmethod
from typing import Any, final

from my_llm.codec.shared.conversion import capture, dictionaries, dictionary_views, match_all
from my_llm.codec.shared.options import path_value
from my_llm.codec.shared.transport import validate_values
from my_llm.exceptions import MessageValidationError
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Omit


class BaseEncoder(ABC):
    """Read recognized input fields, then own only the translated values."""

    __slots__ = ()

    @property
    @abstractmethod
    def protocol(self) -> str: ...

    @property
    @abstractmethod
    def message_type(self) -> Any: ...

    @property
    @abstractmethod
    def tool_type(self) -> Any: ...

    def validate_values(self, values: list, schema: Any, *, filtered: bool = False) -> None:
        validate_values(values, schema, filtered=filtered)

    @final
    def request(self, inputs: dict, profile: ResolvedProfile) -> dict:
        """Encode the complete call, borrowing controls until Decoder captures them."""
        history = list(dictionary_views(inputs["messages"]))
        tools = inputs["tools"]
        tools = None if tools is None or isinstance(tools, Omit) else list(dictionary_views(tools))
        self.validate_input(history, tools)
        return {
            **inputs,
            "profile": profile,
            "messages": self.messages(history),
            "tools": self.tools(tools) if tools is not None else None,
        }

    def validate_input(self, messages: list, tools: list | None = None) -> None:
        """Validate the bound input schema before projecting recognized values."""
        try:
            self.validate_values(messages, self.message_type)
            self.validate_values(tools or [], self.tool_type)
        except MessageValidationError as exc:
            raise MessageValidationError(
                "message or tool does not use the bound input protocol"
            ) from exc

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
