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
    """Abstract converter from native protocol values to intermediate dictionaries.

    Implement protocol, message_type, tool_type, match_item, usage and the protected
    conversion hooks. Instances must be stateless and declare empty __slots__.
    Public messages/tools/chunks methods accept lists, an optional ResolvedProfile
    and protocol-specific keyword options. None selects standard field behavior.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def protocol(self) -> str:
        """Registered name of the native source protocol."""
        ...

    @property
    @abstractmethod
    def message_type(self) -> Any:
        """Schema type accepted for native message validation."""
        ...

    @property
    @abstractmethod
    def tool_type(self) -> Any:
        """Schema type accepted for native tool declaration validation."""
        ...

    def validate_values(self, values: list, schema: Any, *, filtered: bool = False) -> None:
        """Validate each item against schema, raising MessageValidationError on failure.

        Set filtered=True only when values have already been filtered to the schema.
        """
        validate_values(values, schema, filtered=filtered)

    @final
    def request(self, inputs: dict, profile: ResolvedProfile) -> dict:
        """Return encoded messages/tools alongside a complete call's generation options.

        inputs must include messages and tools. profile supplies upstream settings;
        the returned dictionary also contains profile for BaseDecoder.prepare.
        Invalid input raises MessageValidationError.
        """
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
        """Return whether the nonempty list conforms to this protocol as a whole."""
        return match_all(values, self.match_item)

    @final
    def messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Return independent intermediate message dictionaries from native values."""
        return capture(self._messages(dictionaries(values), profile, **parameters))

    @final
    def tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Return independent intermediate tool declarations from native values."""
        return capture(self._tools(dictionaries(values), profile, **parameters))

    @final
    def chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Return intermediate event dictionaries from the supplied native event list."""
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
        """Read a semantic response field using an optional output Parameter.

        Args:
            value: Native response dictionary or supported SDK model.
            profile: Resolved settings, or None for the default path.
            name: Output Parameter field name.
            default_path: Rooted native path used without an output override.
            default: Value returned when the path is absent or routing is disabled.
            use_default: Whether a declaration's non-omit default replaces default.

        An explicit None at the addressed path is preserved. Input declarations do
        not change response lookup; omitted or unsupported declarations return default.
        """
        if profile is not None:
            declaration = profile.parameters.get(name)
            if isinstance(declaration, Omit) or (
                declaration is not None and not declaration.supports
            ):
                return default
            if declaration is not None and declaration.source == "output":
                default_path = declaration.path
                if use_default and not isinstance(declaration.default_value, Omit):
                    default = declaration.default_value
        return path_value(value, default_path, default)

    @abstractmethod
    def match_item(self, item: dict) -> bool:
        """Return whether one native message or tool dictionary matches this protocol."""
        ...

    @abstractmethod
    def _messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Translate recognized native messages into intermediate message dictionaries."""
        ...

    @abstractmethod
    def _tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Translate native tool declarations into intermediate tool dictionaries."""
        ...

    @abstractmethod
    def _chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Build owned operations, capturing borrowed mutable leaves after projection."""
        ...

    @abstractmethod
    def usage(self, value: dict | None) -> dict | None:
        """Return intermediate token accounting from native usage, or None if absent."""
        ...
