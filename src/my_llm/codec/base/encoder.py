"""Native parsing into the fixed, SDK-independent middleware field set."""

from abc import ABC, abstractmethod
from typing import Any, final

from my_llm.basic.data import dictionaries, dictionary_views
from my_llm.basic.exceptions import MessageValidationError
from my_llm.cache.compiled_cache import validate_values
from my_llm.codec.shared.conversion import match_all
from my_llm.codec.shared.options import path_value
from my_llm.middleware.models import FunctionTool, Message, messages_from_wire, tools_from_wire
from my_llm.middleware.protocol import ProtocolMiddleWare
from my_llm.middleware.stream import StreamBatch, stream_from_wire
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Omit


class BaseEncoder(ABC):
    """Abstract converter from native protocol values to typed middleware models.

    Implement protocol, message_type, tool_type, match_item, usage and the protected
    conversion hooks. Each codec factory owns a separate encoder instance.
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
        the returned envelope contains profile and a typed middleware field for
        BaseDecoder.prepare. Unknown source fields never enter that middleware.
        Invalid input raises MessageValidationError.
        """
        history = list(dictionary_views(inputs["messages"]))
        tools = inputs["tools"]
        tools = None if tools is None or isinstance(tools, Omit) else list(dictionary_views(tools))
        self._validate_input(history, tools)
        return {
            **{key: value for key, value in inputs.items() if key not in ("messages", "tools")},
            "profile": profile,
            "middleware": ProtocolMiddleWare(
                messages=messages_from_wire(
                    self._messages(
                        history,
                        None,
                        enable_message_cache=inputs.get("enable_message_cache", False),
                    )
                ),
                tools=tools_from_wire(self._tools(tools, None)) if tools is not None else None,
            ),
        }

    @final
    def validate_input(self, messages: list, tools: list | None = None) -> None:
        """Validate native input; customize projection and validate_values hooks."""
        self._validate_input(
            dictionaries(messages), dictionaries(tools) if tools is not None else None
        )

    def _validate_input(self, messages: list, tools: list | None) -> None:
        """Validate views already normalized at the public input boundary."""
        try:
            self.validate_values(self._validation_messages(messages), self.message_type)
            if tools is not None:
                self.validate_values(self._validation_tools(tools), self.tool_type)
        except MessageValidationError as exc:
            raise MessageValidationError(
                "message or tool does not use the bound input protocol"
            ) from exc

    def _validation_messages(self, messages: list) -> list:
        return messages

    def _validation_tools(self, tools: list) -> list:
        return tools

    @final
    def matches(self, values: list) -> bool:
        """Return whether the nonempty list conforms to this protocol as a whole."""
        return match_all(values, self.match_item)

    @final
    def messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[Message]:
        """Return supported middleware messages; silently discard unsupported fields."""
        return messages_from_wire(self._messages(dictionaries(values), profile, **parameters))

    @final
    def tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[FunctionTool]:
        """Return SDK-independent function declarations with owned application data."""
        return tools_from_wire(self._tools(dictionaries(values), profile, **parameters))

    @final
    def chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[StreamBatch]:
        """Return middleware operation batches from the supplied native events."""
        return stream_from_wire(self._chunks(dictionaries(values), profile, **parameters))

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
            default: Value returned when the addressed source path is absent.
            use_default: Whether a declaration's non-omit default replaces default.

        An explicit None at the addressed path is preserved. Input declarations do
        not change response lookup. Disabled target bindings do not erase supported
        source content; the decoder reports any resulting semantic loss.
        """
        if profile is not None:
            declaration = profile.parameters.get(name)
            if (
                declaration is not None
                and not isinstance(declaration, Omit)
                and declaration.source == "output"
            ):
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
        """Project recognized native messages before constructing middleware models."""
        ...

    @abstractmethod
    def _tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Project native function declarations before constructing middleware models."""
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
