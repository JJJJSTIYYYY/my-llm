"""Stateless official SDK lifecycle and request execution contract."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Hashable
from contextlib import AbstractAsyncContextManager
from typing import Any, final

from pydantic import ValidationError

from my_llm.codec.shared.options import PreparedOptions, resolve_options, set_path
from my_llm.codec.shared.transport import (
    _MISSING,
    adapter,
    declared_path,
    path_item,
    projected,
    supported,
    validate_history,
)
from my_llm.exceptions import ConfigurationError, MessageValidationError, UnsupportedCapabilityError
from my_llm.profile.paths import path_keys
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Omit


class BaseTransport(ABC):
    """Validate request parameters, filter unsupported fields and execute native requests.

    BaseCodec composes this transport with its encoder and decoder. MyLLM owns
    cached SDK clients; each upstream context owns only its response resources.
    Profiles and clients are call arguments, never retained on this object.
    """

    __slots__ = ()
    usage_modes = ("none", "native")
    supports_previous_response_id = False

    @property
    @abstractmethod
    def request_schema(self) -> dict[str, Any]:
        """The official SDK's root keyword parameters and nested protocol schemas."""
        ...

    @property
    @abstractmethod
    def message_path(self) -> tuple[str, ...]:
        """Path to native request history, declared by the protocol owner."""
        ...

    @property
    @abstractmethod
    def tool_path(self) -> tuple[str, ...]:
        """Path to native tool declarations, declared by the protocol owner."""
        ...

    @property
    @abstractmethod
    def message_type(self) -> Any:
        """Public SDK message schema used to filter protocol structure."""
        ...

    @property
    @abstractmethod
    def tool_type(self) -> Any:
        """Public SDK tool schema; arbitrary parameter JSON stays intact."""
        ...

    def validate_input(self, messages: list, tools: list | None = None) -> None:
        """Check the declared input protocol without filtering the stored values."""
        try:
            self._validate_values(messages, self.message_type)
            self._validate_values(tools or [], self.tool_type)
        except MessageValidationError as exc:
            raise MessageValidationError(
                "message or tool does not use the bound input protocol"
            ) from exc

    def _validate_values(self, values: list, schema: Any, *, filtered: bool = False) -> None:
        try:
            if not isinstance(values, list):
                raise TypeError("protocol values must be a list")
            validator = adapter(schema)
            for value in values:
                parsed = validator.validate_python(value if filtered else projected(value, schema))
                # SDK TypedDicts include lazy Iterable validators. Serializing
                # forces nested validation rather than accepting an unchecked iterator.
                validator.dump_json(parsed, warnings="error")
        except (ValidationError, TypeError, ValueError) as exc:
            raise MessageValidationError(
                "message or tool does not conform to the native protocol"
            ) from exc

    def validate_parameters(
        self,
        profile: ResolvedProfile,
        *,
        previous_response_id: Any = None,
        **parameters: Any,
    ) -> PreparedOptions:
        """Validate and prepare generation settings for the request's Decoder."""
        self.validate_parameter_roots(profile)
        options = resolve_options(profile, **parameters)
        previous = previous_response_id
        if previous is not None and not isinstance(previous, Omit):
            if (
                not self.supports_previous_response_id
                or not isinstance(previous, str)
                or not previous
            ):
                raise ConfigurationError(
                    "previous_response_id requires a supported API and a non-empty ID"
                )
        return options

    @final
    def request(
        self,
        request: dict,
        profile: ResolvedProfile,
        *,
        history: list | None = None,
        **parameters: Any,
    ) -> dict:
        """Filter native payloads and validate their owned request envelopes."""
        if history is not None:
            validate_history(
                history,
                allow_external_calls=self.allows_external_calls(
                    parameters.get("previous_response_id")
                ),
            )
        original = request
        request = self.filter_request(request, profile)
        messages = self._parent(request, self.message_path)
        if messages is None or self.message_path[-1] not in messages:
            raise MessageValidationError("native request requires messages")
        key = self.message_path[-1]
        if not isinstance(messages[key], list):
            raise MessageValidationError("native messages must be a list")
        messages[key] = self.filter_messages(messages[key], profile, **parameters)
        self._validate_values(messages[key], self.message_type, filtered=True)
        tools = self._parent(request, self.tool_path)
        if tools is not None and tools.get(self.tool_path[-1]) is not None:
            key = self.tool_path[-1]
            if not isinstance(tools[key], list):
                raise MessageValidationError("native tools must be a list")
            if tools[key] and not profile.get("supports_function_calling", False):
                raise UnsupportedCapabilityError("model does not support function calling")
            tools[key] = self.filter_tools(tools[key], profile)
            self._validate_values(tools[key], self.tool_type, filtered=True)
        request = self.validate_request(request, profile, **parameters)
        return self.extend_request(request, original, profile)

    def filter_request(self, request: dict, profile: ResolvedProfile) -> dict:
        """Select SDK fields and explicit profile extensions by declared location."""
        # Message/tool hooks own these leaves, including nested SDK locations.
        # Copy their envelopes here and filter the delegated payloads once below.
        delegated = (self.message_path, self.tool_path)
        result = {}
        for name, schema in self.request_schema.items():
            if name not in request or isinstance(request[name], Omit):
                continue
            try:
                result[name] = supported(
                    request[name],
                    schema,
                    delegated_paths=tuple(path[1:] for path in delegated if path[0] == name),
                )
            except UnsupportedCapabilityError as exc:
                raise ConfigurationError(f"invalid {name}: {exc}") from exc
        return result

    def extend_request(self, result: dict, request: dict, profile: ResolvedProfile) -> dict:
        """Restore only explicit profile leaves absent from the SDK declaration."""
        for name, parameter in profile.parameters.items():
            if isinstance(parameter, Omit) or parameter.phase != "decode":
                continue
            path = parameter.path
            keys = path_keys(path)
            if keys[0] not in self.request_schema:
                raise ConfigurationError(
                    f"{name} parameter path points to unsupported SDK parameter {keys[0]!r}"
                )
            schema = self.request_schema[keys[0]]
            if declared_path(schema, tuple(keys[1:])):
                continue
            item = path_item(request, keys)
            if item is not _MISSING and not isinstance(item, Omit):
                set_path(result, path, supported(item, Any))
        return result

    def validate_parameter_roots(self, profile: ResolvedProfile) -> None:
        """Reject declared request routes that the selected SDK cannot receive."""
        for name, parameter in profile.parameters.items():
            if isinstance(parameter, Omit) or parameter.phase != "decode":
                continue
            root = path_keys(parameter.path)[0]
            if root not in self.request_schema:
                raise ConfigurationError(
                    f"{name} parameter path points to unsupported SDK parameter {root!r}"
                )

    def _parent(self, request: dict, path: tuple[str, ...]) -> dict | None:
        for key in path[:-1]:
            child = request.get(key)
            if not isinstance(child, dict):
                return None
            request = request[key]
        return request

    def filter_messages(self, messages: list, profile: ResolvedProfile, **parameters: Any) -> list:
        return [supported(message, self.message_type) for message in messages]

    def filter_tools(self, tools: list, profile: ResolvedProfile) -> list:
        return [supported(tool, self.tool_type) for tool in tools]

    @abstractmethod
    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        """Validate native options while keeping SDK request values as dictionaries.

        Replace additional nested containers when changing them; unmodified
        options can remain borrowed from the input request.
        """
        ...

    def allows_external_calls(self, previous_response_id: Any) -> bool:
        return False

    @abstractmethod
    def create_client(self, profile: ResolvedProfile, **parameters: Any) -> Any:
        """Create an official SDK client for the instance-owned cache."""
        ...

    async def close_client(self, client: Any) -> None:
        """Close a cached SDK client through its public async close interface."""
        await client.close()

    def client_key(self, profile: ResolvedProfile, **parameters: Any) -> Hashable:
        """Identify request options bound to the SDK client, if any."""
        return None

    def validate_profile(self, profile: ResolvedProfile) -> None:
        """Validate SDK connection settings before creating any client."""
        self.validate_parameter_roots(profile)
        if profile.get("usage_mode", "none") not in self.usage_modes:
            raise ConfigurationError("usage_mode is incompatible with the selected API")

    @abstractmethod
    def upstream(
        self, client: Any, request: dict, profile: ResolvedProfile, **parameters: Any
    ) -> AbstractAsyncContextManager[AsyncIterator[Any]]:
        """Execute a native request and close its response, borrowing the client."""
        ...
