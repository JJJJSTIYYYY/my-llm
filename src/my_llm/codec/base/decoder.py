"""Stateless translation of recognized intermediate values and request options."""

from abc import ABC, abstractmethod
from typing import Any, final

from my_llm.codec.shared.conversion import capture
from my_llm.codec.shared.options import (
    PreparedOptions,
    request_parameters,
    resolve_options,
    set_path,
)
from my_llm.codec.shared.transport import (
    _MISSING,
    _plan,
    declared_path,
    path_item,
    supported,
    validate_history,
    validate_values,
)
from my_llm.exceptions import ConfigurationError, MessageValidationError, UnsupportedCapabilityError
from my_llm.profile.paths import path_keys
from my_llm.profile.resolution import ResolvedProfile, client_parameters
from my_llm.profile.types import Omit


class BaseDecoder(ABC):
    """Prepare and validate final SDK dictionaries; never construct SDK clients."""

    __slots__ = ()
    body_container: str | None = None
    headers_field: str | None = "$.extra_headers"
    body_overlay: str | None = None
    omit_empty_body = False

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

    def validate_parameters(
        self,
        profile: ResolvedProfile,
        *,
        previous_response_id: Any = None,
        **parameters: Any,
    ) -> PreparedOptions:
        """Validate and prepare generation settings for the request's Decoder."""
        if not isinstance(parameters.get("stream", False), bool):
            raise ConfigurationError("stream must be a bool")
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
    def native_request(
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
        self.validate_values(messages[key], self.message_type, filtered=True)
        tools = self._parent(request, self.tool_path)
        if tools is not None and tools.get(self.tool_path[-1]) is not None:
            key = self.tool_path[-1]
            if not isinstance(tools[key], list):
                raise MessageValidationError("native tools must be a list")
            if tools[key] and not profile.get("supports_function_calling", False):
                raise UnsupportedCapabilityError("model does not support function calling")
            tools[key] = self.filter_tools(tools[key], profile)
            self.validate_values(tools[key], self.tool_type, filtered=True)
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
        for name, parameter in profile.request_parameters:
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
        for name, parameter in profile.request_parameters:
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
        apply = _plan(self.message_type).apply
        return [apply(message) for message in messages]

    def filter_tools(self, tools: list, profile: ResolvedProfile) -> list:
        apply = _plan(self.tool_type).apply
        return [apply(tool) for tool in tools]

    def allows_external_calls(self, previous_response_id: Any) -> bool:
        return False

    def validate_profile(self, profile: ResolvedProfile) -> None:
        """Validate SDK connection settings before creating any client."""
        self.validate_parameter_roots(profile)
        if profile.get("usage_mode", "none") not in self.usage_modes:
            raise ConfigurationError("usage_mode is incompatible with the selected API")

    def validate_values(self, values: list, schema: Any, *, filtered: bool = False) -> None:
        validate_values(values, schema, filtered=filtered)

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        return request

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
        options: PreparedOptions | None = None,
        **parameters: Any,
    ) -> dict:
        """Translate once, then capture/validate each native leaf once."""
        if options is None:
            options = self.validate_parameters(profile, **parameters)
        # These internal translations borrow intermediate application leaves.
        # native_request establishes the independently mutable SDK ownership.
        native_messages = self._messages(messages, profile, **parameters)
        native_tools = self._tools(tools, profile, **parameters) if tools is not None else None
        request = self.assemble_request(
            native_messages,
            native_tools,
            profile,
            request=self.request_parameters(profile, options),
            **parameters,
        )
        return self.native_request(request, profile, history=messages, **parameters)

    @final
    def prepare(self, intermediate: dict, client: dict) -> dict:
        """Return the three explicit destinations of a complete generate call."""
        profile = intermediate["profile"]
        parameters = {
            key: value
            for key, value in intermediate.items()
            if key not in ("messages", "tools", "profile")
        }
        options = self.validate_parameters(profile, **parameters)
        client = self.request_client(client, options)
        return {
            "client": client,
            "request": self.request(
                intermediate["messages"],
                intermediate["tools"],
                profile,
                options=options,
                **parameters,
            ),
            "local": {"stream": parameters["stream"], "client_key": self.client_key(client)},
        }

    def client_parameters(self, profile: ResolvedProfile) -> dict:
        """Prepare connection arguments once per MyLLM, independently of calls."""
        return client_parameters(profile)

    def request_client(self, client: dict, options: PreparedOptions) -> dict:
        """Return read-only instance settings, or a request-local SDK variant."""
        return client

    def client_key(self, client: dict) -> Any:
        return None

    @final
    def request_parameters(self, profile: ResolvedProfile, options: PreparedOptions) -> dict:
        """Apply decode declarations at SDK-root paths for every protocol."""
        request = request_parameters(
            profile,
            options,
            body_container=self.body_container,
            headers_field=self.headers_field,
            overlay=self.body_overlay,
            omit_empty_body=self.omit_empty_body,
        )
        return self._request_parameters(request, profile, options)

    def _request_parameters(
        self, request: dict, profile: ResolvedProfile, options: PreparedOptions
    ) -> dict:
        """Add protocol-specific controls after the common parameter pipeline."""
        return request

    @abstractmethod
    def assemble_request(
        self,
        messages: list,
        tools: list | None,
        profile: ResolvedProfile,
        *,
        request: dict,
        **parameters: Any,
    ) -> dict:
        """Place translated messages in the already prepared request envelope."""
        ...

    @final
    def messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return capture(self._messages(values, profile, **parameters))

    @final
    def tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return capture(self._tools(values, profile, **parameters))

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
        """Translate recognized message fields without retaining source envelopes."""
        ...

    @abstractmethod
    def _tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Translate recognized tool definitions and their open parameter schemas."""
        ...

    @abstractmethod
    def _chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Build independent events, capturing only borrowed recognized payloads."""
        ...

    @abstractmethod
    def usage(self, value: dict | None) -> dict | None:
        """Translate token accounting."""
        ...
