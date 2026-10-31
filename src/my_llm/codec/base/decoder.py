"""Target rendering with complete drafts and collected semantic losses."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from functools import wraps
from typing import Any, final

from my_llm.basic.data import capture, dictionary_views
from my_llm.basic.exceptions import ConfigurationError, MessageValidationError
from my_llm.basic.semantic_loss import (
    SemanticLossContext,
    SemanticLossItem,
    collect_losses,
    report_loss,
)
from my_llm.cache.compiled_cache import (
    _MISSING,
    _plan,
    declared_path,
    path_item,
    supported,
    validate_values,
)
from my_llm.codec.shared.capabilities import content_capability, supports_content
from my_llm.codec.shared.conversion import validate_history
from my_llm.codec.shared.input import InputPolicy
from my_llm.codec.shared.options import (
    PreparedOptions,
    delete_path,
    path_value,
    request_parameters,
    resolve_options,
    set_path,
)
from my_llm.codec.shared.output import output_chunks, output_messages
from my_llm.middleware.history import HistoryNormalizer
from my_llm.middleware.protocol import wire_values
from my_llm.middleware.validation import (
    validate_events,
    validate_messages,
    validate_tools,
)
from my_llm.profile.paths import path_keys
from my_llm.profile.resolution import ResolvedProfile, client_parameters
from my_llm.profile.types import Omit, Parameter


def _collect_translation(method):
    """Collect nested decoder losses and raise only after a complete draft is available."""

    @wraps(method)
    def call(self, *args, **kwargs):
        """Join the active loss scope or complete a new outer decoder scope."""
        context = SemanticLossContext(None, self.protocol)
        with collect_losses(context) as (losses, owner):
            draft = method(self, *args, **kwargs)
            return losses.complete(draft) if owner else draft

    return call


class BaseDecoder(ABC):
    """Abstract target-protocol converter and native request builder.

    Implement the abstract schema properties, conversion hooks, usage and
    assemble_request. Each codec factory owns a separate decoder instance.
    Public messages/tools/chunks methods accept typed middleware values, optional
    profile overrides and protocol options, and return native dictionaries.
    Unsupported source fields have already been discarded. Supported values that
    cannot be represented are collected until the complete draft exists, then
    raised together as SemanticLossError. Request preparation performs no I/O.
    """

    __slots__ = ()
    body_container: str | None = None
    headers_field: str | None = "$.extra_headers"
    body_overlay: str | None = None
    omit_empty_body = False

    usage_modes = ("none", "native")
    supports_previous_response_id = False
    message_delegated_paths: tuple[tuple[str, ...], ...] = ()

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
        """Return PreparedOptions after validating profile and generation controls.

        parameters accepts generation options such as stream, max_tokens, reasoning,
        reasoning_effort, extra_headers and extra_body. previous_response_id must be a
        nonempty ID. Invalid options raise ConfigurationError;
        unsupported controls are retained as losses for the completed request.
        """
        if not isinstance(parameters.get("stream", False), bool):
            raise ConfigurationError("stream must be a bool")
        options = resolve_options(profile, **parameters)
        previous = previous_response_id
        if previous is not None and not isinstance(previous, Omit):
            if not isinstance(previous, str) or not previous:
                raise ConfigurationError("previous_response_id requires a non-empty ID")
            if not self.supports_previous_response_id:
                options = replace(
                    options,
                    losses=(
                        *options.losses,
                        SemanticLossItem(
                            previous,
                            "$.previous_response_id",
                            "protocol does not support previous_response_id",
                        ),
                    ),
                )
        return options

    @final
    @_collect_translation
    def native_request(
        self,
        request: dict,
        profile: ResolvedProfile,
        *,
        history: list | None = None,
        **parameters: Any,
    ) -> dict:
        """Filter and validate the final native SDK request without performing I/O.

        Optional history is canonical input for association validation. The regular
        request path has already checked it before translation and omits it here.
        Filter the request envelope, messages and tools to their declared SDK
        schemas, preserving independent ownership of mutable application data.
        Then validate native values and protocol-specific request semantics.

        Unknown target values may record semantic losses during filtering.
        Malformed message/tool structure raises MessageValidationError; invalid
        request controls raise ConfigurationError. Every request follows the same
        filtering path, including messages changed by a translation extension or
        message_transform. A translation flag never bypasses capability checks.
        """
        if history is not None:
            validate_history(
                history,
                allow_external_calls=self.allows_external_calls(
                    parameters.get("previous_response_id")
                ),
            )
        original = request
        request = self.filter_request(request, profile)
        usage = profile.get("usage_mode", "none")
        if usage not in self.usage_modes:
            report_loss(usage, "protocol does not support this usage_mode", path="$.usage_mode")
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
            path = "$" + "".join(f".{key}" for key in self.tool_path)
            tools[key] = self._supported_tools(tools[key], profile, path=path)
            tools[key] = self.filter_tools(tools[key], profile)
            self.validate_values(tools[key], self.tool_type, filtered=True)
        self.filter_unsupported_parameters(request, original, profile)
        request = self.validate_request(request, profile, **parameters)
        return self.extend_request(request, original, profile)

    def filter_unsupported_parameters(
        self, request: dict, original: dict, profile: ResolvedProfile
    ) -> None:
        """Remove explicitly unsupported native input bindings and record their values."""
        for name, path, value in self._unsupported_parameters(original, profile):
            report_loss(value, f"model does not support {name}", path=path)
            delete_path(request, path)

    def _unsupported_parameters(self, request: dict, profile: ResolvedProfile):
        """Yield supplied unsupported bindings, honoring supported enable defaults."""
        for name, binding in profile.parameters.items():
            if not isinstance(binding, Parameter) or binding.source != "input" or binding.supports:
                continue
            # Enablement can own a compound value containing the effort leaf.
            # Keep its supported defaults; unsupported public effort controls
            # are reported separately in PreparedOptions.
            enable = profile.parameter("reasoning_enable", "input")
            enabled_value = _MISSING
            if (
                name == "reasoning_effort"
                and enable is not None
                and path_keys(binding.path)[: len(path_keys(enable.path))] == path_keys(enable.path)
            ):
                enabled_value = path_item(
                    enable.default_value, path_keys(binding.path)[len(path_keys(enable.path)) :]
                )
            paths = [binding.path]
            if self.body_overlay is not None and path_keys(binding.path)[0] != self.body_overlay:
                paths.append(f"$.{self.body_overlay}{binding.path[1:]}")
            for path in paths:
                value = path_value(request, path, _MISSING)
                if value is _MISSING or isinstance(value, Omit):
                    continue
                if enabled_value is not _MISSING and value == enabled_value:
                    continue
                yield name, path, value

    def filter_request(self, request: dict, profile: ResolvedProfile) -> dict:
        """Select SDK fields and explicit profile extensions by declared location."""
        # Message/tool hooks own these leaves, including nested SDK locations.
        # Copy their envelopes here and filter the delegated payloads once below.
        delegated = (self.message_path, self.tool_path)
        result = {}
        for name, schema in self.request_schema.items():
            if name not in request or isinstance(request[name], Omit):
                continue
            value = _plan(
                schema,
                delegated_paths=tuple(path[1:] for path in delegated if path[0] == name),
            ).decode(request[name], f"$.{name}")
            if value is not _MISSING:
                result[name] = value
        return result

    def extend_request(self, result: dict, request: dict, profile: ResolvedProfile) -> dict:
        """Restore only explicit profile leaves absent from the SDK declaration."""
        unsupported = None
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
            if unsupported is None:
                unsupported = tuple(
                    path_keys(path) for _, path, _ in self._unsupported_parameters(request, profile)
                )
            if any(keys[: len(parent)] == parent for parent in unsupported):
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
        """Find the dictionary containing a request path's final key, if present."""
        for key in path[:-1]:
            child = request.get(key)
            if not isinstance(child, dict):
                return None
            request = request[key]
        return request

    def filter_messages(self, messages: list, profile: ResolvedProfile, **parameters: Any) -> list:
        """Filter native messages through one shared per-message pipeline.

        Protocol hooks supply any grouping context and adapt one message at a time.
        Capability filtering and SDK field projection run in the same outer loop
        for all protocols, including translated and cached history. Returned values
        own mutable payloads; native_request validates their final SDK shapes after
        this hook returns, including changes made by public hook overrides.
        Model capabilities and reasoning-feedback configuration are resolved once
        per filtering pass; this does not cache or trust message validity.
        """
        path = "$" + "".join(f".{key}" for key in self.message_path)
        decode = _plan(self.message_type, self.message_delegated_paths).decode
        policy = InputPolicy(profile)
        filter_message = self._filter_message
        result = []
        for index, message, feedback in self._message_inputs(messages, policy):
            value = filter_message(
                message,
                policy,
                path=f"{path}[{index}]",
                decode=decode,
                feedback=feedback,
                **parameters,
            )
            if value is not None:
                result.append(value)
        return result

    def _message_inputs(
        self, messages: list, policy: InputPolicy
    ) -> Iterator[tuple[int, dict, bool]]:
        """Yield native views and optional group-level reasoning feedback context."""
        for index, message in enumerate(dictionary_views(messages)):
            yield index, message, False

    def _filter_message(
        self,
        message: dict,
        policy: InputPolicy,
        *,
        path: str,
        decode: Callable,
        feedback: bool,
        **parameters: Any,
    ) -> dict | None:
        """Project one native message; protocols override their own content adaptation."""
        value = decode(message, path)
        return None if value is _MISSING else value

    def filter_tools(self, tools: list, profile: ResolvedProfile) -> list:
        """Return independent tool declarations restricted to the target tool schema."""
        path = "$" + "".join(f".{key}" for key in self.tool_path)
        return _plan(list[self.tool_type]).decode(tools, path)

    def _supported_tools(
        self, tools: list, profile: ResolvedProfile | None, *, path: str = "$.tools"
    ) -> list:
        """Apply the same definition capability at every decoder entry point."""
        if not tools or supports_content("tool_definition", profile):
            return tools
        for index, tool in enumerate(tools):
            content_capability(
                tool,
                "tool_definition",
                profile,
                path=getattr(tool, "origin_path", f"{path}[{index}]"),
            )
        return []

    def allows_external_calls(self, previous_response_id: Any) -> bool:
        """Return whether server history may satisfy missing tool-call references.

        The default is False; supported protocols can use previous_response_id.
        """
        return False

    def validate_profile(self, profile: ResolvedProfile) -> None:
        """Validate SDK connection settings before creating any client."""
        self.validate_parameter_roots(profile)

    def validate_values(self, values: list, schema: Any, *, filtered: bool = False) -> None:
        """Validate each item against schema, raising MessageValidationError on failure.

        Set filtered=True only for values already filtered to the declared schema.
        """
        validate_values(values, schema, filtered=filtered)

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        """Return request after protocol-specific semantic validation.

        The default accepts request unchanged. Overrides reject malformed values
        with ConfigurationError and collect unsupported values as semantic losses.
        """
        return request

    @property
    @abstractmethod
    def protocol(self) -> str:
        """Registered name of the native target protocol."""
        ...

    @final
    @_collect_translation
    def request(
        self,
        messages: list,
        tools: list | None,
        profile: ResolvedProfile,
        *,
        options: PreparedOptions | None = None,
        message_transform: Callable[[list], list] | None = None,
        **parameters: Any,
    ) -> dict:
        """Return validated native SDK call arguments for intermediate messages/tools.

        Args:
            messages: Intermediate message history.
            tools: Intermediate tool declarations, or None to omit tools.
            profile: Resolved upstream provider/model settings.
            options: Already validated controls, or None to validate parameters here.
            message_transform: Optional transform of translated messages before assembly.
            **parameters: Generation controls and protocol-specific options.

        Invalid controls or history raise ConfigurationError or MessageValidationError.
        Supported content that cannot be represented raises SemanticLossError with
        the complete request draft. This method performs no I/O.

        Validate middleware structure and repair tool associations in one visit
        before translation. Deduplicate call IDs, merge result parts, discard
        whole messages with orphan results, and supply empty missing results.
        A protocol supporting server history may retain external result IDs.

        If message_transform is present, apply it to translated native messages
        before assembly. The message cache uses this hook to append the current
        batch to prior translated history; only this batch is repaired by the
        combined structure/association scan. Role failures are reported after
        tool/control checks so their existing error precedence is preserved.
        Finally native_request filters and validates the assembled
        SDK arguments. Cache publication and semantic-loss policy belong to BaseCodec.
        """
        messages = wire_values(messages)
        tools = wire_values(tools) if tools is not None else None
        history = HistoryNormalizer(
            allow_external_calls=self.allows_external_calls(parameters.get("previous_response_id"))
        )
        validate_messages(messages, history=history)
        if tools is not None:
            validate_tools(tools)
            tools = self._supported_tools(tools, profile)
        if options is None:
            options = self.validate_parameters(profile, **parameters)
        for loss in options.losses:
            report_loss(loss.origin_value, loss.reason, path=loss.origin_path)
        messages = history.finish(has_cached_history=parameters.get("has_cached_history", False))
        # These internal translations borrow intermediate application leaves.
        # native_request establishes the independently mutable SDK ownership.
        native_messages = self._messages(messages, profile, **parameters)
        if message_transform is not None:
            native_messages = message_transform(native_messages)
        native_tools = self._tools(tools, profile, **parameters) if tools is not None else None
        request = self.assemble_request(
            native_messages,
            native_tools,
            profile,
            request=self.request_parameters(profile, options),
            **parameters,
        )
        if options.effort is None and options.losses:
            parameters = {**parameters, "reasoning": None, "reasoning_effort": "none"}
        return self.native_request(request, profile, **parameters)

    @final
    @_collect_translation
    def prepare(self, intermediate: dict, client: dict) -> dict:
        """Return client, request and local dictionaries for a complete encoded call.

        intermediate is the dictionary from BaseEncoder.request; client contains
        prepared connection arguments. local contains stream and a client_key.
        This method creates no SDK clients and sends no network request.

        Validate generation controls once, derive request-specific connection
        settings, and pass the typed middleware plus any translated-history
        transform into request(). Return separate client, request and local
        dictionaries so transport execution needs no profile or raw user options.
        """
        profile = intermediate["profile"]
        parameters = {
            key: value
            for key, value in intermediate.items()
            if key not in ("profile", "middleware", "message_transform")
        }
        options = self.validate_parameters(profile, **parameters)
        client = self.request_client(client, options)
        return {
            "client": client,
            "request": self.request(
                intermediate["middleware"].messages,
                intermediate["middleware"].tools,
                profile,
                options=options,
                message_transform=intermediate.get("message_transform"),
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
        """Return a hashable reuse key for client settings; the default is None.

        Override when distinct connection settings require distinct clients.
        """
        return None

    @final
    def request_parameters(self, profile: ResolvedProfile, options: PreparedOptions) -> dict:
        """Apply input declarations at SDK-root paths for every protocol."""
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
    @_collect_translation
    def messages(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Return independent native message dictionaries from intermediate values."""
        values = wire_values(values)
        validate_messages(values)
        values = output_messages(values, parameters.pop("semantic_profile", None))
        return capture(self._messages(values, profile, **parameters))

    @final
    @_collect_translation
    def tools(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Return independent native tool declarations from intermediate values."""
        values = wire_values(values)
        validate_tools(values)
        values = self._supported_tools(values, profile)
        return capture(self._tools(values, profile, **parameters))

    @final
    @_collect_translation
    def chunks(
        self, values: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Return native event dictionaries from the supplied intermediate events."""
        values = wire_values(values)
        validate_events(values)
        values = output_chunks(values, parameters.pop("semantic_profile", None))
        return self._chunks(values, profile, **parameters)

    def result(self, messages: list) -> dict | list:
        """Return the first native message as a completed result.

        Protocols with list-shaped results override this method. messages must contain
        at least one item for this default implementation.
        """
        return messages[0]

    @abstractmethod
    def _messages(
        self, values: Iterable[dict], profile: ResolvedProfile | None = None, **parameters: Any
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
