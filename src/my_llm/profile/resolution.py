"""Resolve profile layers and validate generation/client settings."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Literal, TypeAliasType, Union, get_args, get_origin

from my_llm.basic.exceptions import ConfigurationError, UnsupportedProviderError
from my_llm.basic.protocols import codec_for
from my_llm.profile.base import _declarations
from my_llm.profile.configuration import Protocol, ProtocolDefaults, merge_dicts
from my_llm.profile.model_profile import ModelProfile
from my_llm.profile.paths import path_keys
from my_llm.profile.provider_profile import ProviderProfile
from my_llm.profile.types import Attribute, Config, Omit, Parameter


@dataclass(frozen=True)
class ResolvedProfile:
    """Read-only configuration snapshot returned by resolve_profiles.

    Attributes:
        protocol: Selected upstream protocol name.
        model: Selected model ID.
        values: Resolved defaults and behavior settings, keyed by field name.
        parameters: Parameter declarations or omit, keyed by field name.
        request_parameters: Tuple of supported request-input parameter bindings.
        connection: Validated connection settings.

    Treat dictionaries and nested values as read-only. Applications normally pass
    ProviderProfile and ModelProfile to MyLLM rather than constructing this class.
    """

    protocol: Protocol
    model: str
    values: dict[str, Any]
    parameters: dict[str, Parameter[Any] | Omit] = field(default_factory=dict)

    request_parameters: tuple[tuple[str, Parameter[Any]], ...] = ()
    connection: dict[str, Any] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a resolved value by key, using default for absent or omitted values.

        An explicit stored None is returned as None. Values are borrowed; do not
        mutate returned dictionaries or other nested objects.
        """
        value = self.values.get(key, default)
        return default if isinstance(value, Omit) else value

    def parameter(
        self, name: str, source: Literal["input", "output"] | None = None
    ) -> Parameter[Any] | None:
        """Return a supported Parameter binding, optionally restricted by source.

        Args:
            name: Profile field name.
            source: Optional "input" or "output" filter.

        Returns:
            The declaration, or None if absent, omitted, unsupported or source-mismatched.
            A declaration with default_value=omit may still be supported.
        """
        value = self.parameters.get(name)
        if (
            isinstance(value, Parameter)
            and value.supports
            and (source is None or value.source == source)
        ):
            return value
        return None


def resolve_profiles(
    provider: ProviderProfile,
    model: ModelProfile,
    *,
    defaults_for: Callable[[str], ProtocolDefaults] | None = None,
) -> ResolvedProfile:
    """Resolve independent settings in protocol-default, provider, model order.

    Args:
        provider: ProviderProfile with an explicit protocol Config.
        model: ModelProfile containing a nonempty model ID.
        defaults_for: Optional instance-owned protocol-default lookup. By default,
            create the selected registered codec independently.

    Returns:
        An independent ResolvedProfile snapshot. None inherits; omit disables.
        Parameter wrappers replace whole declarations; Config dictionaries merge
        by key. Explicit default_value=None is retained. Attribute metadata does
        not appear in resolved settings, apart from the selected model identity.

    Raises:
        ConfigurationError: Declarations, values or connection settings are invalid.
        UnsupportedProviderError: The upstream protocol is unavailable or cannot
            support generation.

    Direct integrations should also call MyCodec.validate_profile before use to
    check protocol-specific constraints. MyLLM performs both steps automatically.
    """
    if not isinstance(provider, ProviderProfile) or not isinstance(model, ModelProfile):
        raise ConfigurationError("provider and model must be profile instances")
    if (
        not isinstance(model.model, Attribute)
        or not isinstance(model.model.default_value, str)
        or not model.model.default_value.strip()
    ):
        raise ConfigurationError("model must be a non-empty string")
    if provider.protocol is not None and not isinstance(provider.protocol, Config):
        raise ConfigurationError("protocol must use Config[Protocol]")
    protocol = provider.protocol.default_value if provider.protocol is not None else None
    defaults = codec_for(protocol).defaults if defaults_for is None else defaults_for(protocol)
    if not isinstance(defaults, ProtocolDefaults):
        raise ConfigurationError("codec defaults must be a ProtocolDefaults instance")
    declarations: dict[str, tuple[str, str | None, str]] = {}
    values: dict[str, Any] = {}
    parameters: dict[str, Parameter[Any] | Omit] = {}
    for profile in (defaults, provider, model):
        for name, kind, annotation, source, scope in _declarations(type(profile)):
            identity = (kind, source, scope)
            previous = declarations.setdefault(name, identity)
            if previous[0] != kind:
                raise ConfigurationError(f"{name} must use the same profile marker in every layer")
            if kind == "parameter" and previous != identity:
                raise ConfigurationError(
                    f"{name} must keep the same source and scope in every layer"
                )
            value = getattr(profile, name)
            if value is None:
                continue
            if not _matches(value, annotation):
                raise ConfigurationError(f"{name} does not match its declared profile type")
            if kind == "attribute":
                continue
            if isinstance(value, (Parameter, Config)) and (
                value.name != name or isinstance(value, Parameter) and value.source != source
            ):
                raise ConfigurationError(f"{name} identity must be bound by Profile.__post_init__")
            if kind == "parameter":
                parameters[name] = value
                values[name] = value.default_value if isinstance(value, Parameter) else value
            elif isinstance(value, Omit):
                values[name] = value
            elif isinstance(value.default_value, dict):
                values[name] = merge_dicts(values.get(name), value.default_value)
            else:
                values[name] = value.default_value
    # Snapshot payloads once; immutable bindings only change for copied defaults.
    values = deepcopy(values)
    for name, parameter in parameters.items():
        if isinstance(parameter, Parameter) and parameter.default_value is not values[name]:
            parameters[name] = copy(parameter)
            object.__setattr__(parameters[name], "default_value", values[name])
    if values.get("protocol") != protocol:
        raise ConfigurationError("protocol cannot change after selecting codec defaults")
    resolved = ResolvedProfile(protocol, model.model.default_value, values, parameters)
    for key in (
        "supports_stream",
        "supports_function_calling",
        "supports_message_name",
        "replay_reasoning_content_when_tool_calls",
    ):
        value = resolved.get(key, False)
        if value is not None and not isinstance(value, bool):
            raise ConfigurationError(f"{key} must be a bool, None or omit")
    if not resolved.get("supports_stream", False):
        raise UnsupportedProviderError("provider must support upstream streaming")
    usage = resolved.get("usage_mode", "none")
    if usage is not None and (not isinstance(usage, str) or not usage):
        raise ConfigurationError("usage_mode must be a non-empty string, None or omit")
    if resolved.parameter("max_tokens", "input") is not None:
        validate_max_tokens(resolved.get("max_tokens"))
    object.__setattr__(
        resolved,
        "request_parameters",
        tuple(
            (name, parameter)
            for name, parameter in parameters.items()
            if isinstance(parameter, Parameter)
            and parameter.source == "input"
            and parameter.supports
            and declarations[name][2] == "request"
        ),
    )
    connection_bindings = tuple(
        (name, parameter)
        for name, parameter in parameters.items()
        if isinstance(parameter, Parameter)
        and parameter.supports
        and declarations[name][2] == "client"
    )
    resolved.connection.update(_connection_parameters(resolved, connection_bindings))
    return resolved


def _matches(value: Any, annotation: Any) -> bool:
    """Validate wrapper payload types without walking open dictionaries."""
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin in (Parameter, Config):
        if not isinstance(value, origin):
            return False
        payload = value.default_value
        return (
            payload is None
            or isinstance(payload, Omit)
            or _matches(payload, get_args(annotation)[0])
        )
    if origin is Attribute:
        return isinstance(value, Attribute) and _matches(
            value.default_value, get_args(annotation)[0]
        )
    if isinstance(annotation, TypeAliasType):
        return _matches(value, annotation.__value__)
    if origin in (Union, UnionType):
        return any(_matches(value, member) for member in get_args(annotation))
    if origin is Literal:
        return value in get_args(annotation)
    expected = origin or annotation
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    try:
        return isinstance(value, expected)
    except TypeError:
        return False


def validate_max_tokens(value: Any) -> None:
    """Require a positive integer or None; raise ConfigurationError otherwise."""
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ConfigurationError("max_tokens must be a positive integer, None or omit")


def _connection_parameters(profile: ResolvedProfile, bindings: tuple) -> dict[str, Any]:
    """Validate SDK connection fields from the construction-time profile.

    Client-scoped bindings target SDK authentication or extra_headers entries.
    Decoder selects the SDK header keyword. Unset/omitted values are excluded.
    """
    result = {}
    for key in ("base_url", "timeout", "max_retries"):
        value = profile.get(key)
        if value is None:
            continue
        if key == "base_url" and (not isinstance(value, str) or not value.strip()):
            raise ConfigurationError(f"{key} must be a non-empty string")
        if key == "timeout" and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
        ):
            raise ConfigurationError("timeout must be positive")
        if key == "max_retries" and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ConfigurationError("max_retries must be a non-negative integer")
        result[key] = value
    for name, parameter in bindings:
        keys = path_keys(parameter.path)
        if keys != ("api_key",) and not (
            len(keys) == 2 and keys[0] == "extra_headers" and isinstance(keys[1], str)
        ):
            raise ConfigurationError(f"{name} path must be $.api_key or $.extra_headers.<name>")
        value = profile.get(name)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name} must be a non-empty string")
            if keys == ("api_key",):
                result[keys[0]] = value
            else:
                result.setdefault("default_headers", {})[keys[1]] = value
    return result


def client_parameters(
    profile: ResolvedProfile, *, headers_field: str = "default_headers"
) -> dict[str, Any]:
    """Read construction-time connection settings without parsing them again."""
    result = dict(profile.connection)
    headers = result.pop("default_headers", None)
    if headers is not None:
        result[headers_field] = dict(headers)
    return result
