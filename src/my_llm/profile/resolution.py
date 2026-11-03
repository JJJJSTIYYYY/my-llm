"""Resolve profile layers and validate generation/client settings."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from types import UnionType
from typing import Any, Literal, TypeAliasType, Union, get_args, get_origin

from my_llm.basic.exceptions import ConfigurationError, UnsupportedProviderError
from my_llm.basic.protocols import codec_for
from my_llm.profile.base import Profile, _declarations
from my_llm.profile.configuration import Protocol, ProtocolDefaults, merge_dicts
from my_llm.profile.model_profile import ModelProfile
from my_llm.profile.paths import path_keys
from my_llm.profile.provider_profile import ProviderProfile
from my_llm.profile.types import Attribute, Config, Omit, Parameter


@dataclass
class ResolvedProfile:
    """Configuration snapshot returned by resolve_profiles.

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
        provider: ProviderProfile containing connection settings and defaults.
        model: ModelProfile containing a nonempty model ID and optional overrides.
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
    The model's protocol overrides the provider's before selecting codec defaults.
    """
    if not isinstance(provider, ProviderProfile) or not isinstance(model, ModelProfile):
        raise ConfigurationError("provider and model must be profile instances")
    if (
        not isinstance(model.model, Attribute)
        or not isinstance(model.model.default_value, str)
        or not model.model.default_value.strip()
    ):
        raise ConfigurationError("model must be a non-empty string")
    protocol = None
    for profile in (provider, model):
        if profile.protocol is not None:
            if not isinstance(profile.protocol, Config):
                raise ConfigurationError("protocol must use Config[Protocol]")
            protocol = profile.protocol.default_value
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
    # Snapshot payloads once and isolate mutable parameter bindings.
    values = deepcopy(values)
    for name, parameter in parameters.items():
        if isinstance(parameter, Parameter):
            parameters[name] = copy(parameter)
            object.__setattr__(parameters[name], "default_value", values[name])
    if values.get("protocol") != protocol:
        raise ConfigurationError("protocol cannot change after selecting codec defaults")
    resolved = ResolvedProfile(protocol, model.model.default_value, values, parameters)
    for key in (
        "supports_stream",
        "supports_function_calling",
        "supports_vision",
        "supports_message_name",
    ):
        value = resolved.get(key, False)
        if value is not None and not isinstance(value, bool):
            raise ConfigurationError(f"{key} must be a bool, None or omit")
    if resolved.get("should_feedback_reasoning_item", "none") not in (
        "when_tool_calls",
        "always",
        "none",
    ):
        raise ConfigurationError(
            "should_feedback_reasoning_item must be when_tool_calls, always, none or omit"
        )
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


def resolve_profiles_with_hash(
    provider: ProviderProfile,
    model: ModelProfile,
    *,
    defaults_for: Callable[[str], ProtocolDefaults] | None = None,
) -> tuple[ResolvedProfile, int]:
    """Resolve profiles normally and fingerprint the complete snapshot.

    Dictionary and binding order is retained. The integer is a SHA-256 digest,
    including connection settings and all Parameter fields. Attributes excluded
    by resolve_profiles remain excluded. Payloads may contain JSON scalars,
    dictionaries, lists, tuples, bytes, sets, enums and dataclass instances.
    Unsupported or circular payloads raise ConfigurationError.

    Treat the returned snapshot as read-only when reusing it with MyLLM.
    """
    resolved = resolve_profiles(provider, model, defaults_for=defaults_for)
    try:
        encoded = json.dumps(_profile_hash_data(resolved), separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ConfigurationError("resolved profile contains an unhashable payload") from exc
    return resolved, int.from_bytes(sha256(encoded.encode("utf-8")).digest(), "big")


def _profile_hash_data(value: Any) -> Any:
    """Encode values by content and type, without copying or sorting dictionaries."""
    if isinstance(value, Omit):
        return ("omit",)
    if isinstance(value, Enum):
        return (
            "enum",
            type(value).__module__,
            type(value).__qualname__,
            _profile_hash_data(value.value),
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return (
            "dict",
            [(_profile_hash_data(key), _profile_hash_data(item)) for key, item in value.items()],
        )
    if isinstance(value, (list, tuple)):
        return (
            "list" if isinstance(value, list) else "tuple",
            [_profile_hash_data(item) for item in value],
        )
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, (set, frozenset)):
        return (
            "set" if isinstance(value, set) else "frozenset",
            sorted((_profile_hash_data(item) for item in value), key=json.dumps),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            type(value).__module__,
            type(value).__qualname__,
            [(item.name, _profile_hash_data(getattr(value, item.name))) for item in fields(value)],
        )
    raise TypeError(f"unsupported profile hash payload type: {type(value).__qualname__}")


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


def merge_profiles(older: Profile, newer: Profile) -> Profile:
    """Merge profiles of the same concrete type into an independent copy.

    None inherits; omit disables. Every other newer field replaces the whole
    older declaration, including Config dictionaries and wrapped None values.
    The older value never prevents an override, even when it is None or omit.
    """
    if not isinstance(older, Profile) or not isinstance(newer, Profile):
        raise ConfigurationError("older and newer must be profile instances")
    if type(older) is not type(newer):
        raise ConfigurationError("Cannot merge profiles of different types")

    result = copy(older)
    for name, _, _, _, _ in _declarations(type(older)):
        value = getattr(newer, name)
        if value is None:
            continue
        setattr(result, name, value)
    return deepcopy(result)
