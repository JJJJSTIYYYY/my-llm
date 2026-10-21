"""Resolve profile layers and validate generation/client settings."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field, fields
from functools import lru_cache
from types import UnionType
from typing import Any, Literal, TypeAliasType, Union, get_args, get_origin, get_type_hints

from my_llm.exceptions import ConfigurationError, UnsupportedProviderError
from my_llm.profile.configuration import Protocol, ProtocolDefaults, merge_dicts
from my_llm.profile.model_profile import ModelProfile
from my_llm.profile.paths import path_keys
from my_llm.profile.provider_profile import ProviderProfile
from my_llm.profile.types import Config, Omit, Parameter
from my_llm.registry import codec_for


@dataclass(frozen=True)
class ResolvedProfile:
    """Construction-time snapshot, with values and parameter bindings separated.

    Applications normally pass ProviderProfile/ModelProfile to MyLLM instead.
    Decoder validates protocol-specific constraints. Treat values, parameters
    and their nested payloads as read-only.
    """

    protocol: Protocol
    model: str
    values: dict[str, Any]
    parameters: dict[str, Parameter[Any] | Omit] = field(default_factory=dict)

    request_parameters: tuple[tuple[str, Parameter[Any]], ...] = ()
    connection: dict[str, Any] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a resolved default or code setting, falling back for omit."""
        value = self.values.get(key, default)
        return default if isinstance(value, Omit) else value

    def parameter(
        self, name: str, source: Literal["input", "output"] | None = None
    ) -> Parameter[Any] | None:
        """Read a binding by field name, optionally restricted to its input/output source."""
        value = self.parameters.get(name)
        if (
            isinstance(value, Parameter)
            and value.supports
            and (source is None or value.source == source)
        ):
            return value
        return None


def resolve_profiles(provider: ProviderProfile, model: ModelProfile) -> ResolvedProfile:
    """Validate and snapshot protocol, provider and model settings in order.

    None inherits a setting; omit disables it. Parameter wrappers replace the
    entire binding and Config dictionary defaults merge by key. A default_value
    of None remains an explicit value. Excludes descriptive model metadata.
    """
    if not isinstance(provider, ProviderProfile) or not isinstance(model, ModelProfile):
        raise ConfigurationError("provider and model must be profile instances")
    if not isinstance(model.model, str) or not model.model.strip():
        raise ConfigurationError("model must be a non-empty string")
    defaults = codec_for(provider.protocol).defaults
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
    resolved = ResolvedProfile(provider.protocol, model.model, values, parameters)
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


@lru_cache(maxsize=128)
def _declarations(profile_type: type) -> tuple[tuple[str, str, Any, str | None, str], ...]:
    """Cache type analysis, bounded for applications constructing dynamic types."""
    annotations = get_type_hints(profile_type, include_extras=True)
    result = []
    descriptive = {"protocol", "model", "models", "max_context_window"}
    for item in fields(profile_type):
        if item.name in descriptive:
            continue
        annotation = annotations[item.name]
        kinds = _kinds(annotation)
        if len(kinds) != 1:
            raise ConfigurationError(f"{item.name} must declare exactly one of Parameter or Config")
        kind = kinds.pop()
        source = item.metadata.get("source")
        scope = item.metadata.get("scope", "request")
        if scope not in ("client", "request"):
            raise ConfigurationError("parameter scope must be client or request")
        if kind == "parameter":
            if source not in ("input", "output"):
                raise ConfigurationError(f"{item.name} must declare input/output source in Profile")
            if scope == "client" and source != "input":
                raise ConfigurationError(f"{item.name} must use the input source")
        result.append((item.name, kind, annotation, source, scope))
    return tuple(result)


def _kinds(annotation: Any) -> set[str]:
    origin = get_origin(annotation)
    if origin in (Parameter, Config) or annotation in (Parameter, Config):
        kind = "parameter" if (origin or annotation) is Parameter else "config"
        arguments = get_args(annotation)
        if arguments and _kinds(arguments[0]):
            return {"parameter", "config"}
        return {kind}
    if isinstance(annotation, TypeAliasType):
        return _kinds(annotation.__value__)
    if origin in (Union, UnionType):
        return set().union(*(_kinds(argument) for argument in get_args(annotation)))
    return set()


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
