"""SDK-root parameter paths and request-owned generation controls."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from my_llm.codec.shared.conversion import _MISSING, plain, read
from my_llm.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.profile.configuration import EFFORTS, merge_dicts
from my_llm.profile.paths import path_keys
from my_llm.profile.resolution import ResolvedProfile, validate_max_tokens
from my_llm.profile.types import Omit, Parameter, omit


@dataclass(frozen=True, slots=True)
class PreparedOptions:
    """Validated generation controls passed to protocol extension hooks.

    Attributes:
        tokens: Resolved token limit, None, or omit.
        headers: Request headers after merging call and profile values.
        body: Native generation options after merging call and profile values.
        effort: Mapped native reasoning effort, or None when inactive.
        explicit_tokens: Whether the call supplied max_tokens explicitly.

    Frozen dataclass returned by BaseDecoder.validate_parameters. Treat its
    nested dictionaries as read-only when passing options between hooks.
    """

    tokens: Any
    headers: dict
    body: dict
    effort: str | None
    explicit_tokens: bool = False


def path_value(value: Any, path: str, default: Any = None) -> Any:
    """Read only the addressed keys; a present None is a value, not absence."""
    for key in path_keys(path):
        if isinstance(key, int):
            if not isinstance(value, (list, tuple)) or key >= len(value):
                return default
            value = value[key]
        else:
            value = read(value, key, _MISSING)
        if value is _MISSING:
            return default
    return value


def set_path(value: dict, path: str, item: Any) -> None:
    """Write one addressed value, creating only the necessary containers."""
    keys = path_keys(path)
    for index, key in enumerate(keys):
        if isinstance(key, int):
            if not isinstance(value, list):
                raise ConfigurationError(f"path requires a list: {path!r}")
            if key >= len(value):
                value.extend([None] * (key + 1 - len(value)))
        elif not isinstance(value, dict):
            raise ConfigurationError(f"path requires a dictionary: {path!r}")
        if index == len(keys) - 1:
            value[key] = item
            return
        child = value.get(key) if isinstance(value, dict) else value[key]
        if child is None:
            child = [] if isinstance(keys[index + 1], int) else {}
        elif not isinstance(child, (dict, list)):
            child = plain(child)
        value[key] = child
        value = child


def delete_path(value: dict, path: str) -> None:
    """Delete a present parameter without creating a missing parent."""
    keys = path_keys(path)
    for key in keys[:-1]:
        value = (
            read(value, key, _MISSING)
            if isinstance(key, str)
            else (value[key] if isinstance(value, list) and key < len(value) else _MISSING)
        )
        if value is _MISSING or value is None:
            return
    key = keys[-1]
    if isinstance(value, dict) and isinstance(key, str):
        value.pop(key, None)
    elif isinstance(value, list) and isinstance(key, int) and key < len(value):
        del value[key]


def merge_parameters(*layers: dict | Omit | None) -> dict:
    """Merge explicitly configured dictionaries and consume deletion markers."""
    result = {}
    for layer in layers:
        if isinstance(layer, Omit):
            result.clear()
        elif layer is not None:
            for key, value in layer.items():
                if isinstance(value, Omit):
                    result.pop(key, None)
                elif isinstance(value, dict):
                    previous = result.get(key)
                    if not isinstance(previous, dict):
                        previous = plain(previous)
                    result[key] = merge_parameters(
                        previous if isinstance(previous, dict) else {}, value
                    )
                else:
                    result[key] = _parameter_copy(value)
    return result


def _parameter_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return merge_parameters(value)
    if isinstance(value, list):
        return [_parameter_copy(item) for item in value if not isinstance(item, Omit)]
    return deepcopy(value)


def resolve_options(
    profile: ResolvedProfile,
    *,
    extra_headers: Any = None,
    extra_body: Any = None,
    max_tokens: Any = _MISSING,
    reasoning: bool | None = None,
    reasoning_effort: str = "none",
    **local: Any,
) -> PreparedOptions:
    """Validate controls once, independently of their native parameter paths."""
    if reasoning_effort not in EFFORTS:
        raise ConfigurationError(f"unsupported reasoning_effort: {reasoning_effort!r}")
    if reasoning is not None and not isinstance(reasoning, bool):
        raise ConfigurationError("reasoning must be a bool")
    if reasoning is False and reasoning_effort != "none":
        raise ConfigurationError("reasoning=False conflicts with a non-none effort")
    effort = "medium" if reasoning is True and reasoning_effort == "none" else reasoning_effort
    if effort != "none":
        enable = profile.parameters.get("reasoning_enable")
        if isinstance(enable, Parameter) and not enable.supports:
            raise UnsupportedCapabilityError("model does not support reasoning")
        if profile.parameter("reasoning_effort", "input") is not None:
            mapping = profile.get("reasoning_effort_mapping")
            effort = mapping.get(effort) if mapping is not None else None
            if not isinstance(effort, str) or not effort:
                raise UnsupportedCapabilityError(
                    f"model does not support effort {reasoning_effort!r}"
                )
    if max_tokens is _MISSING:
        parameter = profile.parameter("max_tokens", "input")
        tokens = parameter.default_value if parameter is not None else omit
    else:
        tokens = max_tokens
    if not isinstance(tokens, Omit):
        validate_max_tokens(tokens)
    headers = merge_dicts(profile.get("extra_headers", {}), extra_headers)
    body = merge_dicts(profile.get("extra_body", {}), extra_body, preserve_omit=True)
    return PreparedOptions(
        tokens,
        deepcopy(headers),
        merge_parameters(body),
        None if effort == "none" else effort,
        max_tokens is not _MISSING,
    )


def check_protected(body: dict, protected: set[str]) -> None:
    if not isinstance(body, dict):
        raise ConfigurationError("request body parameters must be dictionaries")
    if any(key in body for key in protected):
        raise ConfigurationError("extra_body cannot override protocol parameters")


def validate_parameter_paths(
    profile: ResolvedProfile,
    options: PreparedOptions,
    *,
    protected: set[str],
    containers: tuple[str, ...] = (),
) -> None:
    """Check declared output routes without traversing native input fields."""
    for name, parameter in profile.request_parameters:
        path = parameter.path
        if name in ("reasoning_enable", "reasoning_effort") and options.effort is None:
            continue
        keys = path_keys(path)
        target = keys[1:] if keys[0] in containers else keys
        if not target or target[0] in protected:
            raise ConfigurationError(f"{name} parameter path cannot override protocol parameters")


def _apply_parameter(
    request: dict,
    parameter: Parameter,
    *,
    generated: Any = _MISSING,
    overlay: str | None = None,
) -> None:
    """Apply a control, present value or default, in that order of precedence."""
    path = parameter.path
    # The native Decoder declares whether an SDK body overlay aliases root fields.
    keys = path_keys(path)
    overlay_values = request.get(overlay) if overlay is not None else None
    overlay_root = (
        isinstance(overlay_values, dict) and keys[0] != overlay and keys[0] in overlay_values
    )
    if overlay_root and keys[0] not in request:
        request[keys[0]] = overlay_values[keys[0]]
    configured = parameter.default_value
    if generated is not _MISSING:
        if isinstance(generated, Omit):
            delete_path(request, path)
        else:
            set_path(request, path, _parameter_copy(generated))
        if overlay_root:
            if keys[0] in request:
                overlay_values[keys[0]] = request[keys[0]]
            else:
                overlay_values.pop(keys[0], None)
        return
    current = path_value(request, path, _MISSING)
    if current is not _MISSING:
        if isinstance(current, Omit):
            delete_path(request, path)
        return
    if not isinstance(configured, Omit):
        set_path(request, path, _parameter_copy(configured))


def request_parameters(
    profile: ResolvedProfile,
    options: PreparedOptions,
    *,
    body_container: str | None = None,
    headers_field: str | None = "$.extra_headers",
    overlay: str | None = None,
    omit_empty_body: bool = False,
) -> dict:
    """Build SDK keyword dictionaries using every declared request parameter."""
    body = deepcopy(options.body)
    request = {body_container: body} if body_container is not None else body
    if omit_empty_body and not body:
        request = {}
    if headers_field is not None and options.headers:
        set_path(request, headers_field, deepcopy(options.headers))
    enable = profile.parameter("reasoning_enable", "input")
    if (
        options.effort is not None
        and enable is not None
        and not isinstance(enable.default_value, Omit)
    ):
        _apply_parameter(request, enable, generated=enable.default_value, overlay=overlay)
    for name, parameter in profile.request_parameters:
        if name in ("reasoning_enable", "reasoning_effort"):
            continue
        generated = options.tokens if name == "max_tokens" and options.explicit_tokens else _MISSING
        _apply_parameter(request, parameter, generated=generated, overlay=overlay)
    effort = profile.parameter("reasoning_effort", "input")
    if options.effort is not None and effort is not None:
        _apply_parameter(request, effort, generated=options.effort, overlay=overlay)
    return request


def validate_request_options(
    profile: ResolvedProfile,
    options: PreparedOptions,
    *,
    protected: set[str],
    containers: tuple[str, ...] = (),
) -> PreparedOptions:
    """Protect request envelopes and validate all configured parameter routes."""
    check_protected(options.body, protected)
    validate_parameter_paths(profile, options, protected=protected, containers=containers)
    return options
