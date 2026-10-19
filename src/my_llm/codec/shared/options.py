"""Declarative field paths, generation-option translation and validation helpers."""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from openai import Omit

from my_llm.codec.shared.conversion import plain, read
from my_llm.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.profile.configuration import EFFORTS, merge_dicts
from my_llm.profile.paths import path_keys
from my_llm.profile.resolution import ResolvedProfile, validate_max_tokens


@dataclass(frozen=True, slots=True)
class PreparedOptions:
    """Request-owned settings; binding templates retain omit until their layers merge."""

    tokens: int | None
    headers: dict
    body: dict
    effort: str | None
    bindings: tuple[dict, ...]


def set_path(value: dict, path: str, item: Any) -> None:
    """Write a path, creating containers and retaining existing sibling fields."""
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
        child = (
            plain(child) if child is not None else ([] if isinstance(keys[index + 1], int) else {})
        )
        value[key] = child
        value = child


def path_value(value: Any, path: str | Omit) -> Any:
    """Read dotted attributes/mapping keys and numeric indexes, without dumping SDK objects."""
    if isinstance(path, Omit):
        return None
    for key in path_keys(path):
        if isinstance(key, int):
            if not isinstance(value, (list, tuple)) or key >= len(value):
                return None
            value = value[key]
        else:
            value = read(value, key)
        if value is None:
            return None
    return value


def expand_value(value: Any, effort: str) -> Any:
    """Expand containers once and capture mutable leaves of the binding."""
    if isinstance(value, dict):
        return {key: expand_value(item, effort) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_value(item, effort) for item in value]
    if type(value) in (str, int, float, bool, bytes, type(None)):
        return effort if value == "{effort}" else value
    return deepcopy(value)


def _binding_value(value: Any) -> Any:
    if isinstance(value, dict):
        return merge_bindings(value)
    if isinstance(value, list):
        return [_binding_value(item) for item in value]
    return deepcopy(value)


def merge_bindings(*layers: dict | Omit | None) -> dict:
    """Merge binding dictionaries recursively; omit deletes a key at its own depth.

    Lists and scalar values replace earlier values. Profile layering remains
    governed by merge_dicts; this rule applies to generation bindings only.
    """
    result = {}
    for layer in layers:
        if isinstance(layer, Omit):
            result.clear()
        elif layer is not None:
            for key, value in layer.items():
                if isinstance(value, Omit):
                    result.pop(key, None)
                elif isinstance(value, dict):
                    previous = plain(result.get(key))
                    result[key] = merge_bindings(
                        previous if isinstance(previous, dict) else None, value
                    )
                else:
                    result[key] = _binding_value(value)
    return result


def resolve_options(
    profile: ResolvedProfile,
    *,
    extra_headers: Any = None,
    extra_body: Any = None,
    max_tokens: Any = None,
    reasoning: bool | None = None,
    reasoning_effort: str = "none",
) -> PreparedOptions:
    """Validate and resolve common settings once at the Transport boundary."""
    if reasoning_effort not in EFFORTS:
        raise ConfigurationError(f"unsupported reasoning_effort: {reasoning_effort!r}")
    if reasoning is not None and not isinstance(reasoning, bool):
        raise ConfigurationError("reasoning must be a bool")
    if reasoning is False and reasoning_effort != "none":
        raise ConfigurationError("reasoning=False conflicts with a non-none effort")
    effort = "medium" if reasoning is True and reasoning_effort == "none" else reasoning_effort
    if effort != "none":
        if not profile.get("supports_reasoning", False):
            raise UnsupportedCapabilityError("model does not support reasoning")
        if profile.get("supports_reasoning_effort", False):
            effort = profile.get("reasoning_effort_mapping", {}).get(effort)
            if not isinstance(effort, str) or not effort:
                raise UnsupportedCapabilityError(
                    f"model does not support effort {reasoning_effort!r}"
                )
    tokens = profile.get("max_tokens") if max_tokens is None else max_tokens
    if isinstance(tokens, Omit):
        tokens = None
    validate_max_tokens(tokens)
    headers = merge_dicts(profile.get("extra_headers", {}), extra_headers)
    body = merge_dicts(profile.get("extra_body", {}), extra_body)
    effort = None if effort == "none" else effort
    bindings = []
    if effort is not None:
        bindings.append(profile.get("reasoning_enable_parameter", {}))
        if profile.get("supports_reasoning_effort", False):
            bindings.append(profile.get("reasoning_effort_parameter", {}))
    return PreparedOptions(
        tokens,
        deepcopy(headers),
        deepcopy(body),
        effort,
        tuple(expand_value(binding, effort) for binding in bindings),
    )


def check_protected(body: dict, protected: set[str]) -> None:
    if not isinstance(body, dict):
        raise ConfigurationError("request body bindings must be dictionaries")
    if set(body) & protected:
        raise ConfigurationError("extra_body cannot override protocol parameters")


def validate_max_tokens_path(
    profile: ResolvedProfile,
    tokens: int | None,
    *,
    protected: set[str],
    containers: tuple[str, ...] = (),
    request: dict | None = None,
) -> None:
    """Validate a token route using container names declared by the native transport."""
    path = profile.get("max_tokens_path")
    if tokens is None or path is None:
        return
    keys = path_keys(path)
    target = keys[1:] if keys[0] in containers else keys
    if not target or target[0] in protected:
        raise ConfigurationError("max_tokens_path cannot override protocol parameters")
    if request is not None:
        set_path(dict(request), path, tokens)


def apply_max_tokens(
    request: dict, profile: ResolvedProfile, tokens: int | None, *, overlay: str | None = None
) -> None:
    """Translate the output-token limit to its configured request path."""
    path = profile.get("max_tokens_path")
    if tokens is not None and path is not None:
        set_path(request, path, tokens)
        keys = path_keys(path)
        body = request.get(overlay) if overlay is not None else None
        if keys[0] != overlay and isinstance(body, dict) and keys[0] in body:
            body[keys[0]] = deepcopy(request[keys[0]])


def native_options(options: PreparedOptions) -> PreparedOptions:
    """Merge validated declarative bindings without resolving settings again."""
    body = merge_bindings(options.body)
    for binding in options.bindings:
        body = merge_bindings(
            body,
            binding.get("extra_body"),
            {key: value for key, value in binding.items() if key != "extra_body"},
        )
    return replace(options, body=body)


def validate_bindings(
    options: PreparedOptions,
    *,
    protected: set[str],
    allowed: set[str] | None = None,
) -> tuple[dict, ...]:
    """Validate declarative SDK bindings without changing their values."""
    checked = []
    for binding in options.bindings:
        if not isinstance(binding, dict):
            raise ConfigurationError("request body bindings must be dictionaries")
        binding = merge_bindings(binding)
        if allowed is not None and set(binding) - allowed:
            raise ConfigurationError("reasoning bindings contain unsupported SDK parameters")
        check_protected(binding, protected)
        if "extra_body" in binding:
            check_protected(binding["extra_body"], protected)
        checked.append(binding)
    return tuple(checked)


def validate_native_options(
    options: PreparedOptions,
    *,
    request_fields: set[str],
    protected: set[str],
) -> PreparedOptions:
    """Validate and merge native bindings once, before request translation."""
    check_protected(options.body, protected)
    validate_bindings(options, protected=protected)
    options = native_options(options)
    unknown = set(options.body) - request_fields
    if unknown:
        raise ConfigurationError(
            f"unsupported native request options: {', '.join(sorted(unknown))}"
        )
    return options


def prepare_overlay_options(
    profile: ResolvedProfile,
    options: PreparedOptions,
    *,
    protected: set[str],
    overlay: str,
) -> PreparedOptions:
    """Merge validated overlay bindings and check the configured token path."""
    body = merge_bindings(options.body, *(binding.get(overlay) for binding in options.bindings))
    validate_max_tokens_path(
        profile,
        options.tokens,
        protected=protected,
        containers=(overlay,),
        request={overlay: body},
    )
    return replace(options, body=body)


def overlay_request_parameters(
    request: dict,
    profile: ResolvedProfile,
    options: PreparedOptions,
    *,
    overlay: str,
    headers_field: str,
) -> dict:
    """Attach request-owned headers and an overlay, retaining generated values."""
    body = deepcopy(options.body)
    for key in request.keys() & body.keys():
        body[key] = deepcopy(request[key])
    if options.headers:
        request[headers_field] = deepcopy(options.headers)
    if body:
        request[overlay] = body
    apply_max_tokens(request, profile, options.tokens, overlay=overlay)
    return request
