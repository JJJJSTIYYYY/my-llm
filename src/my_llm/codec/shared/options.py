"""Protocol-independent validation and ownership of generation options."""

from copy import deepcopy
from typing import Any

from openai import Omit

from my_llm.codec.shared.values import _expand
from my_llm.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.profile.configuration import EFFORTS, merge_dicts
from my_llm.profile.resolution import ResolvedProfile, validate_max_tokens


def resolve_options(
    profile: ResolvedProfile,
    *,
    protected: set[str],
    extra_headers: Any = None,
    extra_body: Any = None,
    max_tokens: Any = None,
    reasoning: bool | None = None,
    reasoning_effort: str = "none",
) -> tuple[Any, dict, dict, str | None]:
    """Resolve common policy; native names and bindings belong to the decoder."""
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
    check_protected(body, protected)
    return tokens, deepcopy(headers), deepcopy(body), None if effort == "none" else effort


def check_protected(body: dict, protected: set[str]) -> None:
    if not isinstance(body, dict):
        raise ConfigurationError("request body bindings must be dictionaries")
    if set(body) & protected:
        raise ConfigurationError("extra_body cannot override protocol parameters")


def bindings(profile: ResolvedProfile, effort: str | None, *, expand: bool = True):
    """Yield isolated, expanded declarative bindings in layer order."""
    if effort is None:
        return
    values = [profile.get("reasoning_enable_parameter", {})]
    if profile.get("supports_reasoning_effort", False):
        values.append(profile.get("reasoning_effort_parameter", {}))
    for value in values:
        yield _expand(value, effort) if expand else deepcopy(value)


def native_options(
    profile: ResolvedProfile, *, request_fields: set[str], protected: set[str], **parameters: Any
):
    """Merge declarative body bindings and reject unsupported options explicitly."""
    tokens, headers, body, effort = resolve_options(profile, protected=protected, **parameters)
    for binding in bindings(profile, effort):
        expanded_body = binding.pop("extra_body", None)
        check_protected(binding, protected)
        if expanded_body is not None:
            check_protected(expanded_body, protected)
        body = merge_dicts(body, expanded_body, binding)
    unknown = set(body) - request_fields
    if unknown:
        raise ConfigurationError(
            f"unsupported native request options: {', '.join(sorted(unknown))}"
        )
    return tokens, headers, deepcopy(body), effort


def overlay_generated(request: dict, body: dict) -> dict:
    """Keep generated values authoritative in a caller-selected overlay."""
    for key in request.keys() & body.keys():
        body[key] = deepcopy(request[key])
    return body
