"""Shared native request parameter validation."""

from copy import deepcopy
from typing import Any

from openai import Omit

from my_llm.codec.utils.common import _expand
from my_llm.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.profile.utils.common import EFFORTS, merge_dicts
from my_llm.profile.utils.resolution import ResolvedProfile, validate_max_tokens


def native_parameters(
    profile: ResolvedProfile,
    *,
    request_fields: tuple[str, ...],
    extra_headers: Any = None,
    extra_body: Any = None,
    max_tokens: Any = None,
    reasoning: bool | None = None,
    reasoning_effort: str = "none",
    previous_response_id: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], str | None]:
    if previous_response_id is not None and not isinstance(previous_response_id, Omit):
        raise ConfigurationError("previous_response_id requires OpenAI Responses API")
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
    headers = merge_dicts(profile.values["extra_headers"], extra_headers)
    body = merge_dicts(profile.values["extra_body"], extra_body)
    protected = {
        "model",
        "stream",
        "messages",
        "contents",
        "input",
        "tools",
        "system",
        "system_instruction",
        "previous_response_id",
        "config",
    }
    if set(body) & protected:
        raise ConfigurationError("extra_body cannot override protocol parameters")
    if effort != "none":
        bindings = [profile.get("reasoning_enable_parameter", {})]
        if profile.get("supports_reasoning_effort", False):
            bindings.append(profile.get("reasoning_effort_parameter", {}))
        for binding in bindings:
            expanded = _expand(binding, effort)
            binding_body = expanded.pop("extra_body", None)
            if set(expanded) & protected or (
                isinstance(binding_body, dict) and set(binding_body) & protected
            ):
                raise ConfigurationError("reasoning bindings cannot override protocol parameters")
            body = merge_dicts(body, binding_body, expanded)
    # Isolate mutable configuration once, after merging; codecs borrow it read-only.
    return (
        tokens,
        deepcopy(headers),
        deepcopy({key: body[key] for key in request_fields if key in body}),
        None if effort == "none" else effort,
    )
