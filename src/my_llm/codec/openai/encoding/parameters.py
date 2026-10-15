"""OpenAI request keyword construction with profile-based reasoning bindings."""

from copy import deepcopy
from typing import Any

from openai import Omit

from my_llm.codec.utils.common import _expand
from my_llm.profile.utils.common import EFFORTS, merge_dicts
from my_llm.profile.utils.resolution import ResolvedProfile, validate_max_tokens
from my_llm.types.exceptions import ConfigurationError


def request_parameters(
    profile: ResolvedProfile,
    *,
    extra_headers: Any = None,
    extra_body: Any = None,
    max_tokens: Any = None,
    reasoning_effort: str = "none",
    reasoning: bool | None = None,
) -> dict[str, Any]:
    from my_llm.types.exceptions import UnsupportedCapabilityError

    if reasoning_effort not in EFFORTS:
        raise ConfigurationError(f"unsupported reasoning_effort: {reasoning_effort!r}")
    if reasoning is not None and not isinstance(reasoning, bool):
        raise ConfigurationError("reasoning must be a bool")
    if reasoning is False and reasoning_effort != "none":
        raise ConfigurationError("reasoning=False conflicts with a non-none effort")
    if reasoning is True and reasoning_effort == "none":
        reasoning_effort = "medium"
    headers = merge_dicts(profile.values["extra_headers"], extra_headers)
    body = merge_dicts(profile.values["extra_body"], extra_body)
    if set(body) & {"model", "stream", "messages", "input", "tools", "previous_response_id"}:
        raise ConfigurationError(
            "extra_body cannot override model, stream, messages, input, tools or previous_response_id"
        )
    kwargs: dict[str, Any] = {}
    if reasoning_effort != "none":
        if not profile.get("supports_reasoning", False):
            raise UnsupportedCapabilityError("model does not support reasoning")
        bindings = [profile.get("reasoning_enable_parameter", {})]
        mapped = reasoning_effort
        if profile.get("supports_reasoning_effort", False):
            mapped = profile.get("reasoning_effort_mapping", {}).get(reasoning_effort)
            if not isinstance(mapped, str) or not mapped:
                raise UnsupportedCapabilityError(
                    f"model does not support effort {reasoning_effort!r}"
                )
            bindings.append(profile.get("reasoning_effort_parameter", {}))
        for binding_index, binding in enumerate(bindings):
            if set(binding) - {"reasoning", "reasoning_effort", "extra_body"}:
                raise ConfigurationError(
                    "reasoning bindings may target reasoning, reasoning_effort or extra_body"
                )
            for key, raw_value in binding.items():
                if isinstance(raw_value, Omit):
                    continue
                value = _expand(raw_value, mapped)
                if key == "extra_body":
                    if isinstance(value, dict) and set(value) & {
                        "model",
                        "stream",
                        "messages",
                        "input",
                        "tools",
                        "previous_response_id",
                    }:
                        raise ConfigurationError(
                            "reasoning extra_body cannot override protocol parameters"
                        )
                    body = merge_dicts(body, value)
                elif key == "reasoning":
                    if profile.api_type != "responses":
                        raise ConfigurationError("reasoning SDK parameter requires Responses API")
                    combined = merge_dicts({"summary": "detailed"}, kwargs.get(key), raw_value)
                    if binding_index == 1 and "effort" not in raw_value:
                        combined["effort"] = mapped
                    kwargs[key] = _expand(combined, mapped)
                else:
                    if profile.api_type != "chat_completions" or not isinstance(value, str):
                        raise ConfigurationError(
                            "reasoning_effort SDK parameter requires Chat Completions"
                        )
                    kwargs[key] = value
    tokens = profile.get("max_tokens") if max_tokens is None else max_tokens
    if isinstance(tokens, Omit):
        tokens = None
    validate_max_tokens(tokens)
    if tokens is not None:
        kwargs[
            "max_output_tokens" if profile.api_type == "responses" else "max_completion_tokens"
        ] = tokens
    # SDK extra_body takes precedence over named kwargs: generated fields must win.
    for key, value in kwargs.items():
        if key in body:
            body[key] = value
    if headers:
        kwargs["extra_headers"] = headers
    if body:
        kwargs["extra_body"] = body
    if profile.get("usage_mode", "none") == "stream_options":
        kwargs["stream_options"] = {"include_usage": True}
        if "stream_options" in body:
            body["stream_options"] = {"include_usage": True}
    # Named reasoning parameters are newly constructed; only extras may borrow
    # nested values from a profile or caller-owned configuration.
    return deepcopy(kwargs) if body or headers else kwargs
