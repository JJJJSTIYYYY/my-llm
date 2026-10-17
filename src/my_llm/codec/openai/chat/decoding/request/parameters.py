"""Native reasoning bindings and request options."""

from typing import Any

from my_llm.codec.shared.options import (
    bindings,
    check_protected,
    overlay_generated,
    resolve_options,
)
from my_llm.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}


def request_parameters(profile: ResolvedProfile, **parameters: Any) -> dict:
    tokens, headers, body, effort = resolve_options(profile, protected=PROTECTED, **parameters)
    request = {}
    for binding in bindings(profile, effort):
        if set(binding) - {"reasoning_effort", "extra_body"}:
            raise ConfigurationError("reasoning bindings contain unsupported SDK parameters")
        extra = binding.get("extra_body", {})
        check_protected(extra, PROTECTED)
        body = merge_dicts(body, extra)
        if "reasoning_effort" in binding:
            value = binding["reasoning_effort"]
            if not isinstance(value, str):
                raise ConfigurationError("reasoning_effort must be a string")
            request["reasoning_effort"] = value
    if body.get("n") not in (None, 1):
        raise UnsupportedCapabilityError("only one generated candidate is supported")
    if tokens is not None:
        request["max_completion_tokens"] = tokens
    if profile.get("usage_mode", "none") == "stream_options":
        request["stream_options"] = {"include_usage": True}
    body = overlay_generated(request, body)
    if headers:
        request["extra_headers"] = headers
    if body:
        request["extra_body"] = body
    return request
