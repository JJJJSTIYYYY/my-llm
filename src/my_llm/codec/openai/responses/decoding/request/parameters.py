"""Native reasoning bindings and request options."""

from typing import Any

from openai import Omit

from my_llm.codec.shared.options import (
    bindings,
    check_protected,
    overlay_generated,
    resolve_options,
)
from my_llm.codec.shared.values import _expand
from my_llm.exceptions import ConfigurationError
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}


def request_parameters(profile: ResolvedProfile, **parameters: Any) -> dict:
    tokens, headers, body, effort = resolve_options(profile, protected=PROTECTED, **parameters)
    request = {}
    for index, binding in enumerate(bindings(profile, effort, expand=False)):
        binding = {key: value for key, value in binding.items() if not isinstance(value, Omit)}
        if set(binding) - {"reasoning", "extra_body"}:
            raise ConfigurationError("reasoning bindings contain unsupported SDK parameters")
        extra = _expand(binding.get("extra_body", {}), effort)
        check_protected(extra, PROTECTED)
        body = merge_dicts(body, extra)
        if "reasoning" in binding:
            value = binding["reasoning"]
            if not isinstance(value, dict):
                raise ConfigurationError("reasoning must be a dictionary")
            combined = merge_dicts({"summary": "detailed"}, request.get("reasoning"), value)
            if index == 1 and "effort" not in value:
                combined["effort"] = effort
            request["reasoning"] = _expand(combined, effort)
    if tokens is not None:
        request["max_output_tokens"] = tokens
    body = overlay_generated(request, body)
    if headers:
        request["extra_headers"] = headers
    if body:
        request["extra_body"] = body
    return request
