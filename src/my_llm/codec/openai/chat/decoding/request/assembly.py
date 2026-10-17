"""Native OpenAI request assembly."""

from typing import Any

from openai import Omit

from my_llm.codec.openai.chat.decoding.request.parameters import request_parameters
from my_llm.exceptions import ConfigurationError
from my_llm.profile.resolution import ResolvedProfile


def build(messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any) -> dict:
    previous = parameters.pop("previous_response_id", None)
    has_previous = previous is not None and not isinstance(previous, Omit)
    if has_previous:
        raise ConfigurationError("previous_response_id requires Responses and a non-empty ID")
    result = request_parameters(profile, **parameters)
    result.update(model=profile.model, stream=True, messages=messages)
    if tools is not None:
        result["tools"] = tools

    return result
