"""Native history placement and SDK request assembly."""

from typing import Any

from openai import Omit

from my_llm.codec.anthropic.decoding.request.parameters import request_parameters
from my_llm.exceptions import ConfigurationError
from my_llm.profile.resolution import ResolvedProfile


def build(messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any) -> dict:
    previous = parameters.pop("previous_response_id", None)
    if previous is not None and not isinstance(previous, Omit):
        raise ConfigurationError("previous_response_id requires OpenAI Responses API")
    result = request_parameters(profile, **parameters)
    system = [m for m in messages if m.get("role") in ("system", "developer")]
    history = [m for m in messages if m.get("role") not in ("system", "developer")]
    if not history:
        raise ConfigurationError("a non-system message is required")
    result.update(model=profile.model, stream=True, messages=history)
    if system:
        result["system"] = [
            block
            for message in system
            for block in (
                [{"type": "text", "text": message["content"]}]
                if isinstance(message["content"], str)
                else message["content"]
            )
        ]
    if tools is not None:
        result["tools"] = tools
    return result
