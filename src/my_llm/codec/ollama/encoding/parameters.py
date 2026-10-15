"""Ollama generation options, output-token limit and thinking switch."""

from typing import Any

from my_llm.codec.utils.native import native_parameters
from my_llm.profile.utils.resolution import ResolvedProfile
from my_llm.types.exceptions import ConfigurationError, UnsupportedCapabilityError

REQUEST_FIELDS = ("format", "keep_alive", "think", "options")


def request_parameters(profile: ResolvedProfile, **parameters: Any) -> dict[str, Any]:
    tokens, _, request, effort = native_parameters(
        profile, request_fields=REQUEST_FIELDS, **parameters
    )
    if tokens is not None:
        request["options"] = {**request.get("options", {}), "num_predict": tokens}
    if effort:
        request["think"] = _thinking_value(request, profile, effort)
    return request


def _thinking_value(request: dict[str, Any], profile: ResolvedProfile, effort: str) -> str | bool:
    if profile.get("supports_reasoning_effort", False):
        if effort not in ("low", "medium", "high"):
            raise UnsupportedCapabilityError("map reasoning effort to an Ollama-supported level")
        return effort
    thinking = request.get("think", True)
    if thinking is False:
        raise ConfigurationError("reasoning conflicts with think=False")
    return thinking
