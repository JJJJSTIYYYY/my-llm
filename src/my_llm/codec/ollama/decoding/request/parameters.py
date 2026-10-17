"""Ollama generation options, output-token limit and thinking switch."""

from inspect import signature
from typing import Any

from ollama import AsyncClient

from my_llm.codec.shared.options import native_options
from my_llm.codec.shared.values import plain
from my_llm.exceptions import ConfigurationError
from my_llm.profile.resolution import ResolvedProfile

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}
REQUEST_FIELDS = set(signature(AsyncClient.chat).parameters) - PROTECTED - {"self"}


def request_parameters(profile: ResolvedProfile, **parameters: Any) -> dict[str, Any]:
    tokens, _, request, effort = native_options(
        profile, request_fields=REQUEST_FIELDS, protected=PROTECTED, **parameters
    )
    if tokens is not None:
        request["options"] = {**plain(request.get("options") or {}), "num_predict": tokens}
    if effort:
        request["think"] = _thinking_value(request, profile, effort)
    return request


def _thinking_value(request: dict[str, Any], profile: ResolvedProfile, effort: str) -> str | bool:
    if profile.get("supports_reasoning_effort", False):
        # Ollama's protocol permits model-defined strings. The profile owns
        # the model's supported mapping; there is no universal level enum.
        return effort
    thinking = request.get("think")
    if thinking is False:
        raise ConfigurationError("reasoning conflicts with think=False")
    return True if thinking is None else thinking
