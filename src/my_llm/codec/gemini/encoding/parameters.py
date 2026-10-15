"""Gemini generation, HTTP and thinking configuration."""

from typing import Any

from google.genai import types

from my_llm.codec.utils.native import native_parameters
from my_llm.profile.utils.resolution import ResolvedProfile
from my_llm.types.exceptions import ConfigurationError, UnsupportedCapabilityError

REQUEST_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "seed",
    "thinking_config",
    "response_mime_type",
    "response_schema",
    "response_json_schema",
    "presence_penalty",
    "frequency_penalty",
)


def request_parameters(profile: ResolvedProfile, **parameters: Any) -> dict[str, Any]:
    tokens, headers, config, effort = native_parameters(
        profile, request_fields=REQUEST_FIELDS, **parameters
    )
    # Admit native generation settings only, never native cloud tools or AFC overrides.
    config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    if tokens is not None:
        config["max_output_tokens"] = tokens
    if headers:
        config["http_options"] = types.HttpOptions(headers=headers)
    if effort:
        _apply_reasoning(config, profile, effort)
    return config


def _apply_reasoning(config: dict[str, Any], profile: ResolvedProfile, effort: str) -> None:
    thinking = {**config.get("thinking_config", {}), "include_thoughts": True}
    if profile.get("supports_reasoning_effort", False):
        if effort not in ("minimal", "low", "medium", "high"):
            raise UnsupportedCapabilityError("map reasoning effort to a Gemini-supported level")
        thinking["thinking_level"] = effort
        thinking.pop("thinking_budget", None)
    elif not any(key in thinking for key in ("thinking_level", "thinking_budget")):
        thinking["thinking_budget"] = -1
    config["thinking_config"] = thinking
    if thinking.get("thinking_budget") == 0:
        raise ConfigurationError("reasoning conflicts with a zero Gemini thinking budget")
