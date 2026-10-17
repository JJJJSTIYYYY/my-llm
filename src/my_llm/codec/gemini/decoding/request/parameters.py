"""Gemini generation, HTTP and thinking configuration."""

from typing import Any

from google.genai import types

from my_llm.codec.shared.options import native_options
from my_llm.codec.shared.values import plain
from my_llm.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile

PROTECTED = {
    "model",
    "stream",
    "messages",
    "contents",
    "input",
    "tools",
    "system_instruction",
    "previous_response_id",
    "config",
    "automatic_function_calling",
    "http_options",
}
REQUEST_FIELDS = set(types.GenerateContentConfig.model_fields) - PROTECTED


def request_parameters(profile: ResolvedProfile, **parameters: Any) -> dict[str, Any]:
    tokens, headers, config, effort = native_options(
        profile, request_fields=REQUEST_FIELDS, protected=PROTECTED, **parameters
    )
    if config.get("candidate_count") not in (None, 1):
        raise UnsupportedCapabilityError("only one generated candidate is supported")
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
    thinking = {"include_thoughts": True, **plain(config.get("thinking_config") or {})}
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
