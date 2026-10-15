"""Anthropic SDK generation parameters and thinking configuration."""

from typing import Any

from anthropic import omit

from my_llm.codec.utils.native import native_parameters
from my_llm.profile.utils.resolution import ResolvedProfile
from my_llm.types.exceptions import ConfigurationError, UnsupportedCapabilityError

REQUEST_FIELDS = ("temperature", "top_p", "top_k", "stop_sequences", "thinking", "output_config")


def request_parameters(profile: ResolvedProfile, **parameters: Any) -> dict[str, Any]:
    tokens, headers, body, effort = native_parameters(
        profile, request_fields=REQUEST_FIELDS, **parameters
    )
    if tokens is None:
        raise ConfigurationError("Anthropic requires max_tokens in the model or generation")
    request = {"max_tokens": tokens, "extra_headers": headers or omit}
    # Only native generation fields are admitted. OpenAI-specific settings are omitted.
    request.update({key: body.get(key, omit) for key in REQUEST_FIELDS})
    if effort:
        _apply_reasoning(request, body, profile, tokens, effort)
    return request


def _apply_reasoning(
    request: dict[str, Any],
    body: dict[str, Any],
    profile: ResolvedProfile,
    tokens: int,
    effort: str,
) -> None:
    thinking = body.get("thinking", {"type": "enabled", "budget_tokens": 1024})
    if not isinstance(thinking, dict) or thinking.get("type") not in ("enabled", "adaptive"):
        raise ConfigurationError("reasoning requires enabled or adaptive Anthropic thinking")
    if thinking.get("type") == "enabled" and "budget_tokens" not in thinking:
        thinking = {**thinking, "budget_tokens": 1024}
    if thinking.get("type") == "enabled" and (
        not isinstance(thinking.get("budget_tokens"), int)
        or thinking["budget_tokens"] < 1024
        or thinking["budget_tokens"] >= tokens
    ):
        raise ConfigurationError("Anthropic thinking budget must be >=1024 and < max_tokens")
    request["thinking"] = thinking
    if profile.get("supports_reasoning_effort", False):
        if effort not in ("low", "medium", "high", "xhigh", "max"):
            raise UnsupportedCapabilityError("map reasoning effort to an Anthropic-supported level")
        request["output_config"] = {**body.get("output_config", {}), "effort": effort}
