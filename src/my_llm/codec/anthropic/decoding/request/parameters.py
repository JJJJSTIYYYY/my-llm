"""Anthropic SDK generation parameters and thinking configuration."""

from inspect import signature
from typing import Any

from anthropic.resources.messages import AsyncMessages

from my_llm.codec.shared.options import native_options
from my_llm.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile

PROTECTED = {"model", "stream", "messages", "input", "tools", "system", "previous_response_id"}
REQUEST_FIELDS = (
    set(signature(AsyncMessages.create).parameters)
    - PROTECTED
    - {"self", "extra_headers", "extra_query", "extra_body", "timeout"}
)


def request_parameters(profile: ResolvedProfile, **parameters: Any) -> dict[str, Any]:
    tokens, headers, body, effort = native_options(
        profile, request_fields=REQUEST_FIELDS, protected=PROTECTED, **parameters
    )
    if tokens is None:
        raise ConfigurationError("Anthropic requires max_tokens in the model or generation")
    request = {**body, "max_tokens": tokens}
    if headers:
        request["extra_headers"] = headers
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
