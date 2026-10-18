"""Resolve profile layers and validate generation/client settings."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

from openai import Omit

from my_llm.exceptions import ConfigurationError, UnsupportedProviderError
from my_llm.profile.configuration import EFFORTS, Protocol, merge_dicts
from my_llm.profile.defaults import protocol_defaults
from my_llm.profile.model_profile import ModelProfile
from my_llm.profile.paths import path_keys
from my_llm.profile.provider_profile import ProviderProfile


@dataclass(frozen=True)
class ResolvedProfile:
    """Configuration snapshot with generic field validation for codec integrations.

    Applications normally pass ProviderProfile/ModelProfile to MyLLM instead.
    Transport validates protocol-specific constraints. Treat values and their
    nested payloads as read-only.
    """

    protocol: Protocol
    model: str
    values: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        """Read a resolved field, using default for missing or omitted values."""
        value = self.values.get(key, default)
        return default if isinstance(value, Omit) else value


def resolve_profiles(provider: ProviderProfile, model: ModelProfile) -> ResolvedProfile:
    """Validate and snapshot protocol, provider and model settings in order.

    Excludes descriptive context-window metadata and the provider's catalogue.
    Standard defaults come from the registered protocol's declaration.
    Transport validates SDK-specific constraints separately.
    Raises ConfigurationError subclasses for invalid configuration.
    """
    if not isinstance(provider, ProviderProfile) or not isinstance(model, ModelProfile):
        raise ConfigurationError("provider and model must be profile instances")
    if not isinstance(model.model, str) or not model.model.strip():
        raise ConfigurationError("model must be a non-empty string")
    values: dict[str, Any] = {
        "extra_headers": {},
        "extra_body": {},
        "max_tokens": None,
        "supports_stream": True,
        "supports_reasoning": True,
        "supports_function_calling": False,
        "supports_message_name": True,
        "usage_mode": "none",
        "reasoning_effort_mapping": {effort: effort for effort in EFFORTS},
        **protocol_defaults(provider.protocol),
    }
    for profile in (provider, model):
        for item in fields(profile):
            if item.name in ("protocol", "model", "models", "max_context_window"):
                continue
            value = getattr(profile, item.name)
            if value is None:
                continue
            if isinstance(value, dict):
                values[item.name] = merge_dicts(
                    values.get(item.name),
                    value,
                    preserve_omit=item.name
                    in ("reasoning_enable_parameter", "reasoning_effort_parameter"),
                )
            else:
                values[item.name] = value
    # Snapshot configuration once, after all layers have been merged.
    resolved = ResolvedProfile(provider.protocol, model.model, deepcopy(values))
    for key in (
        "supports_stream",
        "supports_reasoning",
        "supports_function_calling",
        "supports_message_name",
        "supports_reasoning_effort",
        "replay_reasoning_content_when_tool_calls",
    ):
        value = resolved.get(key, False)
        if not isinstance(value, bool):
            raise ConfigurationError(f"{key} must be a bool or omit")
    if not resolved.get("supports_stream", False):
        raise UnsupportedProviderError("provider must support upstream streaming")
    usage = resolved.get("usage_mode", "none")
    if not isinstance(usage, str) or not usage:
        raise ConfigurationError("usage_mode must be a non-empty string")
    for key in (
        "extra_headers",
        "extra_body",
        "reasoning_enable_parameter",
        "reasoning_effort_parameter",
        "reasoning_effort_mapping",
    ):
        value = values[key]
        if not isinstance(value, (dict, Omit)):
            raise ConfigurationError(f"{key} must be a dict or omit")
    for key in ("content_path", "reasoning_content_path", "max_tokens_path"):
        value = values[key]
        if not isinstance(value, (str, Omit)) or (isinstance(value, str) and not value):
            raise ConfigurationError(f"{key} must be a non-empty string or omit")
    if resolved.get("max_tokens_path") is not None:
        path_keys(resolved.get("max_tokens_path"))
    validate_max_tokens(resolved.get("max_tokens"))
    client_parameters(resolved)
    return resolved


def validate_max_tokens(value: Any) -> None:
    """Require a positive integer or None; raise ConfigurationError otherwise."""
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ConfigurationError("max_tokens must be a positive integer or omit")


def client_parameters(profile: ResolvedProfile) -> dict[str, Any]:
    """Validate SDK connection fields from the construction-time profile.

    Returns base_url/api_key/timeout/max_retries keyword arguments, excluding
    unset/omitted values. Invalid values raise ConfigurationError.
    """
    result = {}
    for key in ("base_url", "api_key", "timeout", "max_retries"):
        value = profile.get(key)
        if value is None:
            continue
        if key in ("api_key", "base_url") and (not isinstance(value, str) or not value.strip()):
            raise ConfigurationError(f"{key} must be a non-empty string")
        if key == "timeout" and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
        ):
            raise ConfigurationError("timeout must be positive")
        if key == "max_retries" and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ConfigurationError("max_retries must be a non-negative integer")
        result[key] = value
    return result
