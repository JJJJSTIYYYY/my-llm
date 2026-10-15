"""Protocol defaults and explicit provider/model configuration.

None inherits a value, while OpenAI's omit disables it. Dictionary layering
overrides individual keys (not whole dictionaries); omit removes a key.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal

from openai import Omit

from .exceptions import ConfigurationError, UnsupportedProviderError

type APIType = Literal["responses", "chat_completions"]
type Protocol = Literal["openai", "anthropic", "gemini", "ollama"]
type ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]
type Parameter = dict[str, Any] | Omit | None

EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, kw_only=True)
class _Overrides:
    extra_headers: Parameter = None
    extra_body: Parameter = None
    reasoning_enable_parameter: Parameter = None
    reasoning_effort_parameter: Parameter = None
    reasoning_effort_mapping: Parameter = None
    supports_reasoning_effort: bool | Omit | None = None
    content_path: str | Omit | None = None
    reasoning_content_path: str | Omit | None = None
    replay_reasoning_content_when_tool_calls: bool | Omit | None = None


@dataclass(frozen=True, kw_only=True)
class ProviderProfile(_Overrides):
    """User-defined provider; no vendors or models are registered implicitly.

    Reasoning parameters are SDK keyword dictionaries, e.g.
    {"extra_body": {"enable_thinking": True}}. In effort parameters the literal
    "{effort}" is replaced with the value from reasoning_effort_mapping.
    """

    api_type: APIType = "chat_completions"
    protocol: Protocol = "openai"
    base_url: str | Omit | None = None
    api_key: str | Omit | None = field(default=None, repr=False)
    timeout: float | Omit | None = None
    max_retries: int | Omit | None = None
    supports_stream: bool | Omit | None = None
    usage_mode: Literal["none", "native", "stream_options"] | Omit | None = None


@dataclass(frozen=True, kw_only=True)
class ModelProfile(_Overrides):
    """Model ID and overrides on top of its provider's protocol defaults."""

    model: str
    max_tokens: int | Omit | None = None
    supports_reasoning: bool | Omit | None = None
    supports_function_calling: bool | Omit | None = None
    supports_message_name: bool | Omit | None = None
    supported_roles: ClassVar[tuple[str, ...]] = ("system", "user", "assistant", "tool")


def merge_dicts(*layers: Parameter) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in layers:
        if isinstance(layer, Omit):
            result.clear()
        elif layer is not None:
            if not isinstance(layer, dict):
                raise ConfigurationError("dictionary configuration must be a dict or omit")
            for key, value in layer.items():
                if not isinstance(key, str):
                    raise ConfigurationError("configuration keys must be strings")
                if isinstance(value, Omit):
                    result.pop(key, None)
                else:
                    result[key] = deepcopy(value)
    return result


@dataclass(frozen=True)
class ResolvedProfile:
    api_type: APIType
    protocol: Protocol
    model: str
    values: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        value = self.values.get(key, default)
        return default if isinstance(value, Omit) else value


def resolve_profiles(provider: ProviderProfile, model: ModelProfile) -> ResolvedProfile:
    if not isinstance(provider, ProviderProfile) or not isinstance(model, ModelProfile):
        raise ConfigurationError("provider and model must be profile instances")
    if provider.protocol != "openai":
        raise UnsupportedProviderError(f"protocol {provider.protocol!r} is reserved")
    if provider.api_type not in ("responses", "chat_completions"):
        raise UnsupportedProviderError(f"unknown API type: {provider.api_type!r}")
    if not isinstance(model.model, str) or not model.model.strip():
        raise ConfigurationError("model must be a non-empty string")
    responses = provider.api_type == "responses"
    values: dict[str, Any] = {
        "extra_headers": {},
        "extra_body": {},
        "max_tokens": None,
        "supports_stream": True,
        "supports_reasoning": True,
        "supports_function_calling": False,
        "supports_message_name": True,
        "supports_reasoning_effort": responses,
        "replay_reasoning_content_when_tool_calls": True,
        "usage_mode": "none",
        "content_path": "response.output_text.delta" if responses else "$.choices[0].delta.content",
        "reasoning_content_path": "response.reasoning_summary_text.delta"
        if responses
        else "$.choices[0].delta.reasoning_content",
        "reasoning_effort_mapping": {effort: effort for effort in EFFORTS},
        "reasoning_enable_parameter": {"reasoning": {"summary": "detailed", "effort": "medium"}}
        if responses
        else {"extra_body": {"thinking": {"type": "enabled"}}},
        "reasoning_effort_parameter": {"reasoning": {"summary": "detailed", "effort": "{effort}"}}
        if responses
        else {"reasoning_effort": "{effort}"},
    }
    for profile in (provider, model):
        for item in fields(profile):
            if item.name in ("api_type", "protocol", "model"):
                continue
            value = getattr(profile, item.name)
            if value is None:
                continue
            if isinstance(value, dict):
                values[item.name] = merge_dicts(values.get(item.name), value)
            else:
                values[item.name] = value
    resolved = ResolvedProfile(provider.api_type, provider.protocol, model.model, values)
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
    if usage not in ("none", "native", "stream_options"):
        raise ConfigurationError("invalid usage_mode")
    if (responses and usage == "stream_options") or (not responses and usage == "native"):
        raise ConfigurationError("usage_mode is incompatible with the selected API")
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
    for key in ("content_path", "reasoning_content_path"):
        value = values[key]
        if not isinstance(value, (str, Omit)) or (isinstance(value, str) and not value):
            raise ConfigurationError(f"{key} must be a non-empty string or omit")
    validate_max_tokens(resolved.get("max_tokens"))
    client_parameters(resolved)
    return resolved


def validate_max_tokens(value: Any) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ConfigurationError("max_tokens must be a positive integer or omit")


def client_parameters(profile: ResolvedProfile, **overrides: Any) -> dict[str, Any]:
    result = {}
    for key in ("base_url", "api_key", "timeout", "max_retries"):
        value = overrides.get(key)
        if value is None:
            value = profile.get(key)
        if value is None or isinstance(value, Omit):
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
