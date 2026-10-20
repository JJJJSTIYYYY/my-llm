"""Typed declarations for protocol defaults and provider/model overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from my_llm.exceptions import ConfigurationError
from my_llm.profile.types import Config, Omit, Parameter, omit

type Protocol = str
type ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]

EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, kw_only=True)
class ProtocolDefaults:
    """Settings declared by a protocol's Codec, extensible with typed wrappers.

    Parameter binds its own path, missing-value default and translation phase.
    Config controls code behavior. Dataclass field names identify each setting;
    wrapper names and comments are presentation metadata.
    """

    replay_reasoning_content_when_tool_calls: Config[bool] | Omit
    supports_reasoning_effort: Config[bool] | Omit
    max_tokens: Parameter[int] | Omit = omit
    content: Parameter[Any] | Omit = omit
    reasoning_content: Parameter[Any] | Omit = omit
    reasoning_enable: Parameter[Any] | Omit = omit
    reasoning_effort: Parameter[Any] | Omit = omit
    extra_headers: Config[dict[str, Any]] | Omit = field(
        default_factory=lambda: Config(name="Extra headers", default_value={})
    )
    extra_body: Config[dict[str, Any]] | Omit = field(
        default_factory=lambda: Config(name="Extra body", default_value={})
    )
    reasoning_effort_mapping: Config[dict[str, Any]] | Omit = field(
        default_factory=lambda: Config(
            name="Reasoning effort mapping",
            default_value={effort: effort for effort in EFFORTS},
        )
    )
    supports_stream: Config[bool] | Omit = Config(name="Supports streaming", default_value=True)
    supports_reasoning: Config[bool] | Omit = Config(name="Supports reasoning", default_value=True)
    supports_function_calling: Config[bool] | Omit = Config(
        name="Supports function calling", default_value=False
    )
    supports_message_name: Config[bool] | Omit = Config(
        name="Supports message names", default_value=True
    )
    usage_mode: Config[str] | Omit = Config(name="Usage mode", default_value="none")


@dataclass(frozen=True, kw_only=True)
class OverridableProfile:
    """Shared overrides: None inherits, omit disables, wrappers replace settings.

    Extend profiles with Parameter[T] or Config[T] fields. Parameter overrides
    replace path, phase and default together. Config dictionaries merge by key.
    A wrapper's explicit default_value=None is retained.
    """

    extra_headers: Config[dict[str, Any]] | Omit | None = None
    extra_body: Config[dict[str, Any]] | Omit | None = None
    reasoning_effort_mapping: Config[dict[str, Any]] | Omit | None = None
    supports_reasoning_effort: Config[bool] | Omit | None = None
    max_tokens: Parameter[int] | Omit | None = None
    content: Parameter[Any] | Omit | None = None
    reasoning_content: Parameter[Any] | Omit | None = None
    reasoning_enable: Parameter[Any] | Omit | None = None
    reasoning_effort: Parameter[Any] | Omit | None = None
    replay_reasoning_content_when_tool_calls: Config[bool] | Omit | None = None


def merge_dicts(
    *layers: dict[str, Any] | Omit | None, preserve_omit: bool = False
) -> dict[str, Any]:
    """Merge configuration keys, optionally retaining explicit deletion markers.

    Layers are not mutated; nested values are borrowed read-only.
    """
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
                if isinstance(value, Omit) and not preserve_omit:
                    result.pop(key, None)
                else:
                    result[key] = value
    return result
