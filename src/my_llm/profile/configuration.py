"""Protocol defaults and shared provider/model configuration.

None inherits a value, while OpenAI's omit disables it. Dictionary layering
overrides individual keys (not whole dictionaries); omit removes a key.
Generation bindings retain deletion markers until request preparation merges them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from openai import Omit

from my_llm.exceptions import ConfigurationError

type Protocol = str
type ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]
type Parameter = dict[str, Any] | Omit | None

EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, kw_only=True)
class ProtocolDefaults:
    """Required standard settings declared by each protocol's Codec.

    Use omit for selectors or parameter bindings that the protocol does not use.
    Mutable binding values are borrowed read-only and copied during resolution.
    """

    replay_reasoning_content_when_tool_calls: bool
    supports_reasoning_effort: bool
    max_tokens_path: str | Omit
    content_path: str | Omit
    reasoning_content_path: str | Omit
    reasoning_enable_parameter: dict[str, Any] | Omit
    reasoning_effort_parameter: dict[str, Any] | Omit


@dataclass(frozen=True, kw_only=True)
class OverridableProfile:
    """Shared provider/model overrides; None inherits and omit disables a value."""

    extra_headers: Parameter = None
    extra_body: Parameter = None
    reasoning_enable_parameter: Parameter = None
    reasoning_effort_parameter: Parameter = None
    reasoning_effort_mapping: Parameter = None
    supports_reasoning_effort: bool | Omit | None = None
    max_tokens_path: str | Omit | None = None
    content_path: str | Omit | None = None
    reasoning_content_path: str | Omit | None = None
    replay_reasoning_content_when_tool_calls: bool | Omit | None = None


def merge_dicts(*layers: Parameter, preserve_omit: bool = False) -> dict[str, Any]:
    """Merge keys, optionally retaining binding deletion markers for later application.

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
