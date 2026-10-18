"""Protocol defaults and shared provider/model configuration.

None inherits a value, while OpenAI's omit disables it. Dictionary layering
overrides individual keys (not whole dictionaries); omit removes a key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from openai import Omit

from my_llm.exceptions import ConfigurationError

type Protocol = Literal["openai_chat", "openai_responses", "anthropic", "gemini", "ollama"]

PROTOCOLS = ("openai_chat", "openai_responses", "anthropic", "gemini", "ollama")
type ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]
type Parameter = dict[str, Any] | Omit | None

EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


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


def merge_dicts(*layers: Parameter) -> dict[str, Any]:
    """Merge keys without mutating layers; nested values are borrowed read-only."""
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
                    result[key] = value
    return result
