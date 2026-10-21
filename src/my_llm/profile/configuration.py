"""Typed declarations for protocol defaults and provider/model overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from my_llm.exceptions import ConfigurationError
from my_llm.profile.types import Config, Omit, Parameter, omit, set_attr

type Protocol = str
type ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]

EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, kw_only=True)
class ProtocolDefaults:
    """Settings declared by a protocol's Codec, extensible with typed wrappers.

    Parameter binds its own path, default, support and fixed input/output source.
    Field metadata scope="client" targets connection options (built-in api_key);
    other input parameters target requests.
    Config controls code behavior. Dataclass field names identify each setting;
    wrapper identities are fixed by their Profile fields.
    """

    replay_reasoning_content_when_tool_calls: Config[bool] | Omit
    api_key: Parameter[str] | Omit = field(
        default=Parameter(path="$.api_key", default_value=omit),
        repr=False,
        metadata={"scope": "client", "source": "input"},
    )
    max_tokens: Parameter[int] | Omit = field(default=omit, metadata={"source": "input"})
    content: Parameter[Any] | Omit = field(default=omit, metadata={"source": "output"})
    reasoning_content: Parameter[Any] | Omit = field(default=omit, metadata={"source": "output"})
    reasoning_enable: Parameter[Any] | Omit = field(default=omit, metadata={"source": "input"})
    reasoning_effort: Parameter[Any] | Omit = field(default=omit, metadata={"source": "input"})
    extra_headers: Config[dict[str, Any]] | Omit = field(
        default_factory=lambda: Config(default_value={})
    )
    extra_body: Config[dict[str, Any]] | Omit = field(
        default_factory=lambda: Config(default_value={})
    )
    reasoning_effort_mapping: Config[dict[str, Any]] | Omit = field(
        default_factory=lambda: Config(
            default_value={effort: effort for effort in EFFORTS},
        )
    )
    supports_stream: Config[bool] | Omit = Config(default_value=True)
    supports_function_calling: Config[bool] | Omit = Config(default_value=False)
    supports_message_name: Config[bool] | Omit = Config(default_value=True)
    usage_mode: Config[str] | Omit = Config(default_value="none")

    @set_attr
    def __post_init__(self) -> None:
        pass


@dataclass(frozen=True, kw_only=True)
class OverridableProfile:
    """Shared overrides: None inherits, omit disables, wrappers replace settings.

    Extend profiles with Parameter[T] or Config[T] fields. Parameter overrides
    replace path, support and default together. Config dictionaries merge by key.
    A wrapper's explicit default_value=None is retained.
    """

    extra_headers: Config[dict[str, Any]] | Omit | None = None
    extra_body: Config[dict[str, Any]] | Omit | None = None
    reasoning_effort_mapping: Config[dict[str, Any]] | Omit | None = None
    max_tokens: Parameter[int] | Omit | None = field(default=None, metadata={"source": "input"})
    content: Parameter[Any] | Omit | None = field(default=None, metadata={"source": "output"})
    reasoning_content: Parameter[Any] | Omit | None = field(
        default=None, metadata={"source": "output"}
    )
    reasoning_enable: Parameter[Any] | Omit | None = field(
        default=None, metadata={"source": "input"}
    )
    reasoning_effort: Parameter[Any] | Omit | None = field(
        default=None, metadata={"source": "input"}
    )
    replay_reasoning_content_when_tool_calls: Config[bool] | Omit | None = None

    @set_attr
    def __post_init__(self) -> None:
        pass


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
