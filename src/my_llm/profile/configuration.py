"""Typed declarations for protocol defaults and provider/model overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from my_llm.basic.exceptions import ConfigurationError
from my_llm.profile.base import Profile
from my_llm.profile.types import Config, Omit, Parameter, omit, set_attr

type Protocol = str
type ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]

EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, kw_only=True)
class ProtocolDefaults(Profile):
    """Declare default settings for a registered protocol.

    Required keyword argument feedback_reasoning_content is a Config
    or omit. Other fields provide default parameter routes, extra_headers,
    extra_body, reasoning_effort_mapping, capability flags and usage_mode.
    See OverridableProfile for the shared setting meanings.

    Parameter field metadata fixes source="input" or "output"; scope="client"
    marks connection parameters such as api_key. Config and Parameter names are
    assigned from field names. Extend this frozen keyword-only dataclass for
    custom protocol settings; applications normally use ProviderProfile and
    ModelProfile to override protocol defaults.
    """

    feedback_reasoning_content: Config[Literal["when_tool_calls", "always", "none"]] | Omit
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
    supports_vision: Config[bool] | Omit = Config(default_value=False)
    supports_message_name: Config[bool] | Omit = Config(default_value=True)
    usage_mode: Config[str] | Omit = Config(default_value="none")

    @set_attr
    def __post_init__(self) -> None:
        pass


@dataclass(frozen=True, kw_only=True)
class OverridableProfile(Profile):
    """Shared provider/model overrides, all keyword-only and defaulting to None.

    Args:
        extra_headers: Config of default request headers.
        extra_body: Config of native generation options.
        reasoning_effort_mapping: Config mapping public effort levels to native values.
        max_tokens: Input Parameter for the output token limit.
        content: Output Parameter selecting visible text.
        reasoning_content: Output Parameter selecting visible reasoning.
        reasoning_enable: Input Parameter enabling reasoning.
        reasoning_effort: Input Parameter selecting native reasoning effort.
        feedback_reasoning_content: Config selecting feedback of reasoning from
            tool-call messages ("when_tool_calls"), all history ("always") or
            no history ("none"). Omit disables feedback.

    None inherits a declaration; omit disables it. Parameter overrides replace
    path, support and default together. Config dictionaries merge by key, with
    omit deleting a key. A wrapper's default_value=None remains explicit.
    Extend profiles with typed Parameter, Config or Attribute fields.
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
    feedback_reasoning_content: (
        Config[Literal["when_tool_calls", "always", "none"]] | Omit | None
    ) = None

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
