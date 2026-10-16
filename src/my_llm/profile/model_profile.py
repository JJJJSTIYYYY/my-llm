"""Model identity, metadata and provider-default overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from openai import Omit

from my_llm.exceptions import ConfigurationError
from my_llm.profile.utils.common import _Overrides


@dataclass(frozen=True, kw_only=True)
class ModelProfile(_Overrides):
    """Model identity, descriptive metadata and provider-setting overrides.

    Args:
        model: Required, non-empty provider model ID.
        max_context_window: Descriptive integer metadata, default 256000.
            Any integer except bool is accepted. This does not change requests,
            count tokens, truncate history or set an output-token limit.
        max_tokens: Positive output-token limit; None inherits, omit unsets it.
        supports_reasoning: Whether reasoning may be requested; defaults True.
        supports_function_calling: Whether non-empty tools may be supplied;
            defaults False.
        supports_message_name: Retain message names when True (default).
        extra_headers: Header keys overriding provider defaults.
        extra_body: Extra body keys overriding provider defaults.
        reasoning_enable_parameter: Reasoning-enable SDK keywords overriding
            provider keys.
        reasoning_effort_parameter: Effort SDK keywords overriding provider keys;
            "{effort}" is replaced by the mapped effort string.
        reasoning_effort_mapping: Effort mappings overriding provider keys;
            None/omit values disable individual levels.
        supports_reasoning_effort: Override the provider's effort capability.
        content_path: Override the text-delta selector.
        reasoning_content_path: Override the reasoning-delta selector.
        replay_reasoning_content_when_tool_calls: Override whether visible reasoning is replayed. Empty reasoning
            is added only for tool calls while thinking is enabled.

    All fields except model default to None unless stated otherwise. None
    inherits provider/protocol settings; omit disables an inherited setting.
    supported_roles is the fixed system/user/assistant/tool class attribute,
    not a configurable constructor argument. Use dataclasses.replace() to
    create an updated profile, then construct a new MyLLM instance.

    Raises:
        ConfigurationError: max_context_window is not an integer. Other
            configuration is validated when constructing MyLLM or generating.
    """

    model: str
    max_context_window: int = 256000
    max_tokens: int | Omit | None = None
    supports_reasoning: bool | Omit | None = None
    supports_function_calling: bool | Omit | None = None
    supports_message_name: bool | Omit | None = None
    supported_roles: ClassVar[tuple[str, ...]] = ("system", "user", "assistant", "tool")

    def __post_init__(self) -> None:
        if not isinstance(self.max_context_window, int) or isinstance(
            self.max_context_window, bool
        ):
            raise ConfigurationError("max_context_window must be an integer")
