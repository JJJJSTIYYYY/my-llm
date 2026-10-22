"""Model identity, metadata and provider-default overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from my_llm.exceptions import ConfigurationError
from my_llm.profile.configuration import OverridableProfile
from my_llm.profile.types import Attribute, Config, Omit


@dataclass(frozen=True, kw_only=True)
class ModelProfile(OverridableProfile):
    """Model identity, descriptive metadata and provider-setting overrides.

    Args:
        model: Required Attribute containing a nonempty model ID.
        max_context_window: Descriptive Attribute integer, default 256000. It does
            not truncate history or enforce token limits.
        supports_function_calling: Config capability flag; protocol default False.
            Enable it before sending nonempty tool declarations.
        supports_message_name: Config controlling message-name support.

    This is a frozen keyword-only dataclass. None inherits provider settings;
    omit disables an override. supported_roles is the fixed class attribute
    ("system", "user", "assistant", "tool"), not a constructor option.
    Additional keyword fields are inherited from OverridableProfile.
    Use dataclasses.replace to derive a model profile.
    """

    model: Attribute[str]
    max_context_window: Attribute[int] = Attribute(default_value=256000)
    supports_function_calling: Config[bool] | Omit | None = None
    supports_message_name: Config[bool] | Omit | None = None
    supported_roles: ClassVar[tuple[str, ...]] = ("system", "user", "assistant", "tool")

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.max_context_window, Attribute)
            or not isinstance(self.max_context_window.default_value, int)
            or isinstance(self.max_context_window.default_value, bool)
        ):
            raise ConfigurationError("max_context_window must be an integer")
