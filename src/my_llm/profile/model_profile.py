"""Model identity, metadata and provider-default overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from my_llm.exceptions import ConfigurationError
from my_llm.profile.configuration import OverridableProfile
from my_llm.profile.types import Attribute, Config, Omit


@dataclass(frozen=True, kw_only=True)
class ModelProfile(OverridableProfile):
    """Model identity, descriptive metadata and inherited provider overrides.

    None inherits; omit disables a setting. Parameter and Config wrappers retain
    explicit default_value=None. max_context_window is descriptive and never
    changes request parameters. Attribute marks identity and descriptive data;
    extend settings with typed Parameter or Config fields.
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
