"""Model identity, metadata and provider-default overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from my_llm.exceptions import ConfigurationError
from my_llm.profile.configuration import OverridableProfile
from my_llm.profile.types import Config, Omit


@dataclass(frozen=True, kw_only=True)
class ModelProfile(OverridableProfile):
    """Model identity, descriptive metadata and inherited provider overrides.

    None inherits; omit disables a setting. Parameter and Config wrappers retain
    explicit default_value=None. max_context_window is descriptive and never
    changes request parameters. Extend with typed Parameter or Config fields.
    """

    model: str
    max_context_window: int = 256000
    supports_reasoning: Config[bool] | Omit | None = None
    supports_function_calling: Config[bool] | Omit | None = None
    supports_message_name: Config[bool] | Omit | None = None
    supported_roles: ClassVar[tuple[str, ...]] = ("system", "user", "assistant", "tool")

    def __post_init__(self) -> None:
        if not isinstance(self.max_context_window, int) or isinstance(
            self.max_context_window, bool
        ):
            raise ConfigurationError("max_context_window must be an integer")
