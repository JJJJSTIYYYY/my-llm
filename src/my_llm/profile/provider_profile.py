"""Provider connection settings and explicit model catalogue management."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from my_llm.basic.exceptions import ConfigurationError
from my_llm.profile.configuration import OverridableProfile, Protocol
from my_llm.profile.model_profile import ModelProfile
from my_llm.profile.types import Attribute, Config, Omit, Parameter


@dataclass(frozen=True, kw_only=True)
class ProviderProfile(OverridableProfile):
    """Provider connection, upstream protocol and optional model catalogue.

    Args:
        protocol: Config containing a registered protocol name; required for use
            with MyLLM or resolve_profiles, although its field default is None.
        base_url: Config containing the service URL; None inherits SDK defaults.
        api_key: Input Parameter containing the authentication value and path.
            Use the SDK key path or an authentication-header path for the protocol.
        timeout: Config containing a positive timeout in seconds.
        max_retries: Config containing a nonnegative retry count. Ollama accepts
            only None, omit or zero.
        supports_stream: Config capability flag; must remain enabled for generation.
        usage_mode: Config containing a protocol-supported usage mode.
        models: Optional Attribute containing a list of ModelProfile entries.
            It does not select a model automatically.

    This is a frozen keyword-only dataclass with explicit catalogue mutation
    methods. None inherits a setting; omit disables it. Parameter overrides
    replace a whole declaration; Config dictionaries merge by key.
    Additional keyword fields are inherited from OverridableProfile.
    """

    protocol: Config[Protocol] | None = None
    base_url: Config[str] | Omit | None = None
    api_key: Parameter[str] | Omit | None = field(
        default=None, repr=False, metadata={"scope": "client", "source": "input"}
    )
    timeout: Config[float] | Omit | None = None
    max_retries: Config[int] | Omit | None = None
    supports_stream: Config[bool] | Omit | None = None
    usage_mode: Config[str] | Omit | None = None
    models: Attribute[list[ModelProfile]] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.models is not None:
            if not isinstance(self.models, Attribute):
                raise ConfigurationError("models must use Attribute[list[ModelProfile]]")
            object.__setattr__(
                self,
                "models",
                replace(
                    self.models,
                    default_value=list(_models_by_id(self.models.default_value).values()),
                ),
            )

    def add_models(self, model_profile: list[ModelProfile] | ModelProfile) -> None:
        """Add or replace model catalogue entries by model ID.

        Args:
            model_profile: One ModelProfile or a list of ModelProfile instances.

        New IDs append; existing IDs retain their positions. Later duplicate IDs win.
        Stored profiles are reused, not cloned. This mutates the provider catalogue
        and returns None; existing MyLLM instances retain their configuration.
        Invalid entries raise ConfigurationError without changing the catalogue.
        """
        incoming = [model_profile] if isinstance(model_profile, ModelProfile) else model_profile
        updates = _models_by_id(incoming)
        if updates:
            models = {
                model.model.default_value: model
                for model in (self.models.default_value if self.models is not None else [])
            }
            models.update(updates)
            entries = list(models.values())
            catalogue = (
                Attribute(default_value=entries)
                if self.models is None
                else replace(self.models, default_value=entries)
            )
            object.__setattr__(self, "models", catalogue)

    def list_models(self) -> list[ModelProfile]:
        """Return a new list of models in catalogue order, or [] when unset.

        ModelProfile instances are reused. Changing the returned list does not
        add or remove entries in this provider's catalogue.
        """
        return list(self.models.default_value) if self.models is not None else []

    def delete_model(self, model_id: list[str]) -> None:
        """Delete IDs in model_id, ignoring IDs absent from the catalogue.

        Args:
            model_id: List of model ID strings, including an empty list.

        Raises:
            ConfigurationError: model_id is not a list of strings. The catalogue
                is unchanged when validation fails.
        """
        if not isinstance(model_id, list) or any(not isinstance(item, str) for item in model_id):
            raise ConfigurationError("model_id must be a list of strings")
        if self.models is not None and model_id:
            removed = set(model_id)
            object.__setattr__(
                self,
                "models",
                replace(
                    self.models,
                    default_value=[
                        model
                        for model in self.models.default_value
                        if model.model.default_value not in removed
                    ],
                ),
            )


def _models_by_id(models: list[ModelProfile]) -> dict[str, ModelProfile]:
    if not isinstance(models, list) or any(not isinstance(model, ModelProfile) for model in models):
        raise ConfigurationError("models must be a list of ModelProfile instances")
    if any(
        not isinstance(model.model, Attribute)
        or not isinstance(model.model.default_value, str)
        or not model.model.default_value.strip()
        for model in models
    ):
        raise ConfigurationError("model must be a non-empty string")
    return {model.model.default_value: model for model in models}
