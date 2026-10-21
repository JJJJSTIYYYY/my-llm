"""Provider connection settings and explicit model catalogue management."""

from __future__ import annotations

from dataclasses import dataclass, field

from my_llm.exceptions import ConfigurationError
from my_llm.profile.configuration import OverridableProfile, Protocol
from my_llm.profile.model_profile import ModelProfile
from my_llm.profile.types import Config, Omit, Parameter


@dataclass(frozen=True, kw_only=True)
class ProviderProfile(OverridableProfile):
    """Provider connection, API selection and explicit model catalogue.

    None inherits; omit disables a setting. Parameter and Config wrappers retain
    explicit default_value=None. Config dictionaries merge by key. Extend with
    typed Parameter or Config fields. api_key binds a connection parameter;
    other SDK connection values use Config fields.
    """

    protocol: Protocol | None = None
    base_url: Config[str] | Omit | None = None
    api_key: Parameter[str] | Omit | None = field(
        default=None, repr=False, metadata={"scope": "client", "source": "input"}
    )
    timeout: Config[float] | Omit | None = None
    max_retries: Config[int] | Omit | None = None
    supports_stream: Config[bool] | Omit | None = None
    usage_mode: Config[str] | Omit | None = None
    models: list[ModelProfile] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.models is not None:
            object.__setattr__(self, "models", list(_models_by_id(self.models).values()))

    def add_models(self, model_profile: list[ModelProfile] | ModelProfile) -> None:
        """Add one ModelProfile or a list, replacing matching IDs in place.

        New IDs are appended; existing IDs retain their position. Later duplicate
        IDs win. Invalid input raises ConfigurationError without changing the
        catalogue. ModelProfile instances are stored without cloning.
        """
        incoming = [model_profile] if isinstance(model_profile, ModelProfile) else model_profile
        updates = _models_by_id(incoming)
        if updates:
            models = {model.model: model for model in self.models or []}
            models.update(updates)
            object.__setattr__(self, "models", list(models.values()))

    def list_models(self) -> list[ModelProfile]:
        """Return a new list of models in catalogue order, or [] when unset.

        ModelProfile instances are reused. Changing the returned list does not
        add or remove entries in this provider's catalogue.
        """
        return list(self.models or [])

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
                self, "models", [model for model in self.models if model.model not in removed]
            )


def _models_by_id(models: list[ModelProfile]) -> dict[str, ModelProfile]:
    if not isinstance(models, list) or any(not isinstance(model, ModelProfile) for model in models):
        raise ConfigurationError("models must be a list of ModelProfile instances")
    if any(not isinstance(model.model, str) or not model.model.strip() for model in models):
        raise ConfigurationError("model must be a non-empty string")
    return {model.model: model for model in models}
