"""Provider connection settings and explicit model catalogue management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from openai import Omit

from my_llm.profile.model_profile import ModelProfile
from my_llm.profile.utils.common import APIType, Protocol, _Overrides
from my_llm.types.exceptions import ConfigurationError


@dataclass(frozen=True, kw_only=True)
class ProviderProfile(_Overrides):
    """Provider connection, API selection and inherited model defaults.

    All arguments are keyword-only. None inherits a protocol default; omit
    disables a setting. Dictionary settings override keys rather than replacing
    the whole dictionary. Profiles are frozen; use dataclasses.replace() to
    build a changed configuration. The model catalogue has explicit mutators.

    Args:
        api_type: "chat_completions" (default) or "responses".
        protocol: "openai" (default), "anthropic", "gemini" or "ollama".
        base_url: Optional service URL passed to the selected official SDK.
        api_key: Optional SDK credential; excluded from repr.
        timeout: Positive SDK timeout in seconds, or None for the SDK default.
        max_retries: Non-negative SDK retry count, or None for the SDK default.
        supports_stream: Whether upstream streaming is available; defaults True
            when resolved. False/omit is unsupported in both public modes.
        usage_mode: "none" (default), "native" for Responses, or
            "stream_options" for Chat Completions.
        models: Optional list of ModelProfile configurations. Model IDs are
            unique; later entries replace earlier entries with the same ID.
            Generations still require explicit model selection.
        extra_headers: Default request headers, merged by key.
        extra_body: Default extra request-body fields, merged by key.
        reasoning_enable_parameter: SDK keyword dictionary used to enable
            reasoning when effort levels are unsupported.
        reasoning_effort_parameter: SDK keyword dictionary used when effort
            levels are supported. "{effort}" expands to the mapped value.
        reasoning_effort_mapping: Map none/low/medium/high/xhigh/max to provider
            strings; None or omit values disable individual levels.
        supports_reasoning_effort: Defaults True for Responses, False for Chat.
        content_path: Text delta selector; defaults to the API's standard field.
        reasoning_content_path: Reasoning delta selector; defaults to the API's
            standard reasoning-summary/reasoning-content field.
        replay_reasoning_content_when_tool_calls: Include visible reasoning on
            every Chat/Ollama assistant message, using an empty string when
            absent, including messages without tool calls; defaults True.

    Raises:
        ConfigurationError: The model catalogue contains invalid entries.
            Other settings are validated when constructing MyLLM or generating.
    """

    api_type: APIType = "chat_completions"
    protocol: Protocol = "openai"
    base_url: str | Omit | None = None
    api_key: str | Omit | None = field(default=None, repr=False)
    timeout: float | Omit | None = None
    max_retries: int | Omit | None = None
    supports_stream: bool | Omit | None = None
    usage_mode: Literal["none", "native", "stream_options"] | Omit | None = None
    models: list[ModelProfile] | None = None

    def __post_init__(self) -> None:
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
