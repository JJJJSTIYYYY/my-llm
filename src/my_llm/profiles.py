"""Protocol defaults and explicit provider/model configuration.

None inherits a value, while OpenAI's omit disables it. Dictionary layering
overrides individual keys (not whole dictionaries); omit removes a key.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal

from openai import Omit

from my_llm.exceptions import ConfigurationError, UnsupportedProviderError

type APIType = Literal["responses", "chat_completions"]
type Protocol = Literal["openai", "anthropic", "gemini", "ollama"]
type ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]
type Parameter = dict[str, Any] | Omit | None

EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, kw_only=True)
class _Overrides:
    extra_headers: Parameter = None
    extra_body: Parameter = None
    reasoning_enable_parameter: Parameter = None
    reasoning_effort_parameter: Parameter = None
    reasoning_effort_mapping: Parameter = None
    supports_reasoning_effort: bool | Omit | None = None
    content_path: str | Omit | None = None
    reasoning_content_path: str | Omit | None = None
    replay_reasoning_content_when_tool_calls: bool | Omit | None = None


@dataclass(frozen=True, kw_only=True)
class ProviderProfile(_Overrides):
    """Provider connection, API selection and inherited model defaults.

    All arguments are keyword-only. None inherits a protocol default; omit
    disables a setting. Dictionary settings override keys rather than replacing
    the whole dictionary. Profiles are frozen; use dataclasses.replace() to
    build a changed configuration. The model catalogue has explicit mutators.

    Args:
        api_type: "chat_completions" (default) or "responses".
        protocol: "openai" (default); other protocol names are reserved.
        base_url: Optional service URL passed to AsyncOpenAI.
        api_key: Optional credential passed to AsyncOpenAI; excluded from repr.
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
        replay_reasoning_content_when_tool_calls: Replay visible reasoning with
            Chat tool-call history; defaults True.

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
        replay_reasoning_content_when_tool_calls: Override whether visible
            reasoning accompanies Chat tool-call history.

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


def _models_by_id(models: list[ModelProfile]) -> dict[str, ModelProfile]:
    if not isinstance(models, list) or any(not isinstance(model, ModelProfile) for model in models):
        raise ConfigurationError("models must be a list of ModelProfile instances")
    if any(not isinstance(model.model, str) or not model.model.strip() for model in models):
        raise ConfigurationError("model must be a non-empty string")
    return {model.model: model for model in models}


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


@dataclass(frozen=True)
class ResolvedProfile:
    """Validated configuration snapshot for advanced codec integrations.

    Applications normally pass ProviderProfile/ModelProfile to MyLLM instead.
    Treat values and their nested payloads as read-only.
    """

    api_type: APIType
    protocol: Protocol
    model: str
    values: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        """Read a resolved field, using default for missing or omitted values."""
        value = self.values.get(key, default)
        return default if isinstance(value, Omit) else value


def resolve_profiles(provider: ProviderProfile, model: ModelProfile) -> ResolvedProfile:
    """Validate and snapshot protocol, provider and model settings in order.

    Excludes descriptive context-window metadata and the provider's catalogue.
    Raises ConfigurationError subclasses for invalid settings/capabilities.
    """
    if not isinstance(provider, ProviderProfile) or not isinstance(model, ModelProfile):
        raise ConfigurationError("provider and model must be profile instances")
    if provider.protocol != "openai":
        raise UnsupportedProviderError(f"protocol {provider.protocol!r} is reserved")
    if provider.api_type not in ("responses", "chat_completions"):
        raise UnsupportedProviderError(f"unknown API type: {provider.api_type!r}")
    if not isinstance(model.model, str) or not model.model.strip():
        raise ConfigurationError("model must be a non-empty string")
    responses = provider.api_type == "responses"
    values: dict[str, Any] = {
        "extra_headers": {},
        "extra_body": {},
        "max_tokens": None,
        "supports_stream": True,
        "supports_reasoning": True,
        "supports_function_calling": False,
        "supports_message_name": True,
        "supports_reasoning_effort": responses,
        "replay_reasoning_content_when_tool_calls": True,
        "usage_mode": "none",
        "content_path": "response.output_text.delta" if responses else "$.choices[0].delta.content",
        "reasoning_content_path": "response.reasoning_summary_text.delta"
        if responses
        else "$.choices[0].delta.reasoning_content",
        "reasoning_effort_mapping": {effort: effort for effort in EFFORTS},
        "reasoning_enable_parameter": {"reasoning": {"summary": "detailed", "effort": "medium"}}
        if responses
        else {"extra_body": {"thinking": {"type": "enabled"}}},
        "reasoning_effort_parameter": {"reasoning": {"summary": "detailed", "effort": "{effort}"}}
        if responses
        else {"reasoning_effort": "{effort}"},
    }
    for profile in (provider, model):
        for item in fields(profile):
            if item.name in ("api_type", "protocol", "model", "models", "max_context_window"):
                continue
            value = getattr(profile, item.name)
            if value is None:
                continue
            if isinstance(value, dict):
                values[item.name] = merge_dicts(values.get(item.name), value)
            else:
                values[item.name] = value
    # Snapshot configuration once, after all layers have been merged.
    resolved = ResolvedProfile(provider.api_type, provider.protocol, model.model, deepcopy(values))
    for key in (
        "supports_stream",
        "supports_reasoning",
        "supports_function_calling",
        "supports_message_name",
        "supports_reasoning_effort",
        "replay_reasoning_content_when_tool_calls",
    ):
        value = resolved.get(key, False)
        if not isinstance(value, bool):
            raise ConfigurationError(f"{key} must be a bool or omit")
    if not resolved.get("supports_stream", False):
        raise UnsupportedProviderError("provider must support upstream streaming")
    usage = resolved.get("usage_mode", "none")
    if usage not in ("none", "native", "stream_options"):
        raise ConfigurationError("invalid usage_mode")
    if (responses and usage == "stream_options") or (not responses and usage == "native"):
        raise ConfigurationError("usage_mode is incompatible with the selected API")
    for key in (
        "extra_headers",
        "extra_body",
        "reasoning_enable_parameter",
        "reasoning_effort_parameter",
        "reasoning_effort_mapping",
    ):
        value = values[key]
        if not isinstance(value, (dict, Omit)):
            raise ConfigurationError(f"{key} must be a dict or omit")
    for key in ("content_path", "reasoning_content_path"):
        value = values[key]
        if not isinstance(value, (str, Omit)) or (isinstance(value, str) and not value):
            raise ConfigurationError(f"{key} must be a non-empty string or omit")
    validate_max_tokens(resolved.get("max_tokens"))
    client_parameters(resolved)
    return resolved


def validate_max_tokens(value: Any) -> None:
    """Require a positive integer or None; raise ConfigurationError otherwise."""
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ConfigurationError("max_tokens must be a positive integer or omit")


def client_parameters(profile: ResolvedProfile, **overrides: Any) -> dict[str, Any]:
    """Validate SDK connection fields, applying non-None per-call overrides.

    Returns base_url/api_key/timeout/max_retries keyword arguments, excluding
    unset/omitted values. Invalid values raise ConfigurationError.
    """
    result = {}
    for key in ("base_url", "api_key", "timeout", "max_retries"):
        value = overrides.get(key)
        if value is None:
            value = profile.get(key)
        if value is None or isinstance(value, Omit):
            continue
        if key in ("api_key", "base_url") and (not isinstance(value, str) or not value.strip()):
            raise ConfigurationError(f"{key} must be a non-empty string")
        if key == "timeout" and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
        ):
            raise ConfigurationError("timeout must be positive")
        if key == "max_retries" and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ConfigurationError("max_retries must be a non-negative integer")
        result[key] = value
    return result
