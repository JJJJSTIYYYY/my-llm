"""Protocol registration, built-in factories and optional SDK discovery."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING

from my_llm.basic.exceptions import ConfigurationError, UnsupportedProviderError

if TYPE_CHECKING:
    from my_llm.codec.base.codec import BaseCodec

# protocol: (module, class, SDK import, installation extra)
BUILTIN_CODECS = {
    "openai_chat": ("openai.chat.codec", "ChatCompletionsCodec", "openai", "openai"),
    "openai_responses": ("openai.responses.codec", "ResponsesCodec", "openai", "openai"),
    "anthropic": ("anthropic.codec", "AnthropicCodec", "anthropic", "anthropic"),
    "gemini": ("gemini.codec", "GeminiCodec", "google.genai", "google-genai"),
    "ollama": ("ollama.codec", "OllamaCodec", "ollama", "ollama"),
}

_CODECS: dict[str, Callable[[], BaseCodec]] = {}


def sdk_available(protocol: str) -> bool:
    sdk = BUILTIN_CODECS[protocol][2]
    try:
        return find_spec(sdk) is not None
    except ModuleNotFoundError as exc:
        if exc.name not in {sdk, sdk.partition(".")[0]}:
            raise
        return False


def require_sdk(protocol: str) -> None:
    if isinstance(protocol, str) and protocol in BUILTIN_CODECS and not sdk_available(protocol):
        extra = BUILTIN_CODECS[protocol][3]
        raise ModuleNotFoundError(
            f"Protocol {protocol!r} requires an optional SDK; "
            f"install it with: pip install 'my-llm-py[{extra}]'",
            name=BUILTIN_CODECS[protocol][2],
        )


def builtin_codec(protocol: str):
    require_sdk(protocol)
    module, name, _, _ = BUILTIN_CODECS[protocol]
    return getattr(import_module(f"my_llm.codec.{module}"), name)


def __getattr__(name: str):
    for protocol, (_, class_name, _, _) in BUILTIN_CODECS.items():
        if name == class_name:
            return builtin_codec(protocol)
    raise AttributeError(f"module 'my_llm.codec' has no attribute {name!r}")


def register_codec(factory: Callable[[], BaseCodec], *, protocol: str) -> None:
    """Store a constructor already validated by MyCodec.register."""
    if protocol in _CODECS:
        raise ConfigurationError(f"protocol is already registered: {protocol!r}")
    _CODECS[protocol] = factory


def registered_codec_factories() -> dict[str, Callable[[], BaseCodec]]:
    """Return a registration snapshot containing constructors only."""
    return _CODECS.copy()


def codec_for(protocol: str) -> BaseCodec:
    try:
        factory = _CODECS[protocol]
    except (KeyError, TypeError) as exc:
        require_sdk(protocol)
        raise UnsupportedProviderError(f"unsupported protocol: {protocol!r}") from exc
    return factory()
