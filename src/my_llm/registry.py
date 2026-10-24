"""Protocol registration shared by configuration and conversion routing."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from my_llm._optional import require_sdk
from my_llm.exceptions import ConfigurationError, UnsupportedProviderError

if TYPE_CHECKING:
    from my_llm.codec.base.codec import BaseCodec

_CODECS: dict[str, Callable[[], BaseCodec]] = {}


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
