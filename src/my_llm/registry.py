"""Protocol registration shared by configuration and conversion routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from my_llm._optional import require_sdk
from my_llm.exceptions import ConfigurationError, UnsupportedProviderError

if TYPE_CHECKING:
    from my_llm.codec.base.codec import BaseCodec

_CODECS: dict[str, BaseCodec] = {}


def register_codec(codec: BaseCodec) -> None:
    protocol = codec.protocol_or_api_style
    if protocol in _CODECS:
        raise ConfigurationError(f"protocol is already registered: {protocol!r}")
    _CODECS[protocol] = codec


def registered_codecs() -> dict[str, BaseCodec]:
    return _CODECS.copy()


def codec_for(protocol: str) -> BaseCodec:
    try:
        return _CODECS[protocol]
    except (KeyError, TypeError) as exc:
        require_sdk(protocol)
        raise UnsupportedProviderError(f"unsupported protocol: {protocol!r}") from exc
