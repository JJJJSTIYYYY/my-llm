"""Instance-keyed translated histories and consecutive input-batch deduplication."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any

from pydantic import BaseModel

from my_llm.basic.data import capture, plain
from my_llm.basic.exceptions import ConfigurationError


def _fingerprint_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _fingerprint_value(value.value)
    if isinstance(value, BaseModel):
        value = plain(value)
    if isinstance(value, Mapping):
        # Middleware mappings expose wire fields, without provenance. SDK model
        # views expose public fields without process-specific object identities.
        return (
            "mapping",
            sorted((json.dumps(key), _fingerprint_value(item)) for key, item in value.items()),
        )
    if isinstance(value, (list, tuple)):
        return ("sequence", [_fingerprint_value(item) for item in value])
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    return value


@dataclass(slots=True)
class _History:
    messages: list
    fingerprint: bytes | None


class MessageCache:
    """Append translated message lists independently of SDK request structure.

    History captures translation results before the semantic-loss adaptor runs.
    Hooks own their changes and any explicit cache maintenance those changes need.
    Cached history is never parsed, translated or association-validated again.
    Only the last appended input fingerprint is retained, so A/B/A stays intact.
    """

    def __init__(self) -> None:
        self._histories: dict[str, _History] = {}

    def fingerprint(self, messages: Any) -> bytes | None:
        """Hash caller messages independently of generated IDs and request options.

        Unsupported opaque extras remain subject to the existing conversion rules;
        they disable deduplication for this batch rather than reject valid content.
        """
        try:
            encoded = json.dumps(_fingerprint_value(messages), separators=(",", ":"))
        except (TypeError, ValueError, RecursionError):
            return None
        return sha256(encoded.encode("utf-8")).digest()

    def __contains__(self, key: str) -> bool:
        return key in self._histories

    def batch(self, key: str, *, fingerprint: bytes | None = None) -> MessageCacheBatch:
        """Stage a translated batch for request assembly and later publication."""
        return MessageCacheBatch(self, key, fingerprint)

    def _extend(self, key: str, messages: list, fingerprint: bytes | None) -> _History:
        if not isinstance(messages, list):
            raise ConfigurationError("message cache requires a translated message list")
        previous = self._histories.get(key)
        if fingerprint is not None and previous is not None and previous.fingerprint == fingerprint:
            return previous
        history = previous.messages if previous is not None else []
        return _History([*history, *capture(messages)], fingerprint)

    def update(
        self,
        key: str,
        messages: list,
        *,
        fingerprint: bytes | None = None,
    ) -> None:
        """Append an owned translated list unless it repeats the last input batch."""
        self._histories[key] = self._extend(key, messages, fingerprint)

    def get(self, key: str) -> list:
        """Return an independent translated history snapshot."""
        history = self._histories.get(key)
        return capture(history.messages) if history is not None else []

    def clear(self) -> None:
        """Release all histories owned by this cache."""
        self._histories.clear()

    def expire(self, key: str) -> None:
        """Release one instance's translated history; repeated calls are harmless."""
        self._histories.pop(key, None)


class MessageCacheBatch:
    """Request-local translated history, published before semantic-loss handling.

    Request assembly can still fail validation. Staging keeps those failures out
    of history without coupling the cache to transports or stream state.
    """

    def __init__(self, cache: MessageCache, key: str, fingerprint: bytes | None) -> None:
        self._cache = cache
        self._key = key
        self._fingerprint = fingerprint
        self._history: _History | None = None

    def append(self, messages: list) -> list:
        """Combine this translated batch with history and return an owned snapshot."""
        self._history = self._cache._extend(self._key, messages, self._fingerprint)
        return capture(self._history.messages)

    def commit(self) -> None:
        """Publish the staged translation without retaining adaptor/hook edits."""
        if self._history is not None:
            self._cache._histories[self._key] = self._history
