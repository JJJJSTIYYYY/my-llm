"""Instance-keyed translated histories, independent of conversion and transport."""

from typing import Any

from my_llm.basic.data import _MISSING, capture
from my_llm.basic.exceptions import ConfigurationError

type MessagePath = tuple[str, ...]


def _value(request: dict, path: MessagePath) -> Any:
    value = request
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _append(previous: Any, current: Any) -> Any:
    if isinstance(previous, list) and isinstance(current, list):
        return [*previous, *current]
    if isinstance(previous, dict) and isinstance(current, dict):
        return {
            **previous,
            **{key: _append(previous.get(key, _MISSING), value) for key, value in current.items()},
        }
    # A hook may explicitly replace an instruction with a scalar or a new shape.
    return current


class MessageCache:
    """Append translated message fields; keep all other request options call-local.

    Protocol owners declare message/instruction paths. Updates capture only these
    fields after semantic-loss handling; reads return independently owned snapshots.
    Cached history is never parsed, translated or association-validated again.
    """

    def __init__(self) -> None:
        self._histories: dict[str, dict[MessagePath, Any]] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._histories

    def has_messages(self, key: str, path: MessagePath) -> bool:
        """Return whether this key already contains a nonempty native history."""
        return bool(self._histories.get(key, {}).get(path))

    def update(self, key: str, request: dict, *, paths: tuple[MessagePath, ...]) -> None:
        """Atomically append an owned translated batch at the declared paths.

        The first path is the native message list; remaining paths hold optional
        instruction fields. Lists append, dictionary envelopes merge recursively,
        and explicit scalar replacements take precedence.
        """
        if not paths or not isinstance(_value(request, paths[0]), list):
            raise ConfigurationError("cached request requires a native message list")
        history = dict(self._histories.get(key, {}))
        for path in paths:
            value = _value(request, path)
            if value is not _MISSING:
                history[path] = _append(history.get(path, _MISSING), capture(value))
        self._histories[key] = history

    def get(self, key: str, request: dict) -> dict:
        """Return latest history in this call's envelope without sharing cache data."""
        # The request envelope is already owned by this call. Copy only containers
        # receiving cached fields, then capture cached leaves for the SDK.
        result = dict(request)
        for path, value in self._histories.get(key, {}).items():
            parent = result
            for name in path[:-1]:
                child = parent.get(name)
                parent[name] = dict(child) if isinstance(child, dict) else {}
                parent = parent[name]
            parent[path[-1]] = capture(value)
        return result

    def clear(self) -> None:
        """Release all histories owned by this cache."""
        self._histories.clear()

    def expire(self, key: str) -> None:
        """Release one instance's translated history; repeated calls are harmless."""
        self._histories.pop(key, None)
