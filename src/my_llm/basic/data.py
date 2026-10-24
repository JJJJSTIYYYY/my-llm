"""Owned application data and lazy read-only views of SDK models."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from enum import Enum
from typing import Any

from pydantic import BaseModel

from my_llm.basic.exceptions import MessageValidationError

_MISSING = object()
_UNREAD = object()
_IMMUTABLE = frozenset((str, int, float, bool, bytes, type(None)))


def capture(value: Any, memo: dict | None = None) -> Any:
    """Own translated JSON, retaining deepcopy semantics for non-JSON objects.

    Protocol structures have already been projected. Immutable JSON leaves
    need no memo lookup or generic copy dispatch; custom objects still use
    their deepcopy implementation with the same memo.
    """
    kind = type(value)
    if kind in _IMMUTABLE:
        return value
    if kind not in (dict, list):
        return deepcopy(value, memo)
    if memo is None:
        memo = {}
    previous = memo.get(id(value), _MISSING)
    if previous is not _MISSING:
        return previous
    if kind is dict:
        result = {}
        memo[id(value)] = result
        for key, child in value.items():
            result[capture(key, memo)] = capture(child, memo)
    else:
        result = []
        memo[id(value)] = result
        result.extend(capture(child, memo) for child in value)
    return result


class _ModelView(dict):
    """Read SDK fields on demand, without serializing an entire response."""

    __slots__ = ("model", "_raw", "_values")

    def __init__(self, model: BaseModel):
        self.model = model
        self._raw = {}
        self._values = {}

    @property
    def origin_value(self):
        """The original SDK object, without enumerating unsupported fields."""
        return self.model

    def _field(self, key):
        value = self._raw.get(key, _UNREAD)
        if value is _UNREAD:
            value = getattr(self.model, key, _MISSING)
            # Explicit None is present; an unset Optional default is absent.
            if value is None and key not in self.model.model_fields_set:
                value = _MISSING
            self._raw[key] = value
        return value

    def __contains__(self, key):
        return self._field(key) is not _MISSING

    def __getitem__(self, key):
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def get(self, key, default=None):
        value = self._values.get(key, _UNREAD)
        if value is _UNREAD:
            raw = self._field(key)
            value = _MISSING if raw is _MISSING else _view(raw)
            self._values[key] = value
        return default if value is _MISSING else value

    def __iter__(self):
        return (key for key in type(self.model).model_fields if key in self)

    def __len__(self):
        return sum(1 for _ in self)

    def keys(self):
        return dict.fromkeys(self).keys()

    def items(self):
        return ((key, self[key]) for key in self)

    def copy(self):
        return dict(self)

    def __deepcopy__(self, memo):
        return {key: deepcopy(self[key], memo) for key in self}


def _view(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _ModelView(value)
    if isinstance(value, list):
        return [_view(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def read(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return dict(_ModelView(value))
    return value.copy() if isinstance(value, (dict, list)) else value


def dictionary_views(values: Any) -> Iterator[dict[str, Any]]:
    """Borrow dictionaries or lazily read only requested public SDK fields."""
    if not isinstance(values, (list, tuple)):
        raise MessageValidationError("expected a list of protocol dictionaries or SDK objects")
    for value in values:
        value = _view(value)
        if not isinstance(value, dict):
            raise MessageValidationError("expected protocol dictionaries or SDK objects")
        yield value


def dictionaries(values: Any) -> list[dict[str, Any]]:
    """Borrow protocol structures; their recognized output owns copied payloads."""
    return list(dictionary_views(values))
