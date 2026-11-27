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
        """Bind an SDK model and initialize lazy raw-field and projected-value caches."""
        self.model = model
        self._raw = {}
        self._values = {}

    @property
    def origin_value(self):
        """The original SDK object, without enumerating unsupported fields."""
        return self.model

    def _field(self, key):
        """Read and cache a public field, distinguishing explicit None from an unset default."""
        value = self._raw.get(key, _UNREAD)
        if value is _UNREAD:
            value = getattr(self.model, key, _MISSING)
            # Explicit None is present; an unset Optional default is absent.
            if value is None and key not in self.model.model_fields_set:
                value = _MISSING
            self._raw[key] = value
        return value

    def __contains__(self, key):
        """Whether the SDK model supplies this field."""
        return self._field(key) is not _MISSING

    def __getitem__(self, key):
        """Read a projected field or raise KeyError when absent."""
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def get(self, key, default=None):
        """Lazily project one requested field without serializing unrelated SDK fields."""
        value = self._values.get(key, _UNREAD)
        if value is _UNREAD:
            raw = self._field(key)
            value = _MISSING if raw is _MISSING else _view(raw)
            self._values[key] = value
        return default if value is _MISSING else value

    def __iter__(self):
        """Iterate declared fields that are actually present in the SDK model."""
        return (key for key in type(self.model).model_fields if key in self)

    def __len__(self):
        """Count present declared model fields."""
        return sum(1 for _ in self)

    def keys(self):
        """Expose dictionary-style keys for present model fields."""
        return dict.fromkeys(self).keys()

    def items(self):
        """Lazily iterate present fields and their projected values."""
        return ((key, self[key]) for key in self)

    def copy(self):
        """Build a shallow dictionary projection of present declared fields."""
        return dict(self)

    def __deepcopy__(self, memo):
        """Copy present projected model fields with the caller's deepcopy memo."""
        return {key: deepcopy(self[key], memo) for key in self}


def _view(value: Any) -> Any:
    """Wrap SDK models lazily and normalize list elements and enum values."""
    if isinstance(value, BaseModel):
        return _ModelView(value)
    if isinstance(value, list):
        return [_view(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def read(value: Any, key: str, default: Any = None) -> Any:
    """Read one dictionary key or SDK attribute with a shared default."""
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def plain(value: Any) -> Any:
    """Project one SDK model or shallow-copy an ordinary container."""
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
