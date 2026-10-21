"""Declarative parameters, code configuration and SDK-independent omission."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, fields
from functools import lru_cache, wraps
from typing import Any, Generic, Literal, TypeVar

from my_llm.exceptions import ConfigurationError
from my_llm.profile.paths import path_keys

_T = TypeVar("_T")


class Omit:
    """Explicitly omit a configured parameter; every instance is the same marker."""

    __slots__ = ()
    _instance: Omit | None = None

    def __new__(cls) -> Omit:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "omit"

    def __bool__(self) -> bool:
        return False

    def __copy__(self) -> Omit:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Omit:
        return self

    def __reduce__(self) -> tuple[type[Omit], tuple[()]]:
        return Omit, ()


omit = Omit()


@dataclass(frozen=True, kw_only=True)
class Parameter(Generic[_T]):
    """A parameter's route, support and default; Profile fixes its identity.

    source="input" means a user/API request parameter; source="output" means
    an API response parameter. Unsupported parameters always default to omit.
    """

    path: str
    default_value: Omit | None | _T = omit
    supports: bool = True
    comment: str | None = None
    name: str | None = field(default=None, init=False)
    source: Literal["input", "output"] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _validate_comment(self.comment)
        if not isinstance(self.supports, bool):
            raise ConfigurationError("parameter supports must be a bool")
        if not self.supports:
            object.__setattr__(self, "default_value", omit)
        path_keys(self.path)


@dataclass(frozen=True, kw_only=True)
class Config(Generic[_T]):
    """A code setting whose name is fixed by its Profile field."""

    default_value: Omit | None | _T
    comment: str | None = None
    name: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _validate_comment(self.comment)


def set_attr(method):
    """Bind wrapper identity once, from hard-coded Profile declarations.

    Decorate a Profile's __post_init__. Parameter fields declare source through
    dataclass metadata. Copy each unbound wrapper so reusable declarations never
    acquire an identity as a side effect of constructing another profile.
    """

    @wraps(method)
    def bind(self, *args, **kwargs):
        for name, source in _attributes(type(self)):
            value = getattr(self, name)
            if not isinstance(value, (Parameter, Config)):
                continue
            if value.name is not None or isinstance(value, Parameter) and value.source is not None:
                if value.name != name or isinstance(value, Parameter) and value.source != source:
                    raise ConfigurationError(f"{name} name/source have already been assigned")
                continue
            bound = copy(value)
            object.__setattr__(bound, "name", name)
            if isinstance(bound, Parameter):
                if source not in ("input", "output"):
                    raise ConfigurationError(f"{name} must declare input/output source in Profile")
                object.__setattr__(bound, "source", source)
            object.__setattr__(self, name, bound)
        return method(self, *args, **kwargs)

    return bind


@lru_cache(maxsize=128)
def _attributes(profile_type: type) -> tuple[tuple[str, str | None], ...]:
    return tuple((item.name, item.metadata.get("source")) for item in fields(profile_type))


def _validate_comment(comment: str | None) -> None:
    if comment is not None and not isinstance(comment, str):
        raise ConfigurationError("configuration comment must be a string or None")
