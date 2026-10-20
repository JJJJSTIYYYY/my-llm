"""Declarative parameters, code configuration and SDK-independent omission."""

from __future__ import annotations

from dataclasses import dataclass
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
    """Bind a protocol parameter's location, missing-value default and phase.

    encode reads an API response into intermediate state; decode writes a request
    from intermediate state. name and comment are presentation metadata only.
    A present value, including None, always takes precedence over default_value.
    """

    name: str
    path: str
    default_value: Omit | None | _T
    phase: Literal["encode", "decode"]
    comment: str | None = None

    def __post_init__(self) -> None:
        _validate_metadata(self.name, self.comment)
        if self.phase not in ("encode", "decode"):
            raise ConfigurationError("parameter phase must be 'encode' or 'decode'")
        path_keys(self.path)


@dataclass(frozen=True, kw_only=True)
class Config(Generic[_T]):
    """Declare a code setting; name and comment are presentation metadata only."""

    name: str
    default_value: Omit | None | _T
    comment: str | None = None

    def __post_init__(self) -> None:
        _validate_metadata(self.name, self.comment)


def _validate_metadata(name: str, comment: str | None) -> None:
    if not isinstance(name, str):
        raise ConfigurationError("configuration name must be a string")
    if comment is not None and not isinstance(comment, str):
        raise ConfigurationError("configuration comment must be a string or None")
