"""Profile attributes, parameters, configuration and SDK-independent omission."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, fields
from functools import lru_cache, wraps
from typing import Any, Generic, Literal, TypeVar

from my_llm.basic.exceptions import ConfigurationError
from my_llm.profile.paths import path_keys

_T = TypeVar("_T")


class Omit:
    """Type of the public omit marker for explicit omission.

    Use ``omit`` to disable a profile setting or remove a supported request value.
    It differs from None, is false in boolean contexts, and retains its identity
    when copied. Calling Omit() returns the same marker.
    """

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
class Attribute(Generic[_T]):
    """Descriptive or identity data owned by a profile, without parameter routing.

    Args:
        default_value: Required value of the declared type, such as a model ID.
        comment: Optional description for display.

    Read the data through default_value. Attributes do not participate in
    configuration inheritance. Instances are frozen keyword-only dataclasses.
    """

    default_value: _T
    comment: str | None = None

    def __post_init__(self) -> None:
        _validate_comment(self.comment)


@dataclass(frozen=True, kw_only=True)
class Parameter(Generic[_T]):
    """Declare a parameter's path, default value and capability support.

    Args:
        path: Rooted field path, such as ``$.max_tokens`` or ``$.choices[0].delta.content``.
        default_value: Value used when the target field is absent; omit adds no
            default and None is an explicit value.
        supports: Whether automatic routing is supported. False forces the
            default_value to omit.
        comment: Optional description for display.

    Attributes:
        name: Read-only field name assigned by the containing profile.
        source: Read-only "input" or "output" assigned by the profile declaration.

    Input parameters configure requests; output parameters select response values.
    The containing profile determines the source; callers cannot pass name/source
    to this frozen keyword-only dataclass. Invalid paths, comments or support flags
    raise ConfigurationError.
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
    """Declare a behavior setting or a group of native options.

    Args:
        default_value: Required setting value, including explicit None or omit.
            Dictionary values merge by key when profiles are resolved.
        comment: Optional description for display.

    Attributes:
        name: Read-only field name assigned by the containing profile.

    Instances are frozen keyword-only dataclasses. Unlike Parameter, Config has
    no field path; supported setting names determine its meaning.
    """

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
