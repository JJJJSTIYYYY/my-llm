"""Profile attributes, parameters, configuration and SDK-independent omission."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, fields
from functools import wraps
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


@dataclass(kw_only=True)
class Attribute(Generic[_T]):
    """Descriptive or identity data owned by a profile, without parameter routing.

    Args:
        default_value: Required value of the declared type, such as a model ID.
        comment: Optional description for display.

    Read the data through default_value. Attributes do not participate in
    configuration inheritance. Instances are mutable keyword-only dataclasses.
    """

    default_value: _T
    comment: str | None = None

    def __post_init__(self) -> None:
        _validate_comment(self.comment)


@dataclass(kw_only=True)
class Parameter(Generic[_T]):
    """Common path, default value and capability support for routed parameters.

    Args:
        path: Rooted field path, such as ``$.max_tokens`` or ``$.choices[0].delta.content``.
        default_value: Value used when the target field is absent; omit adds no
            default and None is an explicit value.
        supports: Whether the capability is supported. False forces the
            default_value to omit; supplied values become decoder semantic losses.
        comment: Optional description for display.

    Attributes:
        name: Field name assigned by the containing profile.
        source: Direction fixed by InputParameter or OutputParameter.

    Use InputParameter for requests and OutputParameter for responses. The base
    class alone does not declare a direction. Callers cannot pass name/source.
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


@dataclass(kw_only=True)
class InputParameter(Parameter[_T]):
    """Route a request or connection value to its native field path."""

    source: Literal["input"] = field(default="input", init=False)


@dataclass(kw_only=True)
class OutputParameter(Parameter[_T]):
    """Read a response value from a native event.

    event_type selects the exact native event ``type`` (or Chat ``object``).
    An empty string retains the codec's native event classification. Event
    selection applies to streaming chunks; history and snapshots use only path.
    """

    source: Literal["output"] = field(default="output", init=False)
    event_type: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.event_type, str):
            raise ConfigurationError("output parameter event_type must be a string")


@dataclass(kw_only=True)
class Config(Generic[_T]):
    """Declare a behavior setting or a group of native options.

    Args:
        default_value: Required setting value, including explicit None or omit.
            Dictionary values merge by key when profiles are resolved.
        comment: Optional description for display.

    Attributes:
        name: Field name assigned by the containing profile.

    Instances are mutable keyword-only dataclasses. Unlike Parameter, Config has
    no field path; supported setting names determine its meaning.
    """

    default_value: Omit | None | _T
    comment: str | None = None
    name: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _validate_comment(self.comment)


def set_attr(method):
    """Bind wrapper identity once, from hard-coded Profile declarations.

    Decorate a Profile's __post_init__. Parameter subclasses fix their source.
    Copy each unbound wrapper so reusable declarations never
    acquire an identity as a side effect of constructing another profile.
    """

    @wraps(method)
    def bind(self, *args, **kwargs):
        for item in fields(self):
            name = item.name
            value = getattr(self, name)
            if not isinstance(value, (Parameter, Config)):
                continue
            if value.name is not None:
                if value.name != name:
                    raise ConfigurationError(f"{name} name has already been assigned")
                continue
            bound = copy(value)
            object.__setattr__(bound, "name", name)
            object.__setattr__(self, name, bound)
        return method(self, *args, **kwargs)

    return bind


def _validate_comment(comment: str | None) -> None:
    if comment is not None and not isinstance(comment, str):
        raise ConfigurationError("configuration comment must be a string or None")
