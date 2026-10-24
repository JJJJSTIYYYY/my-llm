"""Shared profile member enumeration and cached declaration analysis."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from types import UnionType
from typing import Any, TypeAliasType, Union, get_args, get_origin, get_type_hints

from my_llm.basic.exceptions import ConfigurationError
from my_llm.profile.types import Attribute, Config, Omit, Parameter


@dataclass(frozen=True, kw_only=True)
class Profile:
    """Enumerate declared members, including inherited, unset and omitted fields.

    Each method returns a new dictionary in dataclass field order, with the
    original members as values. No profile layers are resolved or copied.
    """

    def get_attributes(self) -> dict[str, Attribute[Any] | Omit | None]:
        """Return every field declared as Attribute, keyed by field name."""
        return self._members("attribute")

    def get_configs(self) -> dict[str, Config[Any] | Omit | None]:
        """Return every field declared as Config, keyed by field name."""
        return self._members("config")

    def get_parameters(self) -> dict[str, Parameter[Any] | Omit | None]:
        """Return every Parameter field, including unsupported bindings."""
        return self._members("parameter")

    def _members(self, kind: str) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name, declared_kind, _, _, _ in _declarations(type(self))
            if declared_kind == kind
        }


@lru_cache(maxsize=128)
def _declarations(profile_type: type) -> tuple[tuple[str, str, Any, str | None, str], ...]:
    """Cache type analysis, bounded for applications constructing dynamic types."""
    annotations = get_type_hints(profile_type, include_extras=True)
    result = []
    for item in fields(profile_type):
        annotation = annotations[item.name]
        kinds = _kinds(annotation)
        if len(kinds) != 1:
            raise ConfigurationError(
                f"{item.name} must declare exactly one of Parameter, Config or Attribute"
            )
        kind = kinds.pop()
        source = item.metadata.get("source")
        scope = item.metadata.get("scope", "request")
        if scope not in ("client", "request"):
            raise ConfigurationError("parameter scope must be client or request")
        if kind == "parameter":
            if source not in ("input", "output"):
                raise ConfigurationError(f"{item.name} must declare input/output source in Profile")
            if scope == "client" and source != "input":
                raise ConfigurationError(f"{item.name} must use the input source")
        result.append((item.name, kind, annotation, source, scope))
    return tuple(result)


def _kinds(annotation: Any) -> set[str]:
    origin = get_origin(annotation)
    markers = (Parameter, Config, Attribute)
    if origin in markers or annotation in markers:
        kind = {Parameter: "parameter", Config: "config", Attribute: "attribute"}[
            origin or annotation
        ]
        arguments = get_args(annotation)
        if arguments and _kinds(arguments[0]):
            return {"parameter", "config", "attribute"}
        return {kind}
    if isinstance(annotation, TypeAliasType):
        return _kinds(annotation.__value__)
    if origin in (Union, UnionType):
        return set().union(*(_kinds(argument) for argument in get_args(annotation)))
    return set()
