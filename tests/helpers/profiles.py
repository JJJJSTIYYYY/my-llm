"""Build explicit profile declarations from each registered protocol's defaults."""

from dataclasses import replace
from typing import Any

from my_llm import Config, Parameter
from my_llm.registry import codec_for

_MISSING = object()


def config(value: Any) -> Config[Any]:
    return Config(default_value=value)


def parameter(
    name: str,
    value: Any = _MISSING,
    *,
    protocol: str = "openai_chat",
    path: str | None = None,
    supports: bool | None = None,
) -> Parameter[Any]:
    declaration = getattr(codec_for(protocol).defaults, name)
    return replace(
        declaration,
        **({"default_value": value} if value is not _MISSING else {}),
        **({"path": path} if path is not None else {}),
        **({"supports": supports} if supports is not None else {}),
    )
