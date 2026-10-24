"""Explicit intermediate values and standard protocol conversion helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from enum import Enum
from typing import Any

from pydantic import BaseModel

from my_llm.exceptions import MessageValidationError, UnsupportedCapabilityError

FUNCTIONS = "functions"
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


def coalesce_messages(messages: list) -> list[dict]:
    """Combine consecutive Responses fragments from one assistant history turn."""
    result = []
    combined = False
    for message in messages:
        if (
            result
            and message.get("coalesce")
            and result[-1].get("coalesce")
            and message.get("role") == result[-1].get("role") == "assistant"
            and source_protocol(message) == source_protocol(result[-1])
        ):
            if not combined:
                result[-1] = {
                    **select(result[-1], "protocol", "coalesce", "role", "name", "id", "status"),
                    "content": list(result[-1]["content"]),
                }
                combined = True
            result[-1]["content"].extend(message["content"])
        else:
            result.append(message)
            combined = False
    return result


def blocks(content: Any) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(dictionary_views(content))


def uri_source(uri: str) -> dict:
    """Translate a remote or inline URI into an intermediate media source."""
    if not uri.startswith("data:"):
        return {"type": "url", "url": uri}
    header, _, data = uri.partition(",")
    return {
        "type": "base64",
        "media_type": header[5:].removesuffix(";base64"),
        "data": data,
    }


def text(content: list[dict]) -> str:
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


def arguments(value: Any) -> dict:
    """Read an argument object; the public translation boundary captures it."""
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise MessageValidationError("function arguments must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise MessageValidationError("function arguments must be a JSON object")
    return parsed


def json_arguments(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def source_protocol(value: dict) -> str | None:
    return value.get("protocol")


def allocate_call_id(used: set[str], preferred: str) -> str:
    """Choose a deterministic synthetic ID that cannot shadow an explicit call."""
    candidate = preferred
    suffix = 0
    while candidate in used:
        suffix += 1
        candidate = f"{preferred}_{suffix}"
    used.add(candidate)
    return candidate


def associate_tool_result(pending: list[dict], call_id: str | None, name: str | None) -> str | None:
    """Consume the matching pending call, preferring an explicit result ID."""
    match = next(
        (
            call
            for call in pending
            if (call["id"] == call_id if call_id else name is None or call.get("name") == name)
        ),
        None,
    )
    # An explicit unmatched ID must not fall back to a same-name call; history
    # validation will reject the unmatched result at the message boundary.
    if match is None:
        return call_id
    if name and name != match.get("name"):
        raise MessageValidationError("tool result name does not match its call ID")
    # Remove explicit matches too, or a later ID-less result could reuse them.
    pending.remove(match)
    return call_id or match["id"]


def mapping_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def match_all(values, predicate) -> bool:
    try:
        items = list(dictionary_views(values))
        return bool(items) and all(predicate(item) for item in items)
    except (MessageValidationError, TypeError, AttributeError, KeyError):
        return False


def select(value: dict, *fields: str) -> dict:
    """Borrow recognized values while building a fresh protocol structure.

    Message/tool exits capture the translated result. Stream hooks capture
    borrowed mutable leaves when constructing their independent events.
    """
    return {key: child for key in fields if (child := value.get(key, _MISSING)) is not _MISSING}


def select_present(value: dict, *fields: str) -> dict:
    return {
        key: child
        for key in fields
        if (child := value.get(key, _MISSING)) is not _MISSING and child is not None
    }


def project_message(message: dict, project, *, fields: tuple[str, ...] = ("role",)) -> dict:
    content = []
    for block in message["content"]:
        rendered = project(block)
        if rendered is not None:
            content.append(rendered)
    return {**select(message, *fields, "protocol"), "content": content}


def text_result(value) -> str:
    """Represent canonical text blocks and structured results as a string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(
        isinstance(block, dict) and block.get("type") == "text" for block in value
    ):
        return "".join(block.get("text", "") for block in value)
    return json.dumps(value, ensure_ascii=False)


def tool_result_content(value: Any, *, native_types: frozenset[str]) -> Any:
    """Serialize application data; let source codecs project native content lists."""
    if isinstance(value, dict) or (
        isinstance(value, list)
        and not any(isinstance(part, dict) and part.get("type") in native_types for part in value)
    ):
        return json.dumps(value, ensure_ascii=False)
    return value


def encode_functions(tools, unwrap) -> list[dict]:
    result = []
    for tool in tools:
        functions = unwrap(tool)
        if functions is None:
            continue
        result.append({FUNCTIONS: functions})
    return result


def decode_functions(tools, wrap) -> list[dict]:
    result = []
    for tool in dictionary_views(tools):
        functions = tool.get(FUNCTIONS)
        if functions is None:
            raise UnsupportedCapabilityError(
                "intermediate tool requires recognized function definitions"
            )
        result.extend(wrap(functions))
    return result
