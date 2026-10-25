"""Public structural validation of canonical middleware values, without copying.

Wire syntax and typed generation constraints serve different boundaries. Shared
content-kind and reasoning-mark sets keep their semantic restrictions aligned.
Error paths are assembled only while propagating a validation failure.
"""

from collections.abc import Callable

from my_llm.basic.exceptions import MessageValidationError

RESULT_CONTENT_TYPES = ("text", "image")
REASONING_MARKS = ("raw", "summary", "encrypted")

_CONTENT_FIELDS = {
    "text": ("text",),
    "image": ("source",),
    "thinking": ("thinking", "mark"),
    "tool_use": ("id", "name", "input"),
    "tool_result": ("tool_use_id", "content"),
}
_OPERATION_FIELDS = {
    "start": (),
    "metadata": ("path", "values"),
    "usage": ("usage",),
    "snapshot": ("blocks",),
    "reason": (),
    "finish": (),
    "block_set": ("ref", "block"),
    "block_append": ("block",),
    "block_update": ("ref", "block"),
    "block_delta": ("ref", "kind", "field", "value"),
    "block_stop": ("ref",),
}
_EVENT_FIELDS = {
    "start": (),
    "block_start": ("block",),
    "delta": ("field", "value"),
    "block_stop": ("block",),
    "block": ("block",),
    "tool": ("block",),
    "finish": (),
}
_EVENT_OPTIONAL = {
    "start": (),
    "block_start": ("index", "source_ref"),
    "delta": ("index", "source_ref"),
    "block_stop": ("index", "source_ref"),
    "block": ("index", "source_ref"),
    "tool": ("index", "source_ref", "tool_index"),
    "finish": ("reason", "messages", "output_blocks"),
}


class _InvalidWire(Exception):
    """Private validation failure; application exceptions are never intercepted."""

    def __init__(self, message: str, path: str = "") -> None:
        super().__init__(message)
        self.path = path


def _validate(value, check: Callable, path: str) -> None:
    try:
        check(value)
    except _InvalidWire as error:
        raise MessageValidationError(f"{path}{error.path}: {error}") from None


def _object(value) -> None:
    if not isinstance(value, dict):
        raise _InvalidWire("expected a middleware object")


def _sequence(value) -> None:
    if not isinstance(value, (list, tuple)):
        raise _InvalidWire("expected a middleware list or tuple")


def _required(value: dict, names: tuple[str, ...]) -> None:
    for name in names:
        if name not in value:
            raise _InvalidWire("required middleware field is missing", f".{name}")


def _fields(value, names: tuple[str, ...]) -> None:
    _object(value)
    _required(value, names)


def _member(value: dict, name: str, check: Callable) -> None:
    try:
        check(value[name])
    except _InvalidWire as error:
        error.path = f".{name}{error.path}"
        raise


def _each(values, check: Callable) -> None:
    _sequence(values)
    for index, value in enumerate(values):
        try:
            check(value)
        except _InvalidWire as error:
            error.path = f"[{index}]{error.path}"
            raise


def _kind(value, fields: dict, name: str) -> str:
    _fields(value, ("type",))
    kind = value["type"]
    if not isinstance(kind, str) or kind not in fields:
        raise _InvalidWire(f"unknown middleware {name}: {kind!r}", ".type")
    _required(value, fields[kind])
    return kind


def _content_kind(kind) -> None:
    if not isinstance(kind, str) or kind not in _CONTENT_FIELDS:
        raise _InvalidWire(f"unknown middleware content type: {kind!r}")


def _image_source(value) -> None:
    _fields(value, ("type",))
    if value["type"] not in ("url", "base64"):
        raise _InvalidWire("unknown middleware image source", ".type")


def _result_part(value: dict) -> None:
    _content(value)
    if value["type"] not in RESULT_CONTENT_TYPES:
        raise _InvalidWire("middleware tool results require text or image content")


def _result(value) -> None:
    # Untagged application objects and arrays are serialized as text by readers.
    if isinstance(value, list) and all(isinstance(v, dict) and "type" in v for v in value):
        _each(value, _result_part)


def _content(value) -> None:
    kind = _kind(value, _CONTENT_FIELDS, "content type")
    if value.get("item") is not None:
        _member(value, "item", _object)
    if kind == "image":
        _member(value, "source", _image_source)
    elif kind == "thinking" and value["mark"] not in REASONING_MARKS:
        raise _InvalidWire("unknown middleware reasoning mark", ".mark")
    elif kind == "tool_result":
        _member(value, "content", _result)


def _message(value) -> None:
    _fields(value, ("role", "content"))
    if value.get("item") is not None:
        _member(value, "item", _object)
    _member(value, "content", _contents)


def _contents(values) -> None:
    _each(values, _content)


def _messages(values) -> None:
    _each(values, _message)


def _function(value) -> None:
    _fields(value, ("name",))


def _functions(values) -> None:
    _each(values, _function)


def _tool(value) -> None:
    _fields(value, ("functions",))
    _member(value, "functions", _functions)


def _tools(values) -> None:
    _each(values, _tool)


def _patch(value) -> None:
    # A patch is deliberately not a complete content model.
    _fields(value, ("type",))
    _member(value, "type", _content_kind)


def _snapshot_entry(value) -> None:
    _fields(value, ("ref", "block"))
    _member(value, "block", _content)


def _snapshot(values) -> None:
    _each(values, _snapshot_entry)


def _operation(value) -> None:
    kind = _kind(value, _OPERATION_FIELDS, "operation")
    if kind == "block_delta":
        if value["field"] not in ("text", "thinking", "arguments", "signature"):
            raise _InvalidWire("unknown middleware delta field", ".field")
        _member(value, "kind", _content_kind)
    elif kind in ("block_set", "block_append"):
        _member(value, "block", _content)
    elif kind == "block_update":
        _member(value, "block", _patch)
    elif kind == "metadata":
        _member(value, "path", _sequence)
        _member(value, "values", _object)
    elif kind == "snapshot":
        _member(value, "blocks", _snapshot)


def _operations(values) -> None:
    _each(values, _operation)


def _batch(value) -> None:
    _fields(value, ("protocol", "events"))
    _member(value, "events", _operations)


def _batches(values) -> None:
    _each(values, _batch)


def _output_entry(value) -> None:
    _fields(value, ("index", "block"))
    _member(value, "block", _content)


def _output_blocks(values) -> None:
    _each(values, _output_entry)


_EVENT_STRUCTURES = (
    ("block", _content),
    ("messages", _messages),
    ("output_blocks", _output_blocks),
)


def _event(value) -> None:
    kind = _kind(value, _EVENT_FIELDS, "event")
    if kind == "delta" and value["field"] not in ("text", "thinking"):
        raise _InvalidWire("unknown middleware delta field", ".field")
    for name, check in _EVENT_STRUCTURES:
        if name in value:
            if name not in _EVENT_FIELDS[kind] and name not in _EVENT_OPTIONAL[kind]:
                raise _InvalidWire(
                    f"field is not supported by middleware {kind!r} events", f".{name}"
                )
            _member(value, name, check)


def _events(values) -> None:
    _each(values, _event)


def validate_content(value: dict, path: str = "$") -> None:
    """Check canonical content structure; raise MessageValidationError with its path."""
    _validate(value, _content, path)


def validate_messages(values: list, path: str = "$.messages") -> None:
    """Check canonical history structure without model construction or payload copies."""
    _validate(values, _messages, path)


def validate_tools(values: list, path: str = "$.tools") -> None:
    """Check canonical function groups and required function names."""
    _validate(values, _tools, path)


def validate_operations(values: list, path: str = "$.chunks") -> None:
    """Check encoded batches and their operations before state-machine dispatch.

    Partial block updates stay partial; finish may reuse an earlier reason.
    """
    _validate(values, _batches, path)


def validate_events(values: list, path: str = "$.chunks") -> None:
    """Check public events, including optional finish messages and output blocks."""
    _validate(values, _events, path)
