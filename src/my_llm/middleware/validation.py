"""Public structural validation of canonical middleware values, without copying.

Wire syntax and typed generation constraints serve different boundaries. Shared
content-kind and reasoning-mark sets keep their semantic restrictions aligned.
Error paths are assembled only while propagating a validation failure.
"""

from __future__ import annotations

from collections.abc import Callable

from my_llm.basic.exceptions import MessageValidationError
from my_llm.middleware.history import MESSAGE_ROLES, HistoryNormalizer

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
        """Store a validation failure and its relative middleware path."""
        super().__init__(message)
        self.path = path


def _validate(value, check: Callable, path: str, *context: object) -> None:
    """Translate internal failures into path-qualified MessageValidationError."""
    try:
        check(value, *context)
    except _InvalidWire as error:
        raise MessageValidationError(f"{path}{error.path}: {error}") from None


def _object(value) -> None:
    """Require a canonical middleware dictionary."""
    if not isinstance(value, dict):
        raise _InvalidWire("expected a middleware object")


def _sequence(value) -> None:
    """Require a middleware list or tuple."""
    if not isinstance(value, (list, tuple)):
        raise _InvalidWire("expected a middleware list or tuple")


def _required(value: dict, names: tuple[str, ...]) -> None:
    """Reject the first missing required field."""
    for name in names:
        if name not in value:
            raise _InvalidWire("required middleware field is missing", f".{name}")


def _fields(value, names: tuple[str, ...]) -> None:
    """Require a dictionary with the specified fields."""
    _object(value)
    _required(value, names)


def _member(value: dict, name: str, check: Callable, *context: object) -> None:
    """Check one field and prepend its name to an internal error path."""
    try:
        check(value[name], *context)
    except _InvalidWire as error:
        error.path = f".{name}{error.path}"
        raise


def _each(values, check: Callable, *context: object) -> None:
    """Visit a sequence once, forwarding its request-local validation context."""
    _sequence(values)
    for index, value in enumerate(values):
        try:
            check(value, *context)
        except _InvalidWire as error:
            error.path = f"[{index}]{error.path}"
            raise


def _kind(value, fields: dict, name: str) -> str:
    """Read a supported type tag and require the fields declared for that kind."""
    _fields(value, ("type",))
    kind = value["type"]
    if not isinstance(kind, str) or kind not in fields:
        raise _InvalidWire(f"unknown middleware {name}: {kind!r}", ".type")
    _required(value, fields[kind])
    return kind


def _content_kind(kind) -> None:
    """Reject an unknown middleware content type."""
    if not isinstance(kind, str) or kind not in _CONTENT_FIELDS:
        raise _InvalidWire(f"unknown middleware content type: {kind!r}")


def _image_source(value) -> None:
    """Require an image source tagged as URL or base64."""
    _fields(value, ("type",))
    if value["type"] not in ("url", "base64"):
        raise _InvalidWire("unknown middleware image source", ".type")


def _result_part(value: dict, history: HistoryNormalizer | None = None) -> None:
    """Restrict tagged tool-result parts to text and images."""
    _content(value)
    if value["type"] not in RESULT_CONTENT_TYPES:
        raise _InvalidWire("middleware tool results require text or image content")
    if history is not None:
        history.observe_result_part(value)


def _result(value, history: HistoryNormalizer | None = None) -> None:
    # Untagged application objects and arrays are serialized as text by readers.
    """Check tagged result parts; leave untagged application data to result readers."""
    if isinstance(value, list) and all(isinstance(v, dict) and "type" in v for v in value):
        _each(value, _result_part, history)
    elif history is not None:
        if isinstance(value, str):
            history.observe_result_part({"type": "text", "text": value})
        elif value is not None:
            history.observe_result_value(value)


def _content(value, history: HistoryValidator | HistoryNormalizer | None = None) -> None:
    """Check required content fields, reasoning marks and nested tool-result parts."""
    kind = _kind(value, _CONTENT_FIELDS, "content type")
    if value.get("item") is not None:
        _member(value, "item", _object)
    if kind == "image":
        _member(value, "source", _image_source)
    elif kind == "thinking" and value["mark"] not in REASONING_MARKS:
        raise _InvalidWire("unknown middleware reasoning mark", ".mark")
    elif kind == "tool_result":
        normalizer = history if isinstance(history, HistoryNormalizer) else None
        if normalizer is not None:
            normalizer.observe_content(value)
        _member(value, "content", _result, normalizer)
    if history is not None and not (
        kind == "tool_result" and isinstance(history, HistoryNormalizer)
    ):
        history.observe_content(value)


def _message(value, history: HistoryValidator | HistoryNormalizer | None = None) -> None:
    """Require role/content fields and check each canonical content block."""
    _fields(value, ("role", "content"))
    if value.get("item") is not None:
        _member(value, "item", _object)
    if history is not None:
        history.observe_message(value)
    _member(value, "content", _contents, history)
    if isinstance(history, HistoryNormalizer):
        history.end_message()


def _contents(values, history: HistoryValidator | HistoryNormalizer | None = None) -> None:
    """Check a sequence of canonical content blocks."""
    _each(values, _content, history)


def _messages(values, history: HistoryValidator | HistoryNormalizer | None = None) -> None:
    """Check message structures and optionally collect associations in the same scan."""
    _each(values, _message, history)


def _function(value) -> None:
    """Require a function declaration with a name field."""
    _fields(value, ("name",))


def _functions(values) -> None:
    """Check the function declarations in one middleware tool group."""
    _each(values, _function)


def _tool(value) -> None:
    """Require and check a middleware function group."""
    _fields(value, ("functions",))
    _member(value, "functions", _functions)


def _tools(values) -> None:
    """Check all middleware function groups."""
    _each(values, _tool)


def _patch(value) -> None:
    # A patch is deliberately not a complete content model.
    """Check the content type of an intentionally partial block update."""
    _fields(value, ("type",))
    _member(value, "type", _content_kind)


def _snapshot_entry(value) -> None:
    """Require a source reference and a complete snapshot block."""
    _fields(value, ("ref", "block"))
    _member(value, "block", _content)


def _snapshot(values) -> None:
    """Check all source-addressed blocks in a snapshot."""
    _each(values, _snapshot_entry)


def _operation(value, tolerate_auxiliary: bool = False) -> None:
    """Check an operation tag, required fields and supported nested structures."""
    if (
        tolerate_auxiliary
        and isinstance(value, dict)
        and value.get("type") in ("metadata", "usage")
    ):
        return  # Optional data is parsed defensively by its state-machine handler.
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


def _operations(values, tolerate_auxiliary: bool = False) -> None:
    """Check every operation in an encoded stream batch."""
    _each(values, _operation, tolerate_auxiliary)


def _batch(value, tolerate_auxiliary: bool = False) -> None:
    """Require a source protocol and structurally valid stream operations."""
    _fields(value, ("protocol", "events"))
    _member(value, "events", _operations, tolerate_auxiliary)


def _batches(values, tolerate_auxiliary: bool = False) -> None:
    """Check encoded stream batches before state-machine dispatch."""
    _each(values, _batch, tolerate_auxiliary)


def _output_entry(value) -> None:
    """Require an output index and a complete content block."""
    _fields(value, ("index", "block"))
    _member(value, "block", _content)


def _output_blocks(values) -> None:
    """Check indexed blocks included in a completed public event."""
    _each(values, _output_entry)


_EVENT_STRUCTURES = (
    ("block", _content),
    ("messages", _messages),
    ("output_blocks", _output_blocks),
)


def _event(value) -> None:
    """Check a public event and the nested structures allowed for its type."""
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
    """Check a sequence of public middleware events."""
    _each(values, _event)


class HistoryValidator:
    """Accumulate tool associations while structural validation visits each block.

    Association errors are deferred until finish so structural errors anywhere in
    the batch, tool definitions and generation controls retain their precedence.
    Keep this state local to one validation call; mutable middleware models never
    retain a trusted-validation flag.
    """

    __slots__ = ("allow_external_calls", "calls", "used", "count", "error")

    def __init__(self, *, allow_external_calls: bool = False) -> None:
        """Initialize independent call/result state for one complete input batch."""
        self.allow_external_calls = allow_external_calls
        self.calls: set[str] = set()
        self.used: set[str] = set()
        self.count = 0
        self.error: str | None = None

    def observe_message(self, message: dict) -> None:
        """Record the first invalid role while continuing structural checks."""
        self.count += 1
        if self.error is None and message.get("role") not in MESSAGE_ROLES:
            self.error = "invalid message role"

    def observe_content(self, block: dict) -> None:
        """Record the first call/result association error without rescanning content."""
        if self.error is not None:
            return
        kind = block.get("type")
        if kind == "tool_use":
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id:
                self.error = "tool call requires an ID"
            elif call_id in self.calls or call_id in self.used:
                self.error = "duplicate tool call ID"
            else:
                self.calls.add(call_id)
        elif kind == "tool_result":
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str) or not call_id:
                self.error = "tool result requires a matching call ID"
            elif call_id in self.used:
                self.error = "duplicate tool result"
            elif call_id not in self.calls and not self.allow_external_calls:
                self.error = "tool result refers to an unknown call ID"
            else:
                self.used.add(call_id)

    def finish(self) -> None:
        """Reject empty history, a deferred error or calls missing a later result."""
        error = self.error
        if not self.count:
            error = "messages must be non-empty"
        elif error is None and self.calls - self.used:
            error = "tool calls require exactly one result before generating again"
        if error is not None:
            raise MessageValidationError(error)


def validate_content(value: dict, path: str = "$") -> None:
    """Check canonical content structure; raise MessageValidationError with its path."""
    _validate(value, _content, path)


def validate_messages(
    values: list,
    path: str = "$.messages",
    *,
    history: HistoryValidator | HistoryNormalizer | None = None,
) -> None:
    """Check canonical message structure without copying or building models.

    Require role/content fields and supported content tags, including required
    block fields and valid tagged tool-result parts. By default this checks only
    structure, without enforcing role-specific content types or call/result pairing.
    Typed generation input adds role/content checks in generation_input. Supplying
    history also collects roles and tool associations in this traversal; call its
    finish method after other request checks to report association errors.
    A HistoryNormalizer instead repairs associations during this same visit.

    Raise MessageValidationError with the failing middleware path. Input values
    remain unchanged, and an empty sequence is structurally valid here.
    """
    _validate(values, _messages, path, history)


def validate_tools(values: list, path: str = "$.tools") -> None:
    """Check canonical function groups and required function names."""
    _validate(values, _tools, path)


def validate_operations(
    values: list, path: str = "$.chunks", *, tolerate_auxiliary: bool = False
) -> None:
    """Check encoded batches and their operations before state-machine dispatch.

    Partial block updates stay partial; finish may reuse an earlier reason.
    Runtime conversion enables tolerate_auxiliary so usage/metadata payloads are
    left to their defensive handlers. Explicit validation is strict by default.
    """
    _validate(values, _batches, path, tolerate_auxiliary)


def validate_events(values: list, path: str = "$.chunks") -> None:
    """Check public events, including optional finish messages and output blocks."""
    _validate(values, _events, path)
