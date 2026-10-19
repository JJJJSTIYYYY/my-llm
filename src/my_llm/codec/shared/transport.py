"""SDK schema filtering, request-history validation and response ownership."""

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from copy import deepcopy
from functools import cache
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    NotRequired,
    Required,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, TypeAdapter

from my_llm.codec.shared.conversion import dictionary_views
from my_llm.exceptions import MessageValidationError, UnsupportedCapabilityError


@cache
def fields(schema):
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return {name: field.annotation for name, field in schema.model_fields.items()}
    return get_type_hints(schema, include_extras=True)


@cache
def adapter(schema):
    return TypeAdapter(schema)


def _unwrap(schema):
    while get_origin(schema) in (Annotated, Required, NotRequired):
        schema = get_args(schema)[0]
    return schema


def _structured(schema):
    return isinstance(schema, type) and (
        hasattr(schema, "__required_keys__") or issubclass(schema, BaseModel)
    )


def _copy_payload(value):
    if type(value) in (str, int, float, bool, bytes, type(None)):
        return value
    return deepcopy(value)


def _borrow_payload(value):
    return value


class _FilterPlan:
    """Type-only graph node; compiled before it is published in the plan cache."""

    __slots__ = ("apply", "project", "score")


def _same_schema(left, right):
    if left is right:
        return True
    if left != right:
        return False
    # Union equality ignores member order, but filtering resolves tied scores
    # by that order. Compare nested generic arguments in their declared order.
    return all(
        _same_schema(first, second)
        for first, second in zip(get_args(left), get_args(right), strict=True)
    )


class _SchemaKey:
    __slots__ = ("schema", "_hash", "_hashable")

    def __init__(self, schema):
        self.schema = schema
        try:
            self._hash = hash(schema)
            self._hashable = True
        except TypeError:
            # Annotated permits unhashable metadata. Such annotations remain
            # cacheable by identity, with their metadata outside filtering.
            self._hash = id(schema)
            self._hashable = False

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.schema is other.schema or (
            self._hashable and other._hashable and _same_schema(self.schema, other.schema)
        )


def _plan(schema):
    return _compiled_plan(_SchemaKey(schema))


@cache
def _compiled_plan(key):
    # SDK models can be recursive. Register each node before its children so
    # recursive annotations refer to the same plan without storing request data.
    plans = {}

    def compile(annotation):
        annotation = _unwrap(annotation)
        key = id(annotation)
        if key in plans:
            return plans[key]
        node = plans[key] = _FilterPlan()
        origin = get_origin(annotation)
        arguments = get_args(annotation)

        if annotation in (Any, object):
            node.score = lambda value: 0
        elif origin is Literal:
            node.score = lambda value: 1 if value in arguments else -1
        elif origin in (list, tuple, Iterable):
            node.score = lambda value: 1 if isinstance(value, list) else -1
        elif origin is dict:
            node.score = lambda value: 0 if isinstance(value, dict) else -1
        elif isinstance(annotation, type):
            node.score = lambda value: 1 if isinstance(value, annotation) else -1
        else:
            node.score = lambda value: -1

        if origin in (Union, UnionType):
            choices = tuple(compile(item) for item in arguments if item is not type(None))

            def select(value):
                if len(choices) == 1:
                    return choices[0]
                # Strictly greater retains the first candidate when scores tie,
                # matching max(..., key=...) without rescoring the winner.
                choice = choices[0]
                score = choice.score(value)
                for candidate in choices[1:]:
                    candidate_score = candidate.score(value)
                    if candidate_score > score:
                        choice, score = candidate, candidate_score
                if score < 0:
                    raise UnsupportedCapabilityError(
                        "target SDK cannot represent this protocol value"
                    )
                return choice

            node.apply = lambda value: None if value is None else select(value).apply(value)
            node.project = lambda value: None if value is None else select(value).project(value)
        elif origin in (list, tuple, Iterable) and arguments:
            item_plan = compile(arguments[0])
            node.apply = lambda value: (
                [item_plan.apply(item) for item in value]
                if isinstance(value, list)
                else _copy_payload(value)
            )
            node.project = lambda value: (
                [item_plan.project(item) for item in value] if isinstance(value, list) else value
            )
        elif _structured(annotation):
            declared = fields(annotation)
            children = {name: compile(item) for name, item in declared.items()}
            declared_keys = children.keys()
            literal_fields = []
            for name, item in declared.items():
                unwrapped = _unwrap(item)
                if get_origin(unwrapped) is Literal:
                    literal_fields.append((name, get_args(unwrapped)))

            def structured_score(value):
                if not isinstance(value, dict):
                    return 1 if isinstance(value, annotation) else -1
                for name, literals in literal_fields:
                    if name in value and value[name] not in literals:
                        return -1
                return len(value.keys() & declared_keys) + 1

            def structured_copy(value):
                if isinstance(value, BaseModel):
                    serialized = value.model_dump(mode="json", exclude_none=True)
                    if isinstance(serialized, dict):
                        value = serialized
                if not isinstance(value, dict):
                    return _copy_payload(value)
                return {
                    name: children[name].apply(item)
                    for name, item in value.items()
                    if name in children
                }

            def structured_projection(value):
                if isinstance(value, BaseModel):
                    serialized = value.model_dump(mode="json", exclude_none=True)
                    if isinstance(serialized, dict):
                        value = serialized
                if not isinstance(value, dict):
                    return value
                return {
                    name: children[name].project(item)
                    for name, item in value.items()
                    if name in children
                }

            node.score = structured_score
            node.apply = structured_copy
            node.project = structured_projection
        else:
            # Open dictionaries (tool arguments/results and JSON Schema) are
            # atomic application data; protocol filtering never inspects keys.
            node.apply = _copy_payload
            node.project = _borrow_payload
        return node

    return compile(key.schema)


def supported(value, schema):
    """Filter with a cached type plan, isolating mutable application payloads."""
    return _plan(schema).apply(value)


def projected(value, schema):
    """Project SDK fields for read-only validation, borrowing atomic payloads."""
    return _plan(schema).project(value)


def validate_history(messages: list[dict], *, allow_external_calls: bool = False) -> None:
    if not messages:
        raise MessageValidationError("messages must be non-empty")
    calls = set()
    used = set()
    for message in dictionary_views(messages):
        if message.get("role") not in ("system", "developer", "user", "assistant"):
            raise MessageValidationError("invalid message role")
        for block in message.get("content", []):
            if block.get("type") == "tool_use":
                call_id = block.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise MessageValidationError("tool call requires an ID")
                if call_id in calls or call_id in used:
                    raise MessageValidationError("duplicate tool call ID")
                calls.add(call_id)
            elif block.get("type") == "tool_result":
                call_id = block.get("tool_use_id")
                if not isinstance(call_id, str) or not call_id:
                    raise MessageValidationError("tool result requires a matching call ID")
                if call_id in used:
                    raise MessageValidationError("duplicate tool result")
                # Responses may reference calls stored in server-side history.
                # Duplicate results are still invalid in that mode.
                if call_id not in calls and not allow_external_calls:
                    raise MessageValidationError("tool result refers to an unknown call ID")
                used.add(call_id)
    if calls - used:
        raise MessageValidationError(
            "tool calls require exactly one result before generating again"
        )


@asynccontextmanager
async def owned_stream(
    create: Callable[[], Awaitable[Any]],
    *,
    close_stream: str = "close",
) -> AsyncIterator[Any]:
    """Close one response stream without closing its borrowed SDK client."""
    async with AsyncExitStack() as resources:
        upstream = await create()
        resources.push_async_callback(getattr(upstream, close_stream))
        yield upstream
