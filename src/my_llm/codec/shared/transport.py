"""SDK schema filtering, request-history validation and response ownership."""

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from copy import deepcopy
from functools import lru_cache
from inspect import Parameter, signature
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


@lru_cache(maxsize=512)
def fields(schema):
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return {name: field.annotation for name, field in schema.model_fields.items()}
    return get_type_hints(schema, include_extras=True)


def adapter(schema):
    return _cached_adapter(_SchemaKey(schema))


@lru_cache(maxsize=512)
def _cached_adapter(key):
    return TypeAdapter(key.schema)


@lru_cache(maxsize=512)
def parameter_types(method):
    """Read the public SDK's keyword contract once, independently of payload data."""
    annotations = get_type_hints(method, include_extras=True)
    return {
        name: annotations.get(name, Any)
        for name, parameter in signature(method).parameters.items()
        if name != "self"
        and parameter.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
    }


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


_MISSING = object()


def _field_value(value, name):
    """Read a declared field without serializing or inspecting SDK extensions."""
    if isinstance(value, dict):
        return value[name] if name in value else _MISSING
    if isinstance(value, BaseModel):
        item = getattr(value, name, _MISSING)
        # SDK models expose unset optional fields as None. Explicit None is
        # nevertheless a supplied protocol value and must survive projection.
        if item is None and name not in value.model_fields_set:
            return _MISSING
        return item
    return _MISSING


class _FilterPlan:
    """Type-only graph node; compiled before it is published in the plan cache."""

    __slots__ = ("apply", "project", "score", "literals")


def _literal_routes(choices):
    """Compile admissible union branches for each declared literal field."""
    names = dict.fromkeys(name for choice in choices for name in choice.literals)
    routes = []
    for name in names:
        unrestricted = 0
        matches = {}
        for index, choice in enumerate(choices):
            mask = 1 << index
            if name not in choice.literals:
                unrestricted |= mask
                continue
            for literal in choice.literals[name]:
                try:
                    matches[literal] = matches.get(literal, 0) | mask
                except TypeError:
                    # Non-hashable Literal declarations retain ordinary scoring.
                    break
            else:
                continue
            break
        else:
            routes.append((name, unrestricted, matches))
    return routes


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


def _plan(schema, delegated_paths=()):
    return _compiled_plan(_SchemaKey(schema), delegated_paths)


@lru_cache(maxsize=512)
def _compiled_plan(key, delegated_paths):
    # SDK models can be recursive. Register each node before its children so
    # recursive annotations refer to the same plan without storing request data.
    plans = {}

    def compile(annotation, delegated_paths=()):
        annotation = _unwrap(annotation)
        key = (id(annotation), delegated_paths)
        if key in plans:
            return plans[key]
        node = plans[key] = _FilterPlan()
        node.literals = {}
        origin = get_origin(annotation)
        arguments = get_args(annotation)

        if annotation in (Any, object):
            node.score = lambda value: 0
        elif origin is Literal:
            literal_types = tuple({type(item) for item in arguments})
            node.score = lambda value: (
                1 if value in arguments else 0 if isinstance(value, literal_types) else -1
            )
        elif origin in (list, tuple, Iterable, Sequence):
            node.score = lambda value: 1 if isinstance(value, list) else -1
        elif origin in (dict, Mapping):
            node.score = lambda value: 0 if isinstance(value, dict) else -1
        elif isinstance(annotation, type):
            node.score = lambda value: 1 if isinstance(value, annotation) else -1
        else:
            node.score = lambda value: -1

        if origin in (Union, UnionType):
            choices = tuple(
                compile(item, delegated_paths) for item in arguments if item is not type(None)
            )
            routes = _literal_routes(choices)

            def select(value):
                if len(choices) == 1:
                    return choices[0]
                if routes:
                    eligible = (1 << len(choices)) - 1
                    for name, unrestricted, matches in routes:
                        item = (
                            value.get(name, _MISSING)
                            if isinstance(value, dict)
                            else _field_value(value, name)
                        )
                        if item is _MISSING:
                            continue
                        try:
                            eligible &= unrestricted | matches.get(item, 0)
                        except TypeError:
                            # Invalid unhashable values are handled by the same
                            # score/validation path as before indexing.
                            continue
                    candidates = []
                    while eligible:
                        bit = eligible & -eligible
                        candidates.append(choices[bit.bit_length() - 1])
                        eligible ^= bit
                else:
                    candidates = choices
                if not candidates:
                    raise UnsupportedCapabilityError(
                        "target SDK cannot represent this protocol value"
                    )
                # Strictly greater retains the first candidate when scores tie,
                # matching max(..., key=...) without rescoring the winner.
                choice = candidates[0]
                score = choice.score(value)
                for candidate in candidates[1:]:
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
        elif origin in (list, tuple, Iterable, Sequence) and arguments:
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
            for name, item in declared.items():
                unwrapped = _unwrap(item)
                if get_origin(unwrapped) is Literal:
                    node.literals[name] = get_args(unwrapped)
            literal_fields = tuple(node.literals.items())
            children = {
                name: compile(
                    item,
                    tuple(path[1:] for path in delegated_paths if path and path[0] == name),
                )
                for name, item in declared.items()
            }

            def structured_score(value):
                if isinstance(value, dict):
                    for name, literals in literal_fields:
                        item = value.get(name, _MISSING)
                        if item is not _MISSING and item not in literals:
                            return -1
                    return sum(name in value for name in children) + 1
                if not isinstance(value, BaseModel):
                    return -1
                for name, literals in literal_fields:
                    item = _field_value(value, name)
                    if item is not _MISSING and item not in literals:
                        return -1
                return sum(_field_value(value, name) is not _MISSING for name in children) + 1

            def structured_copy(value):
                if isinstance(value, dict):
                    return {
                        name: child.apply(item)
                        for name, child in children.items()
                        if (item := value.get(name, _MISSING)) is not _MISSING
                    }
                if not isinstance(value, BaseModel):
                    return _copy_payload(value)
                return {
                    name: children[name].apply(item)
                    for name in children
                    if (item := _field_value(value, name)) is not _MISSING
                }

            def structured_projection(value):
                if isinstance(value, dict):
                    return {
                        name: child.project(item)
                        for name, child in children.items()
                        if (item := value.get(name, _MISSING)) is not _MISSING
                    }
                if not isinstance(value, BaseModel):
                    return value
                return {
                    name: children[name].project(item)
                    for name in children
                    if (item := _field_value(value, name)) is not _MISSING
                }

            node.score = structured_score
            node.apply = structured_copy
            node.project = structured_projection
        else:
            # Open dictionaries (tool arguments/results and JSON Schema) are
            # atomic application data; protocol filtering never inspects keys.
            node.apply = _copy_payload
            node.project = _borrow_payload
        if () in delegated_paths:
            # Only Transport delegates message/tool leaves to its own hooks.
            # Enclosing structures and unrelated application data remain owned.
            node.apply = _borrow_payload
        return node

    return compile(key.schema, delegated_paths)


def supported(value, schema, *, delegated_paths=()):
    """Filter owned values; Transport alone delegates leaves to its later hooks.

    Delegated leaves remain borrowed until filter_messages/filter_tools returns.
    Ordinary callers omit delegated_paths and always receive isolated payloads.
    """
    if () in delegated_paths:
        return value
    return _plan(schema, delegated_paths).apply(value)


def projected(value, schema):
    """Project SDK fields for read-only validation, borrowing atomic payloads."""
    return _plan(schema).project(value)


def declared_path(schema, keys):
    """Whether an SDK schema already declares a path, including open JSON leaves."""
    return _declared_path(_SchemaKey(schema), keys)


@lru_cache(maxsize=512)
def _declared_path(key, keys):
    schema = _unwrap(key.schema)
    if not keys or schema in (Any, object):
        return True
    origin = get_origin(schema)
    arguments = get_args(schema)
    if origin in (Union, UnionType):
        return any(declared_path(item, keys) for item in arguments if item is not type(None))
    if origin in (dict, Mapping):
        return True
    if origin in (list, tuple, Iterable, Sequence):
        return (
            isinstance(keys[0], int) and bool(arguments) and declared_path(arguments[0], keys[1:])
        )
    if _structured(schema):
        declared = fields(schema)
        return keys[0] in declared and declared_path(declared[keys[0]], keys[1:])
    return False


def path_item(value, keys):
    """Read one declared profile path without walking neighboring fields."""
    for key in keys:
        if isinstance(key, int):
            if not isinstance(value, list) or key >= len(value):
                return _MISSING
            value = value[key]
        else:
            value = _field_value(value, key)
            if value is _MISSING:
                return _MISSING
    return value


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
