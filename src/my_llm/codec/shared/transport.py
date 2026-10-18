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

from my_llm.codec.shared.conversion import OPAQUE, dictionaries
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


def _score(value, schema):
    schema = _unwrap(schema)
    if isinstance(value, dict) and _structured(schema):
        declared = fields(schema)
        for key in value.keys() & declared.keys():
            annotation = _unwrap(declared[key])
            if get_origin(annotation) is Literal and value[key] not in get_args(annotation):
                return -1
        return len(value.keys() & declared.keys()) + 1
    if schema in (Any, object):
        return 0
    if get_origin(schema) is Literal:
        return 1 if value in get_args(schema) else -1
    if isinstance(value, list) and get_origin(schema) in (list, tuple, Iterable):
        return 1
    if isinstance(value, dict) and get_origin(schema) is dict:
        return 0
    return 1 if isinstance(schema, type) and isinstance(value, schema) else -1


def supported(value, schema):
    """Copy declared fields recursively, without pruning JSON Schema or tool data."""
    origin = get_origin(schema)
    arguments = get_args(schema)
    if origin in (Annotated, Required, NotRequired):
        return supported(value, arguments[0])
    if origin in (Union, UnionType):
        if value is None:
            return None
        choices = [choice for choice in arguments if choice is not type(None)]
        if len(choices) == 1:
            return supported(value, choices[0])
        choice = max(choices, key=lambda item: _score(value, item))
        if _score(value, choice) < 0:
            raise UnsupportedCapabilityError("target SDK cannot represent this protocol value")
        return supported(value, choice)
    if isinstance(value, list) and origin in (list, tuple, Iterable):
        return [supported(item, arguments[0]) for item in value] if arguments else deepcopy(value)
    if isinstance(value, dict) and _structured(schema):
        return {
            key: supported(item, fields(schema)[key])
            for key, item in value.items()
            if key in fields(schema)
        }
    # Arbitrary dictionaries (tool arguments/results and JSON Schemas) are atomic
    # application payloads. Their keys are never interpreted as protocol fields.
    return deepcopy(value)


def validate_history(messages: list[dict], *, allow_external_calls: bool = False) -> None:
    if not messages:
        raise MessageValidationError("messages must be non-empty")
    calls = set()
    used = set()
    for message in dictionaries(messages):
        if message.get(OPAQUE):
            continue
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
