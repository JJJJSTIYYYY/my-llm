"""Object JSON Schema conversion and extraction of JSON objects from text."""

import json
import re
from collections.abc import Iterator
from copy import deepcopy
from math import isfinite
from typing import Any
from urllib.parse import unquote

from pydantic import TypeAdapter

_SCHEMA_MAPS = {"properties", "patternProperties", "dependentSchemas"}
_SCHEMA_LISTS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMA_VALUES = {
    "additionalProperties",
    "unevaluatedProperties",
    "propertyNames",
    "items",
    "additionalItems",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "unevaluatedItems",
    "contentSchema",
}
_ANNOTATIONS = {
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$comment",
}


def object_json_schema(schema: type | dict[str, Any]) -> dict[str, Any]:
    """Return an owned object schema with document-local JSON Pointer refs inlined.

    Accept JSON Schema dictionaries and Pydantic-supported types. Resolve refs
    against the original document, remove definitions, and leave annotation/data
    values (such as defaults and examples) intact. Unsupported external/anchor
    references, missing targets and reference cycles raise ValueError.
    """
    if isinstance(schema, dict):
        root = schema
    elif isinstance(schema, type):
        root = TypeAdapter(schema).json_schema()
    else:
        raise TypeError("schema must be a JSON Schema dictionary or a supported Python type")

    def resolve(ref: str) -> tuple[Any, tuple[str, ...]]:
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise ValueError(f"only document-local JSON Pointer references are supported: {ref!r}")
        pointer = unquote(ref[1:])
        if pointer and not pointer.startswith("/"):
            raise ValueError(f"reference must use a JSON Pointer: {ref!r}")
        parts = pointer[1:].split("/") if pointer else []
        if any(re.search(r"~(?![01])", part) for part in parts):
            raise ValueError(f"invalid JSON Pointer escape: {ref!r}")
        keys = tuple(part.replace("~1", "/").replace("~0", "~") for part in parts)
        target = root
        try:
            for key in keys:
                if isinstance(target, list):
                    if not re.fullmatch(r"0|[1-9][0-9]*", key):
                        raise ValueError("invalid array index")
                    target = target[int(key)]
                else:
                    target = target[key]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"unresolved JSON Schema reference: {ref!r}") from exc
        if not isinstance(target, (dict, bool)):
            raise ValueError(f"reference must point to a schema: {ref!r}")
        return target, keys

    def expand(node: Any, active: tuple[tuple[str, ...], ...] = ()) -> Any:
        if not isinstance(node, dict):
            return deepcopy(node)
        result = {}
        for key, value in node.items():
            if key in ("$ref", "$defs", "definitions"):
                continue
            if key in _SCHEMA_MAPS:
                result[key] = {name: expand(child, active) for name, child in value.items()}
            elif key == "dependencies":
                result[key] = {
                    name: expand(child, active) if isinstance(child, dict) else deepcopy(child)
                    for name, child in value.items()
                }
            elif key in _SCHEMA_LISTS or key == "items" and isinstance(value, list):
                result[key] = [expand(child, active) for child in value]
            elif key in _SCHEMA_VALUES:
                result[key] = expand(value, active)
            else:
                result[key] = deepcopy(value)
        if "$ref" not in node:
            return result
        target, keys = resolve(node["$ref"])
        if keys in active:
            raise ValueError(f"circular JSON Schema reference cannot be inlined: {node['$ref']!r}")
        target = expand(target, (*active, keys))
        if isinstance(target, bool):
            return result if target else False
        conflicts = (target.keys() & result.keys()) - _ANNOTATIONS
        if any(target[key] != result[key] for key in conflicts):
            return {"allOf": [target, result]}
        return {**target, **result}

    parameters = expand(root)
    if not isinstance(parameters, dict) or parameters.get("type", "object") != "object":
        raise ValueError("schema must describe a JSON object")
    return parameters


_FENCE_LINE = re.compile(
    r"(?P<fence>`{3,}|~{3,})[ \t]*(?P<language>[\w.+/#-]*)[ \t]*\r?$",
    re.MULTILINE,
)


def _fenced_regions(text: str) -> Iterator[tuple[str | None, str]]:
    """Split code and prose once; an unclosed fence still owns its remaining body."""
    opening = None
    offset = 0
    for match in _FENCE_LINE.finditer(text):
        if opening is None:
            yield None, text[offset : match.start()]
            opening = match
            offset = match.end()
        elif (
            not match["language"].strip()
            and match["fence"][0] == opening["fence"][0]
            and len(match["fence"]) >= len(opening["fence"])
        ):
            yield opening["language"].strip().lower(), text[offset : match.start()]
            opening = None
            offset = match.end()
    yield None if opening is None else opening["language"].strip().lower(), text[offset:]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys instead of silently overwriting generated values."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    """Reject NaN, Infinity and floats that overflow to a non-finite value."""
    number = float(value)
    if not isfinite(number):
        raise ValueError("JSON numbers must be finite")
    return number


def _object_candidates(text: str, decoder: json.JSONDecoder) -> Iterator[dict[str, Any]]:
    """Scan outer containers once, never extracting their nested objects separately."""
    stack = []
    start = 0
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            if not stack:
                start = index
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            if char != stack.pop():
                raise ValueError("response contains mismatched JSON delimiters")
            if not stack:
                try:
                    value = decoder.decode(text[start : index + 1])
                except (ValueError, RecursionError):
                    # Skip the entire invalid container, including all its children.
                    continue
                if isinstance(value, dict):
                    yield value
    if stack:
        raise ValueError("response contains an incomplete JSON container")


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract one unambiguous, complete JSON object without repairing model output.

    Prefer a complete document, then JSON-labelled fences, then unlabelled fences,
    then prose outside other fenced code. Respect outer array/object boundaries,
    quoted strings and escapes; invalid parents never expose valid child objects.
    Multiple valid objects, duplicate keys, non-finite numbers and incomplete
    containers raise ValueError. Never evaluate text as Python or guess missing data.
    """
    text = text.strip().lstrip("\ufeff").strip()
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_object,
        parse_float=_finite_float,
        parse_constant=_finite_float,
    )
    try:
        value = decoder.decode(text)
    except (ValueError, RecursionError):
        pass
    else:
        if isinstance(value, dict):
            return value
        raise ValueError("response JSON must be an object")

    json_blocks, plain_blocks, prose = [], [], []
    for language, region in _fenced_regions(text):
        if language is None:
            prose.append(region)
        elif language in ("json", "application/json"):
            json_blocks.append(region)
        elif not language:
            plain_blocks.append(region)

    result = None
    for region in json_blocks or plain_blocks or prose:
        for value in _object_candidates(region, decoder):
            if result is not None:
                raise ValueError("response contains multiple JSON objects")
            result = value
    if result is None:
        raise ValueError("response contains no valid JSON object")
    return result
