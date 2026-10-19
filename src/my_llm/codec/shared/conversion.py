"""Explicit intermediate values and standard protocol conversion helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

from my_llm.exceptions import MessageValidationError, UnsupportedCapabilityError

FUNCTIONS = "functions"


def read(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value.copy() if isinstance(value, (dict, list)) else value


def dictionary_views(values: Any) -> Iterator[dict[str, Any]]:
    """Read protocol dictionaries or public SDK models without retaining them."""
    if not isinstance(values, (list, tuple)):
        raise MessageValidationError("expected a list of protocol dictionaries or SDK objects")
    for value in values:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", exclude_none=True)
        if not isinstance(value, dict):
            raise MessageValidationError("expected protocol dictionaries or SDK objects")
        yield value


def dictionaries(values: Any) -> list[dict[str, Any]]:
    """Capture owned input values at a public conversion boundary."""
    return [
        value if hasattr(values[index], "model_dump") else deepcopy(value)
        for index, value in enumerate(dictionary_views(values))
    ]


def coalesce_messages(messages: list) -> list[dict]:
    """Combine consecutive Responses fragments from one assistant history turn."""
    result = []
    for message in dictionary_views(messages):
        if (
            result
            and message.get("coalesce")
            and result[-1].get("coalesce")
            and message.get("role") == result[-1].get("role") == "assistant"
            and source_protocol(message) == source_protocol(result[-1])
        ):
            result[-1] = {
                **result[-1],
                "content": [*result[-1]["content"], *message.get("content", [])],
            }
        else:
            result.append(message)
    return result


def blocks(content: Any) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return dictionaries(content)


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
    if isinstance(value, dict):
        return deepcopy(value)
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise MessageValidationError("function arguments must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise MessageValidationError("function arguments must be a JSON object")
    return parsed


def json_arguments(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def tool_output(value: Any) -> Any:
    """Native object results become JSON text where the protocol requires text."""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value


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
    return {key: deepcopy(value[key]) for key in fields if key in value}


def select_present(value: dict, *fields: str) -> dict:
    return {key: child for key, child in select(value, *fields).items() if child is not None}


def project_message(message: dict, project, *, fields: tuple[str, ...] = ("role",)) -> dict:
    content = []
    for block in blocks(message.get("content")):
        rendered = project(block)
        if rendered is not None:
            content.append(rendered)
    return {**select(message, *fields, "protocol"), "content": content}


def unsupported(block: dict):
    raise UnsupportedCapabilityError(
        f"target protocol cannot represent content type {block.get('type')!r}"
    )


def project_block(
    block: dict, fields: dict, *, text_fields: dict | None = None, omitted: tuple = ()
):
    """Apply the target's declared field projection and visible-text fallbacks."""
    kind = block.get("type")
    if kind in omitted:
        return None
    if kind in (text_fields or {}):
        return {"type": "text", "text": block.get(text_fields[kind], "")}
    if kind not in fields:
        return unsupported(block)
    value = select(block, *fields[kind])
    if "source" in value:
        value["source"] = select(value["source"], "type", "url", "media_type", "data", "file_id")
    return value


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


def encode_functions(tools, unwrap) -> list[dict]:
    result = []
    for tool in tools:
        functions = unwrap(tool)
        if functions is None:
            raise UnsupportedCapabilityError("tool type is not supported by the selected protocol")
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
        result.extend(wrap(deepcopy(functions)))
    return result
