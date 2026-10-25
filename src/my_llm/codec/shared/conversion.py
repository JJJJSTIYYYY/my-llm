"""Provider-independent message, media and function conversion helpers."""

from __future__ import annotations

import json
from typing import Any

from my_llm.basic.data import _MISSING, dictionary_views
from my_llm.basic.exceptions import MessageValidationError, UnsupportedCapabilityError

FUNCTIONS = "functions"


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
                    **select(result[-1], "protocol", "coalesce", "role", "name", "item"),
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
