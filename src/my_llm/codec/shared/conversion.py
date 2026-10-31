"""Provider-independent message, media and function conversion helpers."""

from __future__ import annotations

import json
from typing import Any

from my_llm.basic.data import _MISSING, dictionary_views
from my_llm.basic.exceptions import MessageValidationError
from my_llm.basic.semantic_loss import report_loss
from my_llm.middleware.validation import HistoryValidator

FUNCTIONS = "functions"


def split_system_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate instructions from conversation history in one pass."""
    system, history = [], []
    for message in messages:
        if message.get("role") in ("system", "developer"):
            system.append(message)
        else:
            history.append(message)
    return system, history


def blocks(content: Any) -> list[dict]:
    """Normalize absent, string or structured native content into block dictionaries."""
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
    """Concatenate text blocks while ignoring other content kinds."""
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
    """Preserve a native argument string or JSON-serialize an argument object."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def source_protocol(value: dict) -> str | None:
    """Read source provenance used to select native replay behavior."""
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


def mapping_list(value) -> bool:
    """Whether a value is a list containing only dictionaries."""
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def match_all(values, predicate) -> bool:
    """Match a nonempty normalized batch, treating malformed input as a non-match."""
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
    """Project specified fields while omitting missing and None values."""
    return {
        key: child
        for key in fields
        if (child := value.get(key, _MISSING)) is not _MISSING and child is not None
    }


def project_message(message: dict, project, *, fields: tuple[str, ...] = ("role",)) -> dict:
    """Project each content block, skipping unsupported blocks and retaining named fields."""
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
    """Unwrap supported native function declarations into middleware groups."""
    result = []
    for tool in tools:
        functions = unwrap(tool)
        if functions is None:
            continue
        result.append({FUNCTIONS: functions})
    return result


def decode_functions(tools, wrap) -> list[dict]:
    """Wrap canonical function groups as native tools and report unsupported groups."""
    result = []
    for tool in dictionary_views(tools):
        functions = tool.get(FUNCTIONS)
        if functions is None:
            report_loss(tool, "target does not support this tool definition")
            continue
        result.extend(wrap(functions))
    return result


def validate_history(messages: list[dict], *, allow_external_calls: bool = False) -> None:
    """Require nonempty canonical history with one later result per tool call.

    Args:
        messages: Middleware messages, with native tool calls/results already
            normalized to tool_use/tool_result blocks. With translated-history
            caching, this is only the current incremental batch.
        allow_external_calls: Allow result IDs absent from earlier local calls,
            for Responses requests backed by previous_response_id. This trusts
            server history; it does not verify that those external calls exist.

    Raises:
        MessageValidationError: History is empty, a role or call/result ID is
            invalid, a call ID or result repeats, a result has no earlier call
            without the server-history exemption, or any local call lacks a result.

    Matching uses IDs rather than result order: parallel results may arrive in any
    order after their calls. Ordinary messages may intervene; this function does
    not enforce adjacency, turn boundaries, or tool-argument JSON Schema validity.
    The function reads messages without modifying them.
    """
    validator = HistoryValidator(allow_external_calls=allow_external_calls)
    if not messages:
        validator.finish()
    for message in dictionary_views(messages):
        validator.observe_message(message)
        if validator.error is not None:
            raise MessageValidationError(validator.error)
        for block in message.get("content", []):
            validator.observe_content(block)
            if validator.error is not None:
                raise MessageValidationError(validator.error)
    validator.finish()
