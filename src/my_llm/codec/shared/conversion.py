"""Open dictionaries used only inside the codec boundary.

Known fields are translated; extensions keep their original nesting. Provenance
allows exact replay in the source protocol, including signed and opaque blocks.
This is deliberately not a public message schema.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from my_llm.codec.shared.keys import OPAQUE, SOURCE, TRANSLATED
from my_llm.codec.shared.values import plain
from my_llm.exceptions import MessageValidationError, UnsupportedCapabilityError


def dictionaries(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        raise MessageValidationError("expected a list of protocol dictionaries or SDK objects")
    result = [deepcopy(plain(value)) for value in values]
    if any(not isinstance(value, dict) for value in result):
        raise MessageValidationError("expected protocol dictionaries or SDK objects")
    for value in result:
        # Translated values always have priority over native extensions.
        value.update(value.pop(TRANSLATED, {}))
    return result


def _overlay(value: Any, extension: Any) -> Any:
    """Merge trees, with the right-hand value winning conflicts."""
    if isinstance(value, dict) and isinstance(extension, dict):
        return {
            **value,
            **{
                key: _overlay(value[key], child) if key in value else deepcopy(child)
                for key, child in extension.items()
            },
        }
    if isinstance(value, list) and isinstance(extension, list):
        result = deepcopy(value)
        for index, child in enumerate(extension):
            if index < len(result):
                result[index] = _overlay(result[index], child)
            else:
                result.append(deepcopy(child))
        return result
    return deepcopy(extension)


def remember(value: dict, original: Any, protocol: str, *, preserve: dict | None = None) -> dict:
    source = {"protocol": protocol, "value": deepcopy(original)}
    if preserve:
        combined = _overlay(value, preserve)
        # A converted field is authoritative as a whole: merging native
        # collisions into it can add list elements or tool arguments.
        conflicts = {key: child for key, child in value.items() if combined[key] != child}
        if conflicts:
            # Keep the native path and use a separate, reserved key for its
            # translation. dictionaries() always reads this translation first.
            combined[TRANSLATED] = conflicts
        value = combined
    return {**value, SOURCE: source}


def original(value: dict, protocol: str) -> Any:
    source = value.get(SOURCE, {})
    # Replay is valid only for the source protocol; another target must
    # render translated fields rather than reinterpret the native snapshot.
    if source.get("protocol") == protocol:
        return deepcopy(source.get("value"))
    if value.get(OPAQUE):
        raise UnsupportedCapabilityError(
            "opaque native items cannot be replayed to another protocol"
        )
    return None


def extras(value: dict, *known: str) -> dict:
    return {k: deepcopy(v) for k, v in value.items() if k not in (*known, SOURCE, TRANSLATED)}


def merge_fields(previous: dict, update: dict) -> dict:
    """Overwrite snapshot leaves, retaining independently updated mapping fields."""
    result = dict(previous)
    for key, value in update.items():
        result[key] = (
            merge_fields(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


# None is a valid extension value, so consumption needs a distinct sentinel.
_CONSUMED = object()


def _remaining(value: Any, consumed: Any) -> Any:
    if callable(consumed):
        consumed = consumed(value)
    if consumed is True:
        return _CONSUMED
    if isinstance(value, dict) and isinstance(consumed, dict):
        result = {}
        for key, child in value.items():
            if key not in consumed:
                result[key] = deepcopy(child)
            else:
                residue = _remaining(child, consumed[key])
                if residue is not _CONSUMED:
                    result[key] = residue
        return result if result or not value else _CONSUMED
    if isinstance(value, list) and isinstance(consumed, list):
        residues = [_remaining(child, consumed[0]) for child in value]
        if not value or any(child is not _CONSUMED for child in residues):
            # Keep holes so an extension in element i stays at its source index.
            return [child if child is not _CONSUMED else {} for child in residues]
        return _CONSUMED
    return deepcopy(value)


def residual(value: dict, consumed: dict) -> dict:
    """Remove recognized leaves, retaining unknown siblings and list indices.

    True moves an entire atomic value (arguments, results or JSON schemas).
    Mappings consume declared children only; [rule] applies to list elements.
    Empty containers and null extension values are preserved.
    """
    result = _remaining(value, consumed)
    return {} if result is _CONSUMED else result


def coalesce_messages(messages: list) -> list[dict]:
    """Group adjacent assistant fragments marked as parts of one native turn."""
    result = []
    # A source may represent one assistant turn as several native items.
    # Its encoder declares whether those fragments may be grouped for rendering.
    for message in dictionaries(messages):
        if (
            result
            and message.get(SOURCE, {}).get("coalesce_assistant", False)
            and result[-1].get(SOURCE, {}).get("coalesce_assistant", False)
            and message.get("role") == result[-1].get("role") == "assistant"
            and not message.get(OPAQUE)
            and not result[-1].get(OPAQUE)
        ):
            result[-1]["content"].extend(message.get("content", []))
        else:
            result.append(message)
    return result


def blocks(content: Any) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return dictionaries(content)


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
    return value.get(SOURCE, {}).get("protocol")


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
