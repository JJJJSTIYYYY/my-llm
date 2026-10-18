"""Intermediate dictionaries, provenance keys and protocol translation mechanics.

Known fields are translated; extensions retain their nesting and native source.
This module owns message/tool projection and source grouping, without protocol
branches or SDK request validation. It defines no public application schema.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from my_llm.exceptions import MessageValidationError, UnsupportedCapabilityError

SOURCE = "__my_llm_codec_source__"
TRANSLATED = "__my_llm_codec_translated_fields__"
FUNCTIONS = "__my_llm_codec_function_definitions__"
OPAQUE = "__my_llm_codec_opaque_message__"
REPLAY = "__my_llm_codec_replay__"


def read(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value.copy() if isinstance(value, (dict, list)) else value


class Translations(list[dict]):
    """Collect output and its source groups during one decoder translation.

    A group may consume or produce any number of values. ``structure`` declares
    containers whose unknown children can be merged into translated output;
    other translated fields remain authoritative as whole values.
    Call ``group`` before appending the native values produced from those sources.
    Decoder message and tool hooks return this type; their public methods return
    ordinary lists after preserving extensions with the declared groups.
    """

    def __init__(self) -> None:
        super().__init__()
        self.groups: list[tuple[int, list[dict], dict]] = []

    def group(self, sources: list[dict], *, structure: dict | None = None) -> None:
        self.groups.append((len(self), deepcopy(sources), structure or {}))


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
    if preserve is not None:
        source["residual"] = deepcopy(preserve)
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


def coalesce_groups(messages: list) -> list[tuple[list[dict], dict]]:
    """Return original sources alongside each grouped assistant message."""
    result = []
    # A source may represent one assistant turn as several native items.
    # Its encoder declares whether those fragments may be grouped for rendering.
    for message in dictionaries(messages):
        if (
            result
            and message.get(SOURCE, {}).get("coalesce_assistant", False)
            and result[-1][1].get(SOURCE, {}).get("coalesce_assistant", False)
            and message.get("role") == result[-1][1].get("role") == "assistant"
            and not message.get(OPAQUE)
            and not result[-1][1].get(OPAQUE)
        ):
            result[-1][0].append(deepcopy(message))
            result[-1][1]["content"].extend(message.get("content", []))
        else:
            result.append(([deepcopy(message)], message))
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


def mapping_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def match_all(values, predicate) -> bool:
    try:
        items = dictionaries(values)
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
    return {**select(message, *fields, SOURCE), "content": content}


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
    return select(block, *fields[kind])


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


def encode_functions(tools, protocol, unwrap, *, consumed):
    result = []
    for tool in dictionaries(tools):
        functions = unwrap(tool)
        fields = consumed(tool) if callable(consumed) else consumed
        remaining = residual(tool, fields if functions is not None else {})
        result.append(
            remember(
                {FUNCTIONS: functions},
                tool,
                protocol,
                preserve=remaining,
            )
        )
    return result


def decode_functions(tools, protocol, wrap, *, structure=None) -> Translations:
    result = Translations()
    for tool in dictionaries(tools):
        result.group([tool], structure=structure)
        native = original(tool, protocol)
        if native is not None:
            result.append(native)
        elif tool.get(FUNCTIONS) is None:
            raise UnsupportedCapabilityError(
                "opaque native tools cannot be replayed to another protocol"
            )
        else:
            result.extend(wrap(tool[FUNCTIONS]))
    return result
