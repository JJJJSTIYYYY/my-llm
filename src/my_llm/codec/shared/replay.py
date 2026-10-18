"""Preserve conversion leftovers and restore unchanged protocol histories."""

from collections.abc import Sequence
from copy import deepcopy

from my_llm.codec.shared.conversion import (
    FUNCTIONS,
    OPAQUE,
    REPLAY,
    SOURCE,
    dictionaries,
    extras,
    residual,
    source_protocol,
)


def _leftovers(value):
    stored = value.get(SOURCE, {}).get("residual")
    if stored is not None:
        return stored
    fields = {
        "text": {"type", "text"},
        "image": {"type", "source", "detail"},
        "document": {"type", "source"},
        "tool_use": {"type", "id", "name", "input"},
        "tool_result": {"type", "tool_use_id", "name", "content", "is_error"},
        "refusal": {"type", "refusal"},
        "thinking": {"type", "thinking"},
        "redacted_thinking": {"type"},
    }

    def consumed(block):
        result = dict.fromkeys(fields.get(block.get("type"), ()), True)
        if "source" in result:
            result["source"] = dict.fromkeys(("type", "url", "media_type", "data"), True)
        return result

    value = dictionaries([value])[0]
    return residual(extras(value, FUNCTIONS, OPAQUE), {"role": True, "content": [consumed]})


def _has_leftovers(values):
    for value in values:
        if _leftovers(value):
            return True
        for block in value.get("content", []) if isinstance(value.get("content"), list) else []:
            if block.get("type") in ("thinking", "redacted_thinking"):
                return True
    return False


def merge_extensions(extension, converted, structure=None, *, positional=True):
    """Merge declared containers without guessing ambiguous list positions."""
    structure = {} if structure is None else structure
    if isinstance(extension, dict) and isinstance(converted, dict) and isinstance(structure, dict):
        return {
            **deepcopy(extension),
            **{
                key: merge_extensions(extension[key], value, structure[key], positional=positional)
                if key in extension and key in structure
                else deepcopy(value)
                for key, value in converted.items()
            },
        }
    if (
        isinstance(extension, list)
        and isinstance(converted, list)
        and isinstance(structure, list)
        and structure
        and positional
        and len(extension) == len(converted)
    ):
        return [
            merge_extensions(extension[index], value, structure[0], positional=positional)
            for index, value in enumerate(converted)
        ]
    return deepcopy(converted)


def retain(
    values,
    converted,
    protocol,
    kind,
    *,
    groups: Sequence[tuple[int, list[dict], dict]],
    force=False,
):
    """Attach one JSON-safe recovery record to a converted group.

    The record stores both the original intermediate values and the exact output
    they describe. Encoders restore it only when that output is unchanged.
    """
    converted = list(converted)
    if not converted or not values:
        return converted
    if all(source_protocol(value) == protocol for value in values):
        return converted
    if not force and not _has_leftovers(values):
        return converted
    canonical = deepcopy(converted)
    result = deepcopy(converted)
    # Only decoders know which source values produced an output group. Never
    # infer that association from matching positions or list lengths.
    for position, (start, sources, structure) in enumerate(groups):
        end = groups[position + 1][0] if position + 1 < len(groups) else len(result)
        if start == end:
            continue
        for source in sources:
            result[start] = merge_extensions(
                _leftovers(source),
                result[start],
                structure,
                positional=len(sources) == end - start == 1,
            )
    result[0][REPLAY] = {
        "protocol": protocol,
        "kind": kind,
        "values": deepcopy(values),
        "canonical": canonical,
        "output": deepcopy(result),
    }
    return result


def replay_inputs(values, protocol, kind=None):
    """Return translated input copies and unchanged groups eligible for recovery."""
    values = dictionaries(values)
    clean = deepcopy(values)
    restored = {}
    index = 0
    while index < len(values):
        record = clean[index].pop(REPLAY, None)
        if (
            not isinstance(record, dict)
            or record.get("protocol") != protocol
            or record.get("kind") not in ((kind,) if kind else ("messages", "tools"))
        ):
            index += 1
            continue
        output = record.get("output")
        canonical = record.get("canonical")
        original = record.get("values")
        if not (
            isinstance(output, list)
            and output
            and isinstance(canonical, list)
            and len(canonical) == len(output)
            and isinstance(original, list)
            and all(isinstance(item, dict) for item in [*output, *canonical, *original])
        ):
            index += 1
            continue
        count = len(output)
        observed = deepcopy(values[index : index + count])
        observed[0].pop(REPLAY, None)
        if observed == output:
            clean[index : index + count] = deepcopy(canonical)
            restored[index] = (count, original)
            index += count
        else:
            # An edit invalidates the snapshot. Keep the edited target values;
            # never replace them with stale signed or encrypted source data.
            index += 1
    return clean, restored


def _calls(values):
    return [
        block["id"]
        for value in dictionaries(values)
        for block in value.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def encode_retained(encode, values, protocol, kind, profile=None, **parameters):
    """Encode one batch, then recover unchanged groups and their tool identities.

    Encoding together keeps ID-less tool association across group boundaries.
    Newly appended messages are encoded normally.
    """
    clean, restored = replay_inputs(values, protocol, kind)
    encoded = encode(clean, profile, **parameters)
    result = []
    call_ids = {}
    index = 0
    while index < len(encoded):
        if index in restored:
            count, original = restored[index]
            if kind == "messages":
                call_ids.update(zip(_calls(encoded[index : index + count]), _calls(original)))
            result.extend(deepcopy(original))
            index += count
        else:
            value = encoded[index]
            for block in value.get("content", []) if isinstance(value.get("content"), list) else []:
                call_id = block.get("tool_use_id")
                if call_id in call_ids and call_ids[call_id] != call_id:
                    block["tool_use_id"] = call_ids[call_id]
                    value.get(SOURCE, {}).pop("value", None)
            result.append(value)
            index += 1
    return result
