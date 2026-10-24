"""gemini encoding: content, fields and tool helpers."""

import json
from base64 import b64encode

from my_llm.codec.shared.conversion import plain, select, text_result
from my_llm.exceptions import MessageValidationError
from my_llm.semantic_loss import originated

TYPES = {
    name.upper(): name
    for name in ("object", "array", "string", "number", "integer", "boolean", "null")
}
ALIASES = {
    "any_of": "anyOf",
    "anyOf": "anyOf",
    "defs": "$defs",
    "ref": "$ref",
    "additional_properties": "additionalProperties",
    "additionalProperties": "additionalProperties",
    "min_items": "minItems",
    "max_items": "maxItems",
    "min_length": "minLength",
    "max_length": "maxLength",
    "min_properties": "minProperties",
    "max_properties": "maxProperties",
}
COUNTS = {"minItems", "maxItems", "minLength", "maxLength", "minProperties", "maxProperties"}
SCALARS = {
    "default",
    "description",
    "enum",
    "format",
    "maximum",
    "minimum",
    "pattern",
    "required",
    "title",
}

KEYWORDS = tuple(
    dict.fromkeys(
        (
            "type",
            "properties",
            "items",
            "example",
            *sorted(SCALARS),
            *sorted(COUNTS),
            "$defs",
            "$ref",
            "anyOf",
            "additionalProperties",
            *ALIASES,
        )
    )
)


def image_part(part: dict) -> dict | None:
    """Project only recognized image media before SDK input validation."""
    for field in ("inline_data", "file_data"):
        media = part.get(field)
        if isinstance(media, dict) and str(media.get("mime_type", "")).startswith("image/"):
            return {field: select(media, "mime_type", "data", "file_uri")}
    return None


def encode_part(part: dict, call_id: str, *, path: str = "$") -> dict | None:
    common = {"signature": part["thought_signature"]} if "thought_signature" in part else {}
    if isinstance(common.get("signature"), bytes):
        common["signature"] = b64encode(common["signature"]).decode("ascii")
    if "function_call" in part:
        call = part["function_call"]
        args = call.get("args")
        return {
            **common,
            "type": "tool_use",
            "id": call.get("id") or call_id,
            "name": call.get("name"),
            "input": {} if args is None else args,
        }
    if "function_response" in part:
        call = part["function_response"]
        content = call.get("response", {})
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        media = [
            originated(
                encode_part(image, ""),
                f"{path}.function_response.parts[{index}]",
                child,
            )
            for index, child in enumerate(call.get("parts") or [])
            if isinstance(child, dict) and (image := image_part(child)) is not None
        ]
        if media:
            content = [
                originated(
                    {"type": "text", "text": text_result(content)},
                    f"{path}.function_response.response",
                    call.get("response", {}),
                ),
                *media,
            ]
        return {
            **common,
            "type": "tool_result",
            "name": call.get("name"),
            "tool_use_id": call.get("id"),
            "content": content,
        }
    if "text" in part:
        return (
            {**common, "type": "thinking", "thinking": part["text"]}
            if part.get("thought")
            else {**common, "type": "text", "text": part["text"]}
        )
    if "inline_data" in part or "file_data" in part:
        data = part.get("inline_data", part.get("file_data"))
        mime = data.get("mime_type", "application/octet-stream")
        if not mime.startswith("image/"):
            return None
        value = data.get("data")
        if isinstance(value, bytes):
            value = b64encode(value).decode("ascii")
        source = (
            {"type": "base64", "media_type": mime, "data": value}
            if "inline_data" in part
            else {"type": "url", "url": data.get("file_uri"), "media_type": mime}
        )
        return {"type": "image", "source": source}
    return None


def json_schema(schema: dict) -> dict:
    """Normalize only schema keywords; application property names remain untouched."""
    schema = schema if isinstance(schema, dict) else plain(schema)
    if not isinstance(schema, dict):
        raise MessageValidationError("Gemini parameters must be a Schema object")
    result = {}
    # Enumerate recognized keywords, never unrecognized provider extensions.
    for source_key in KEYWORDS:
        if source_key not in schema:
            continue
        value = schema[source_key]
        key = ALIASES.get(source_key, source_key)
        if key == "type":
            if value == "TYPE_UNSPECIFIED":
                continue
            if value not in TYPES:
                raise MessageValidationError(f"unsupported Gemini Schema type: {value!r}")
            result[key] = TYPES[value]
        elif key in ("properties", "$defs"):
            result[key] = {name: json_schema(child) for name, child in value.items()}
        elif key == "items" or (key == "additionalProperties" and isinstance(value, dict)):
            result[key] = json_schema(value)
        elif key == "anyOf":
            result[key] = [json_schema(child) for child in value]
        elif key in COUNTS:
            if isinstance(value, bool) or not str(value).isdigit():
                raise MessageValidationError(f"Gemini Schema {source_key} must be an integer")
            result[key] = int(value)
        elif key == "$ref":
            result[key] = value.replace("#/defs/", "#/$defs/", 1)
        elif key == "example":
            result["examples"] = [value]
        elif key in SCALARS or (key == "additionalProperties" and isinstance(value, bool)):
            result[key] = value
    if schema.get("nullable"):
        # A separate null alternative also handles enum/ref/anyOf constraints.
        return {"anyOf": [result, {"type": "null"}]}
    return result
