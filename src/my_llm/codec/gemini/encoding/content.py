"""gemini encoding: content, fields and tool helpers."""

from copy import deepcopy

from my_llm.codec.shared.conversion import (
    extras,
    residual,
)
from my_llm.exceptions import MessageValidationError

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


def encode_part(part: dict, call_id: str) -> dict:
    if "function_call" in part:
        call = part["function_call"]
        return {
            **residual(part, part_fields(part)),
            "type": "tool_use",
            "id": call.get("id") or call_id,
            "name": call.get("name"),
            "input": call.get("args", {}),
        }
    if "function_response" in part:
        call = part["function_response"]
        return {
            **residual(part, part_fields(part)),
            "type": "tool_result",
            "name": call.get("name"),
            "tool_use_id": call.get("id"),
            "content": call.get("response", {}),
        }
    if "text" in part:
        if part.get("thought"):
            return {**extras(part, "text", "thought"), "type": "thinking", "thinking": part["text"]}
        return {**extras(part, "thought"), "type": "text"}
    if "inline_data" in part or "file_data" in part:
        data = part.get("inline_data", part.get("file_data"))
        mime = data.get("mime_type", "application/octet-stream")
        source = (
            {"type": "base64", "media_type": mime, "data": data.get("data")}
            if "inline_data" in part
            else {"type": "url", "url": data.get("file_uri"), "media_type": mime}
        )
        return {
            **residual(part, part_fields(part)),
            "type": "image" if mime.startswith("image/") else "document",
            "source": source,
        }
    return part


def part_fields(part: dict) -> dict:
    if "function_call" in part:
        return {"function_call": {"id": True, "name": True, "args": True}}
    if "function_response" in part:
        return {"function_response": {"id": True, "name": True, "response": True}}
    if "text" in part:
        return {"text": True, "thought": True}
    if "inline_data" in part:
        return {"inline_data": {"mime_type": True, "data": True}}
    if "file_data" in part:
        return {"file_data": {"mime_type": True, "file_uri": True}}
    return {}


def json_schema(schema: dict) -> dict:
    """Normalize only schema keywords; application property names remain untouched."""
    if not isinstance(schema, dict):
        raise MessageValidationError("Gemini parameters must be a Schema object")
    result = {}
    for source_key, value in schema.items():
        key = ALIASES.get(source_key, source_key)
        if key in ("nullable", "property_ordering", "propertyOrdering"):
            continue
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
            result["examples"] = [deepcopy(value)]
        elif key in SCALARS or (key == "additionalProperties" and isinstance(value, bool)):
            result[key] = deepcopy(value)
        else:
            raise MessageValidationError(f"unsupported Gemini Schema keyword: {source_key}")
    if schema.get("nullable"):
        # A separate null alternative also handles enum/ref/anyOf constraints.
        return {"anyOf": [result, {"type": "null"}]}
    return result
