"""Translate Gemini Schema constraints into ordinary JSON Schema."""

from copy import deepcopy

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
