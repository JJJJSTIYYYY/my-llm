"""Explicit Responses response settings, separate from generated content."""

from copy import deepcopy

from my_llm.codec.shared.conversion import select


def response_parameters(value: dict) -> dict:
    """Retain recognized settings; metadata and JSON Schema are open objects."""
    result = select(value, "temperature", "top_p", "parallel_tool_calls", "metadata")
    if "instructions" in value:
        instructions = value["instructions"]
        result["instructions"] = (
            [projected for item in instructions if (projected := _instruction(item)) is not None]
            if isinstance(instructions, list)
            else deepcopy(instructions)
        )
    if "tools" in value:
        result["tools"] = [
            projected for tool in value["tools"] if (projected := _tool(tool)) is not None
        ]
    if "tool_choice" in value:
        result["tool_choice"] = _choice(value["tool_choice"])
    return result


def _tool(value: dict) -> dict | None:
    if value.get("type") != "function":
        return None
    result = select(
        value,
        "type",
        "name",
        "description",
        "parameters",
        "strict",
        "defer_loading",
        "output_schema",
    )
    if "allowed_callers" in value:
        result["allowed_callers"] = [
            caller
            for caller in value["allowed_callers"] or []
            if caller in ("direct", "programmatic")
        ]
    return result


def _choice(value):
    if not isinstance(value, dict):
        return deepcopy(value)
    result = select(value, "type", "name", "mode", "server_label")
    if "tools" in value:
        result["tools"] = [select(tool, "type", "name", "server_label") for tool in value["tools"]]
    return result


def _instruction(value: dict) -> dict | None:
    kind = value.get("type", "message")
    if kind == "message":
        content = value.get("content", "")
        return {
            **select(value, "type", "role", "id", "status"),
            "content": _instruction_content(content),
        }
    if kind == "function_call":
        return select(value, "type", "id", "status", "call_id", "name", "arguments")
    if kind == "function_call_output":
        return {
            **select(value, "type", "id", "status", "call_id"),
            "output": _instruction_content(value.get("output", "")),
        }
    if kind == "reasoning":
        return {
            **select(value, "type", "id", "status", "encrypted_content"),
            "summary": [select(part, "type", "text") for part in value.get("summary", [])],
        }
    if kind == "item_reference":
        return select(value, "type", "id")
    return None


def _instruction_content(value):
    if not isinstance(value, list):
        return deepcopy(value)
    fields = {
        "input_text": ("text",),
        "output_text": ("text",),
        "refusal": ("refusal",),
        "input_image": ("image_url", "file_id", "detail"),
        "input_file": ("file_data", "file_url", "file_id", "filename"),
    }
    return [
        select(part, "type", *fields[part["type"]]) for part in value if part.get("type") in fields
    ]
