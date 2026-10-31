"""Select the fixed supported Anthropic content fields."""

from my_llm.basic.semantic_loss import originated
from my_llm.codec.shared.conversion import select, tool_result_content

FIELDS = {
    "text": ("text",),
    "thinking": ("thinking", "signature"),
    "redacted_thinking": ("data",),
    "tool_use": ("id", "name", "input"),
    "tool_result": ("tool_use_id",),
    "image": (),
}
TOOL_RESULT_PARTS = frozenset(
    (*FIELDS, "document", "audio", "input_audio", "server_tool_use", "web_search_tool_result")
)


def native_block(block: dict, *, path: str | None = None) -> dict | None:
    """Project a supported Anthropic block, recursively restricting tool-result parts."""
    kind = block.get("type")
    fields = FIELDS.get(kind)
    if fields is None:
        return None
    value = select(block, "type", *fields)
    if kind == "image":
        source = block.get("source", {})
        if source.get("type") not in ("url", "base64"):
            return None
        value["source"] = select(source, "type", "media_type", "data", "url")
    elif kind == "tool_result":
        content = tool_result_content(block.get("content"), native_types=TOOL_RESULT_PARTS)
        if isinstance(content, list):
            value["content"] = [
                child
                for index, item in enumerate(content)
                if (child := native_block(item, path=f"{path}.content[{index}]" if path else None))
                is not None
                and child.get("type") in ("text", "image")
            ]
        else:
            value["content"] = content
    return originated(value, path, block) if path is not None else value


def intermediate_block(block: dict, *, path: str | None = None) -> dict | None:
    """Normalize Anthropic reasoning and supported native blocks into canonical content."""
    value = native_block(block, path=path)
    if value is None:
        return None
    if value["type"] == "redacted_thinking":
        value = {
            "type": "thinking",
            "thinking": "",
            "mark": "encrypted",
            "redacted": True,
            "encrypted_content": value["data"],
            "protocol": "anthropic",
        }
        if path is not None:
            value = originated(value, path, block)
    elif value["type"] == "thinking":
        value["mark"] = "raw"
    return value
