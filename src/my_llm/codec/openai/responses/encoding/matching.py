"""Recognize this native protocol without consulting other codecs."""

from my_llm.codec.shared.matching import mapping_list, match_all


def item_matches(item: dict) -> bool:
    kind = item.get("type")
    if kind == "function":
        return "function" not in item and isinstance(item.get("name"), str)
    if kind in {"function_call", "function_call_output", "reasoning", "item_reference"}:
        return True
    if kind not in (None, "message") or item.get("role") not in (
        "user",
        "assistant",
        "system",
        "developer",
    ):
        return False
    if set(item) & {
        "parts",
        "tool_calls",
        "tool_call_id",
        "reasoning_content",
        "thinking",
        "images",
        "tool_name",
    }:
        return False
    content = item.get("content")
    return (
        isinstance(content, str)
        or content is None
        or (
            mapping_list(content)
            and all(
                block.get("type")
                in {"input_text", "output_text", "input_image", "input_file", "refusal"}
                for block in content
            )
        )
    )


def matches(values) -> bool:
    return match_all(values, item_matches)
