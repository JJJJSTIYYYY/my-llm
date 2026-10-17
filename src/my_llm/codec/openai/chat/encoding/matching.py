"""Recognize this native protocol without consulting other codecs."""

from my_llm.codec.shared.matching import mapping_list, match_all


def item_matches(item: dict) -> bool:
    if item.get("type") == "function":
        return isinstance(item.get("function"), dict)
    if item.get("role") not in ("user", "assistant", "system", "developer", "tool"):
        return False
    if set(item) & {"parts", "type", "thinking", "images", "tool_name"}:
        return False
    calls = item.get("tool_calls") or []
    if not mapping_list(calls) or any(
        not isinstance(call.get("function"), dict)
        or isinstance(call["function"].get("arguments"), dict)
        for call in calls
    ):
        return False
    content = item.get("content")
    return (
        isinstance(content, str)
        or content is None
        or (
            mapping_list(content)
            and all(
                block.get("type") in {"text", "image_url", "file", "input_audio", "refusal"}
                for block in content
            )
        )
    )


def matches(values) -> bool:
    return match_all(values, item_matches)
