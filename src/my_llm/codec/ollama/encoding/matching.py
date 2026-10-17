"""Recognize this native protocol without consulting other codecs."""

from my_llm.codec.shared.matching import mapping_list, match_all


def item_matches(item: dict) -> bool:
    if item.get("type") == "function":
        return isinstance(item.get("function"), dict)
    if item.get("role") not in ("user", "assistant", "system", "tool"):
        return False
    if set(item) & {"parts", "type", "reasoning_content"}:
        return False
    calls = item.get("tool_calls") or []
    if not mapping_list(calls) or any(
        not isinstance(call.get("function"), dict)
        or (
            call["function"].get("arguments") is not None
            and not isinstance(call["function"].get("arguments"), dict)
        )
        for call in calls
    ):
        return False
    return isinstance(item.get("content"), str) or item.get("content") is None


def matches(values) -> bool:
    return match_all(values, item_matches)
