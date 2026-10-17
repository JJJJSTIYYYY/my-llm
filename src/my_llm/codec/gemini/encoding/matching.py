"""Recognize this native protocol without consulting other codecs."""

from my_llm.codec.shared.matching import mapping_list, match_all


def item_matches(item: dict) -> bool:
    if "function_declarations" in item:
        return mapping_list(item["function_declarations"])
    return item.get("role", "user") in ("user", "model", "system", "developer") and mapping_list(
        item.get("parts")
    )


def matches(values) -> bool:
    return match_all(values, item_matches)
