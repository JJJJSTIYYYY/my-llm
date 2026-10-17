"""Recognize this native protocol without consulting other codecs."""

from my_llm.codec.shared.matching import mapping_list, match_all


def item_matches(item: dict) -> bool:
    if "input_schema" in item:
        return isinstance(item["input_schema"], dict) and isinstance(item.get("name"), str)
    if item.get("role") not in ("user", "assistant", "system") or "content" not in item:
        return False
    if set(item) & {
        "parts",
        "type",
        "tool_calls",
        "tool_call_id",
        "reasoning_content",
        "thinking",
        "images",
        "tool_name",
    }:
        return False
    content = item["content"]
    return (
        isinstance(content, str)
        or content is None
        or (
            mapping_list(content)
            and all(
                block.get("type")
                in {
                    "text",
                    "thinking",
                    "redacted_thinking",
                    "tool_use",
                    "tool_result",
                    "image",
                    "document",
                    "server_tool_use",
                    "web_search_tool_result",
                    "web_fetch_tool_result",
                    "code_execution_tool_result",
                }
                for block in content
            )
        )
    )


def matches(values) -> bool:
    return match_all(values, item_matches)
