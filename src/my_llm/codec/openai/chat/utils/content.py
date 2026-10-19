"""Recognized Chat response annotations shared by both translation directions."""

from my_llm.codec.shared.conversion import select


def annotations(values: list[dict]) -> list[dict]:
    """Render the recognized URL citation shape without provider extensions."""
    return [
        {
            "type": "url_citation",
            "url_citation": select(
                item.get("url_citation", item), "start_index", "end_index", "title", "url"
            ),
        }
        for item in values
        if item.get("type") == "url_citation"
    ]


def text_blocks(values: list[dict]) -> list[dict]:
    """A Chat tool result list contains text parts, rather than open objects."""
    return [select(value, "type", "text") for value in values if value.get("type") == "text"]
