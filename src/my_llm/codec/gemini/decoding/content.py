"""Render supported Gemini parts without inventing thought signatures."""

import my_llm.codec.gemini.constants as constants
from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.shared.conversion import text_result
from my_llm.codec.shared.input import input_content, without_feedback
from my_llm.profile.resolution import ResolvedProfile


def decode_part(
    block: dict, *, foreign: bool = False, profile: ResolvedProfile | None = None
) -> dict | None:
    """Render one canonical block as a Gemini part or record its semantic loss.

    Preserve thought signatures only for Gemini-owned content. Tool results
    require text/object responses; images inside them are reported as losses.
    Request history replays native feedback items. Output can render visible
    reasoning independently; foreign feedback is discarded with a recorded loss.
    """
    kind = block.get("type")
    owner = block.get("protocol")
    native_signature = not foreign if owner is None else owner == constants.PROTOCOL
    common = (
        {"thought_signature": block["signature"]}
        if native_signature and "signature" in block
        else {}
    )
    if kind == "tool_use":
        return {
            **common,
            "function_call": {"id": block["id"], "name": block["name"], "args": block["input"]},
        }
    if kind == "tool_result":
        content = block.get("content")
        if isinstance(content, list):
            retained = []
            for child in content:
                if not input_content(child, profile):
                    continue
                if child.get("type") == "image":
                    report_loss(child, "Gemini tool results cannot represent images")
                elif child.get("type") == "text":
                    retained.append(child)
            content = text_result(retained)
        elif not isinstance(content, dict):
            content = text_result(content)
        return {
            **common,
            "function_response": {
                "id": block["tool_use_id"],
                "name": block.get("name"),
                "response": content if isinstance(content, dict) else {"result": content},
            },
        }
    if kind == "thinking":
        item = block.get("feedback_item")
        if foreign:
            block = without_feedback(block)
            item = None
        if not isinstance(item, dict):
            if item is not None:
                report_loss(block, "Gemini reasoning requires a native feedback item")
            return (
                {"text": block["reasoning"], "thought": True}
                if profile is None and block.get("reasoning") is not None
                else None
            )
        return item
    if kind == "text":
        return {**common, "text": block.get("text", "")}
    if kind == "image":
        source = block.get("source", {})
        data = (
            {"data": source.get("data"), "mime_type": source.get("media_type")}
            if source.get("type") == "base64"
            else {
                "file_uri": source.get("url"),
                **({"mime_type": source["media_type"]} if source.get("media_type") else {}),
            }
        )
        return {"inline_data" if source.get("type") == "base64" else "file_data": data}
    return None
