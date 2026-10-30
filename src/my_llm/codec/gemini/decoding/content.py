"""Render supported Gemini parts without inventing thought signatures."""

from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.shared.conversion import text_result
from my_llm.codec.shared.input import input_content
from my_llm.profile.resolution import ResolvedProfile


def decode_part(
    block: dict, *, foreign: bool = False, profile: ResolvedProfile | None = None
) -> dict | None:
    kind = block.get("type")
    owner = block.get("protocol")
    native_signature = not foreign if owner is None else owner == "gemini"
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
        if block.get("redacted") or block.get("mark") == "encrypted":
            report_loss(block, "Gemini cannot represent opaque reasoning from another protocol")
            return None
        if not native_signature or not block.get("signature"):
            report_loss(
                block, "Gemini cannot replay reasoning without its native thought signature"
            )
            return None
        return {
            **common,
            "text": block["thinking"] or "\n".join(block.get("summary", ())),
            "thought": True,
        }
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
