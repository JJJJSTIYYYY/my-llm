"""Translate content blocks only when their shape differs from the target API."""

from typing import Any

from my_llm.codec.utils.common import RESPONSE_BLOCKS
from my_llm.profile import APIType
from my_llm.types.exceptions import MessageValidationError


def encode_content(content: Any, role: str, api_type: APIType) -> Any:
    if not isinstance(content, list):
        return content
    converted = [encode_content_block(block, role, api_type) for block in content]
    if (
        api_type == "chat_completions"
        and role == "tool"
        and any(block.get("type") != "text" for block in converted)
    ):
        raise MessageValidationError("Chat tool results can only contain text blocks")
    return content if all(new is old for new, old in zip(converted, content)) else converted


def encode_content_block(block: dict[str, Any], role: str, api_type: APIType) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise MessageValidationError("content blocks must be dictionaries")
    kind = block.get("type")
    if kind in RESPONSE_BLOCKS:
        _validate_response_block(block)
        return block if api_type == "responses" else _response_block_to_chat(block)
    if kind == "refusal":
        if role != "assistant" or not isinstance(block.get("refusal"), str):
            raise MessageValidationError("refusal blocks require assistant refusal text")
        return block
    if api_type == "chat_completions":
        return block
    if kind == "text":
        if not isinstance(block.get("text"), str):
            raise MessageValidationError("text content blocks require text")
        return {
            "type": "output_text" if role == "assistant" else "input_text",
            "text": block["text"],
            **({"annotations": []} if role == "assistant" else {}),
        }
    if kind == "image_url" and role == "user":
        return _chat_image_to_response(block)
    if kind == "file" and role == "user":
        return _chat_file_to_response(block)
    raise MessageValidationError(f"unsupported Responses content block: {kind!r}")


def _validate_response_block(block: dict[str, Any]) -> None:
    kind = block["type"]
    if kind in ("input_text", "output_text"):
        if not isinstance(block.get("text"), str):
            raise MessageValidationError("text content blocks require text")
    elif kind == "input_image":
        if not any(
            isinstance(block.get(key), str) and block[key] for key in ("image_url", "file_id")
        ):
            raise MessageValidationError("input_image requires image_url or file_id")
    elif kind == "input_file":
        if not any(
            isinstance(block.get(key), str) and block[key]
            for key in ("file_data", "file_id", "file_url")
        ):
            raise MessageValidationError("input_file requires file_data, file_id or file_url")


def _response_block_to_chat(block: dict[str, Any]) -> dict[str, Any]:
    kind = block["type"]
    if kind in ("input_text", "output_text"):
        return {"type": "text", "text": block["text"]}
    if kind == "input_image":
        if not block.get("image_url"):
            raise MessageValidationError("Chat cannot represent input_image with only file_id")
        image = {"url": block["image_url"]}
        if "detail" in block:
            image["detail"] = block["detail"]
        return {"type": "image_url", "image_url": image}
    file = {key: block[key] for key in ("file_data", "file_id", "filename") if key in block}
    if not file.get("file_data") and not file.get("file_id"):
        raise MessageValidationError("Chat cannot represent input_file with only file_url")
    return {"type": "file", "file": file}


def _chat_image_to_response(block: dict[str, Any]) -> dict[str, Any]:
    image = block.get("image_url", {})
    if not isinstance(image, dict) or not isinstance(image.get("url"), str):
        raise MessageValidationError("image_url needs a URL")
    return {"type": "input_image", "image_url": image["url"], "detail": image.get("detail", "auto")}


def _chat_file_to_response(block: dict[str, Any]) -> dict[str, Any]:
    file = block.get("file")
    if not isinstance(file, dict):
        raise MessageValidationError("file content blocks require a file object")
    converted = {"type": "input_file", **file}
    _validate_response_block(converted)
    return converted
