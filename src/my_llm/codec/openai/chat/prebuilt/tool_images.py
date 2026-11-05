"""Move supported tool-result image losses into an explicitly cached user message."""

from weakref import WeakSet

import my_llm.codec.openai.chat.constants as constants
from my_llm.basic.data import read
from my_llm.basic.semantic_loss import SemanticLossContext, SemanticLossGroup
from my_llm.codec.openai.chat.decoding.content import render_content
from my_llm.codec.shared.capabilities import supports_content
from my_llm.codec.shared.conversion import uri_source
from my_llm.middleware.models import ImageContent


def _image(value: object) -> dict | None:
    """Read only image fields from middleware, native dictionaries or SDK models."""
    if isinstance(value, ImageContent):
        source = value.source.to_wire()
    elif read(value, "type") == "image":
        original = read(value, "source")
        source = {
            key: item
            for key in ("type", "url", "data", "media_type")
            if (item := read(original, key)) is not None
        }
    elif read(value, "type") == "input_image":
        uri = read(value, "image_url")
        if not isinstance(uri, str):
            return None
        source = uri_source(uri)
    else:
        return None
    if source.get("type") == "url":
        valid = isinstance(source.get("url"), str) and bool(source["url"])
    else:
        valid = (
            source.get("type") == "base64"
            and isinstance(source.get("data"), str)
            and isinstance(source.get("media_type"), str)
            and bool(source["media_type"])
        )
    return {"type": "image", "source": source} if valid else None


_completed: WeakSet[object] = WeakSet()


def move_tool_images_to_user_message(
    draft: object,
    losses: SemanticLossGroup,
    context: SemanticLossContext,
) -> None:
    """Move images or insert unsupported markers; cache each repair once."""
    if (
        context.origin_source != "input"
        or context.target_protocol != constants.PROTOCOL
        or context.stage != "request"
        or not isinstance(draft, dict)
        or not isinstance(draft.get("messages"), list)
        or context.message_cache is not None
        and context.message_cache.identity in _completed
    ):
        return
    vision = supports_content("image", context.profile)
    content = []
    for loss in losses:
        if loss.reason not in (
            "Chat tool results cannot represent images",
            "model does not support image input",
        ):
            continue
        image = _image(loss.origin_value) if vision else None
        content.append(
            image if image is not None else {"type": "text", "text": "[Unsupported Image]"}
        )
    if not content:
        return
    message = {"role": "user", "content": render_content(content)}
    cache = context.message_cache
    if cache is not None:
        cache.extend([message])
        _completed.add(cache.identity)
    draft["messages"].append(message)
