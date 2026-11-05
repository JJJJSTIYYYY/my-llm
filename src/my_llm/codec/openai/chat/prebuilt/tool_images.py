"""Move supported tool-result image losses into an explicitly cached user message."""

from uuid import uuid4
from weakref import WeakSet

from my_llm.basic.data import read
from my_llm.basic.semantic_loss import SemanticLossContext, SemanticLossGroup
from my_llm.codec.openai.chat.decoding.content import render_content
from my_llm.codec.openai.chat.prebuilt.shared import publish_messages, request_messages
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
    """Replace tool images with IDs and replay each image beside its ID as user input."""
    original = request_messages(draft, context, _completed)
    if original is None:
        return
    vision = supports_content("image", context.profile)
    messages = list(original)
    content = []
    repaired = set()
    for loss in losses:
        if loss.reason not in (
            "Chat tool results cannot represent images",
            "model does not support image input",
        ):
            continue
        target = loss.target
        if target is not None and target.field == "content":
            index = target.message_index
            if (
                index is None
                or index in repaired
                or not 0 <= index < len(messages)
                or messages[index].get("role") != "tool"
            ):
                continue
            parts = []
            for child in read(target.source_value, "content", []):
                if read(child, "type") == "text":
                    parts.append(read(child, "text", ""))
                elif read(child, "type") == "image":
                    reference = f"[Image: image_{uuid4().hex}]"
                    parts.append(reference)
                    content.append({"type": "text", "text": reference + "\n"})
                    image = _image(child) if vision else None
                    content.append(
                        image
                        if image is not None
                        else {"type": "text", "text": "[Unsupported Image]\n"}
                    )
            messages[index] = {**messages[index], "content": "".join(parts)}
            repaired.add(index)
        else:
            image = _image(loss.origin_value) if vision else None
            content.append(
                image if image is not None else {"type": "text", "text": "[Unsupported Image]"}
            )
    if not content:
        return
    messages.append({"role": "user", "content": render_content(content)})
    publish_messages(draft, messages, context, _completed)
