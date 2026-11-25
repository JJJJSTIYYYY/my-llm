"""Move supported tool-result image losses into an explicitly cached user message."""

import json
from hashlib import sha256
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
    """Move lost tool-result images into a trailing Chat user message.

    Register this synchronous callback as ``input_semantic_loss_hook`` on a
    ``MyLLM`` instance with an ``openai_chat`` upstream and
    ``semantic_loss_protocol="hook"``. Use ``chat_semantic_loss_hook`` when
    reasoning-feedback compensation is also required.

    Handles ``Chat tool results cannot represent images`` and
    ``model does not support image input`` losses. For a loss targeting a tool
    message's ``content``, it rebuilds that content from the source tool result:
    text stays in order and each image becomes ``[Image: image_<sha256>]``.
    Hash the UTF-8 compact, key-sorted JSON of its non-None source fields:
    ``type``, ``url``, ``data`` and ``media_type``. URL sources use their address;
    Base64 sources use their encoded data and media type, without fetching or
    decoding image bytes. The same reference precedes the image in one new user
    message appended after the existing history. Tool call IDs, other message fields
    and message order are preserved. Repeated image sources reuse their hash;
    each occurrence is still included in the appended message, and
    each targeted tool result is rebuilt only once per invocation.

    Args:
        draft: Complete native SDK request dictionary with a ``messages`` list.
            The callback replaces that list and affected tool dictionaries,
            then appends at most one user message. It does not mutate the
            application's source messages or other request fields.
        losses: Request losses in reporting order. Tool-content targets supply
            the canonical tool result through ``loss.target.source_value`` and
            its position through ``loss.target.message_index``. For matching
            losses without a tool-content target, ``loss.origin_value`` supplies
            an image or placeholder for the appended user message, without
            adding a tool-result reference.
        context: Pipeline context identifying an input request targeting
            ``openai_chat``. The resolved ``supports_vision`` capability in its
            ``profile`` controls whether images may be sent. Its optional
            ``message_cache`` receives the complete repaired history through
            ``replace(messages)``.

    Returns:
        None. Repairs are applied to ``draft`` in place. Other protocols or
        stages, unsupported draft shapes and unrelated losses produce no
        changes. Under the ``hook`` policy, unrelated losses are accepted
        without a ``SemanticLossError``, even though they remain unrepaired.

    Notes:
        URL and Base64 image sources are supported. When vision is disabled or
        an image source cannot be represented, its position in the appended
        user message contains ``[Unsupported Image]`` instead. Image sources
        may come from middleware models or supported Anthropic/Responses
        dictionaries and SDK objects. Images are collected in loss-reporting
        order and source order within each targeted tool result. If there are
        no image or placeholder parts to append, the draft is left unchanged.

        With caching enabled, repaired tool text, the appended user message and
        image IDs are explicitly saved. Consecutive retries of the same cached
        input batch reuse them without appending another message. Hash references
        remain stable across batches and requests, including without caching.
        The appended image message still follows the complete current history.
        Cache publication errors propagate to the caller; failed publication
        does not mark the batch as completed.
        This callback does not compensate offline conversions or output losses.

    Example:
        Enable image compensation on an existing Chat-backed instance::

            from my_llm.codec.openai.chat.prebuilt import move_tool_images_to_user_message

            llm.set_semantic_loss_policy(
                semantic_loss_protocol="hook",
                input_semantic_loss_hook=move_tool_images_to_user_message,
            )
    """
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
                    source = read(child, "source")
                    fields = {
                        key: value
                        for key in ("type", "url", "data", "media_type")
                        if (value := read(source, key)) is not None
                    }
                    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"))
                    reference = f"[Image: image_{sha256(encoded.encode('utf-8')).hexdigest()}]"
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
