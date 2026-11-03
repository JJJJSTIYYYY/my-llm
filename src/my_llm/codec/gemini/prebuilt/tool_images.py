"""Optional Gemini tool-image relocation with explicit cache synchronization."""

from weakref import WeakSet

from my_llm.basic.semantic_loss import SemanticLossContext, SemanticLossGroup
from my_llm.codec.gemini.decoding.content import decode_part
from my_llm.codec.shared.prebuilt.tool_images import move_tool_images

_completed: WeakSet[object] = WeakSet()


def move_tool_images_to_user_message(
    draft: object, losses: SemanticLossGroup, context: SemanticLossContext
) -> None:
    """Move supported images or insert markers; cache repairs and accept other losses."""
    move_tool_images(
        draft,
        losses,
        context,
        protocol="gemini",
        reason="Gemini tool results cannot represent images",
        message_key="contents",
        completed=_completed,
        render=lambda content: {"role": "user", "parts": [decode_part(part) for part in content]},
    )
