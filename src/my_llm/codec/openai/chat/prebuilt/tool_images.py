"""Optional Chat tool-image relocation with explicit cache synchronization."""

from weakref import WeakSet

import my_llm.codec.openai.chat.constants as constants
from my_llm.basic.semantic_loss import SemanticLossContext, SemanticLossGroup
from my_llm.codec.openai.chat.decoding.content import render_content
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
        protocol=constants.PROTOCOL,
        reason="Chat tool results cannot represent images",
        message_key="messages",
        completed=_completed,
        render=lambda content: {"role": "user", "content": render_content(content)},
    )
