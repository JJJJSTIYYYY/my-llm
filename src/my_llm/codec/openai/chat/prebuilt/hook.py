"""Combined, explicitly enabled Chat request semantic-loss compensation."""

from my_llm.basic.semantic_loss import SemanticLossContext, SemanticLossGroup
from my_llm.codec.openai.chat.prebuilt.reasoning import reasoning_content_feedback
from my_llm.codec.openai.chat.prebuilt.tool_images import move_tool_images_to_user_message


def chat_semantic_loss_hook(
    draft: object, losses: SemanticLossGroup, context: SemanticLossContext
) -> None:
    """Apply visible-reasoning feedback and image-reference repairs to a Chat request."""
    reasoning_content_feedback(draft, losses, context)
    move_tool_images_to_user_message(draft, losses, context)
