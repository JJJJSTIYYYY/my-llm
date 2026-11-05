"""Use visible reasoning when Chat cannot replay a native feedback item."""

from weakref import WeakSet

from my_llm.basic.data import read
from my_llm.basic.semantic_loss import SemanticLossContext, SemanticLossGroup
from my_llm.codec.openai.chat.prebuilt.shared import publish_messages, request_messages
from my_llm.codec.shared.capabilities import supports_content

_completed: WeakSet[object] = WeakSet()


def reasoning_content_feedback(
    draft: object, losses: SemanticLossGroup, context: SemanticLossContext
) -> None:
    """Replace lost replay data with this assistant message's visible reasoning."""
    original = request_messages(draft, context, _completed)
    if original is None or not supports_content("thinking", context.profile):
        return
    messages = list(original)
    repaired = set()
    for loss in losses:
        if loss.reason not in (
            "target protocol cannot preserve another protocol's reasoning feedback_item",
            "reasoning feedback_item must be text",
            "target protocol cannot preserve feedback_item alongside visible reasoning",
        ):
            continue
        target = loss.target
        if target is None or target.field != "reasoning_content":
            continue
        index = target.message_index
        if (
            index is None
            or index in repaired
            or not 0 <= index < len(messages)
            or messages[index].get("role") != "assistant"
        ):
            continue
        parts = [
            read(block, "reasoning")
            for block in read(target.source_value, "content", [])
            if read(block, "type") == "thinking" and isinstance(read(block, "reasoning"), str)
        ]
        if parts:
            messages[index] = {**messages[index], "reasoning_content": "".join(parts)}
            repaired.add(index)
    if repaired:
        publish_messages(draft, messages, context, _completed)
