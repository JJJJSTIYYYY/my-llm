"""Use visible reasoning when Chat cannot replay a native feedback item."""

from weakref import WeakSet

from my_llm.basic.semantic_loss import SemanticLossContext, SemanticLossGroup
from my_llm.codec.openai.chat.prebuilt.shared import publish_messages, request_messages
from my_llm.codec.shared.capabilities import supports_content
from my_llm.utils.data import read

_completed: WeakSet[object] = WeakSet()


def reasoning_content_feedback(
    draft: object, losses: SemanticLossGroup, context: SemanticLossContext
) -> None:
    """Use visible reasoning when a Chat request cannot replay native feedback.

    Register this synchronous callback as ``input_semantic_loss_hook`` on a
    ``MyLLM`` instance with an ``openai_chat`` upstream and
    ``semantic_loss_protocol="hook"``. Use ``chat_semantic_loss_hook`` instead
    when tool-result image compensation is also required.

    Handles decoder losses for foreign reasoning feedback, non-text native
    feedback, or feedback that cannot coexist with visible reasoning. For each
    affected assistant message, it concatenates string-valued
    ``ReasoningContent.reasoning`` blocks in source order, without separators,
    and writes the result to that message's native ``reasoning_content`` field.
    The loss target identifies the message in the complete request history,
    including cached prefixes and any decoder reordering or grouping.

    Args:
        draft: Complete native SDK request dictionary with a ``messages`` list.
            The callback replaces that list and the affected assistant
            dictionaries, retaining their other fields and leaving application
            source messages unchanged.
        losses: Request losses to inspect. A matching loss must target
            ``reasoning_content`` on an assistant message and supply its
            canonical source message through ``loss.target.source_value``.
            Multiple matching losses for one message produce one repair.
        context: Pipeline context identifying an input request targeting
            ``openai_chat``. Its resolved ``profile`` determines reasoning
            support, and its optional ``message_cache`` receives repaired
            history through ``replace(messages)``.

    Returns:
        None. Repairs are applied to ``draft`` in place. No change is made for
        other protocols or stages, unsupported draft shapes, disabled model
        reasoning support, or losses without a matching assistant target and
        string-valued visible reasoning. Unrelated losses are left unrepaired
        but accepted by the ``hook`` policy without a ``SemanticLossError``.

    Notes:
        Only visible reasoning is used: signatures, encrypted feedback and
        redacted data are never converted into text. The decoder's existing
        ``should_feedback_reasoning_item`` policy still controls which history
        produces feedback losses; this callback does not enable excluded
        feedback or change the model's reasoning configuration.

        With caching enabled, repaired history is explicitly saved and the
        callback skips the same cached input batch on consecutive retries.
        Later input batches can be repaired independently. Cache publication
        errors propagate to the caller, and a failed publication does not mark
        that batch as completed.

    Example:
        Enable reasoning compensation on an existing Chat-backed instance::

            from my_llm.codec.openai.chat.prebuilt import reasoning_content_feedback

            llm.set_semantic_loss_policy(
                semantic_loss_protocol="hook",
                input_semantic_loss_hook=reasoning_content_feedback,
            )
    """
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
