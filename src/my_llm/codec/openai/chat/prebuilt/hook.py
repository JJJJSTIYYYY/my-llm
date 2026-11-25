"""Combined, explicitly enabled Chat request semantic-loss compensation."""

from my_llm.basic.semantic_loss import SemanticLossContext, SemanticLossGroup
from my_llm.codec.openai.chat.prebuilt.reasoning import reasoning_content_feedback
from my_llm.codec.openai.chat.prebuilt.tool_images import move_tool_images_to_user_message


def chat_semantic_loss_hook(
    draft: object, losses: SemanticLossGroup, context: SemanticLossContext
) -> None:
    """Compensate for reasoning-feedback and image losses in a Chat request.

    Use this synchronous callback as ``MyLLM``'s ``input_semantic_loss_hook``
    with ``semantic_loss_protocol="hook"`` and an ``openai_chat`` upstream
    provider. The application's ``stdio_protocol`` may be a different protocol.
    The hook is optional and is never registered automatically.

    Calls ``reasoning_content_feedback`` first, then
    ``move_tool_images_to_user_message``. The first restores available visible
    reasoning to the affected assistant's ``reasoning_content`` when native
    feedback cannot be replayed. The second replaces tool-result images with
    stable source-hash references and appends a user message containing the
    referenced images, or ``[Unsupported Image]`` placeholders when they cannot be sent.
    See the individual callbacks for their matching rules and limitations.

    Args:
        draft: Complete native SDK request supplied by the generation pipeline.
            For a Chat request this is a dictionary with a ``messages`` list.
            Repairs replace that list on the same dictionary; other request
            fields and the application's source messages are left unchanged.
        losses: Collected decoder losses for this request. Only the reasoning
            and image losses recognized by the individual callbacks are repaired.
            The collection is neither modified nor returned.
        context: Conversion context supplied by the pipeline. Repairs require
            ``origin_source="input"``, ``target_protocol="openai_chat"`` and
            ``stage="request"``. The resolved ``profile`` controls model
            capabilities; ``message_cache``, when present, receives the repaired
            history explicitly.

    Returns:
        None. The callback changes ``draft`` in place. Other protocols, offline
        ``MyCodec`` conversions, output stages and unsupported draft shapes are
        left unchanged. Under the ``hook`` policy, returning accepts the whole
        draft, including any losses that these callbacks did not repair; no
        additional ``SemanticLossError`` is raised for those losses.

    Notes:
        With message caching enabled, each callback publishes its repairs and
        handles a given input batch once. Consecutive retries of the same cached
        batch reuse the repaired history and image references. New input batches
        are handled separately. Without caching, each request is repaired again;
        identical image sources retain the same hash references. Cache publication
        errors propagate to the caller; the callbacks do not roll back earlier successful repairs.

    Example:
        Given provider and model profiles configured for a Chat upstream::

            from my_llm import MyLLM
            from my_llm.codec.openai.chat.prebuilt import chat_semantic_loss_hook

            llm = MyLLM(
                provider=provider,
                model=model,
                stdio_protocol="openai_responses",
                semantic_loss_protocol="hook",
                input_semantic_loss_hook=chat_semantic_loss_hook,
                enable_message_cache=True,
            )
    """
    reasoning_content_feedback(draft, losses, context)
    move_tool_images_to_user_message(draft, losses, context)
