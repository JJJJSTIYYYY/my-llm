"""Request guards and explicit cache publication shared by Chat loss handlers."""

from weakref import WeakSet

import my_llm.codec.openai.chat.constants as constants
from my_llm.basic.semantic_loss import SemanticLossContext


def request_messages(
    draft: object, context: SemanticLossContext, completed: WeakSet
) -> list | None:
    """Select an unhandled Chat request without changing another conversion stage."""
    if (
        context.origin_source != "input"
        or context.target_protocol != constants.PROTOCOL
        or context.stage != "request"
        or not isinstance(draft, dict)
        or not isinstance(draft.get("messages"), list)
        or context.message_cache is not None
        and context.message_cache.identity in completed
    ):
        return None
    return draft["messages"]


def publish_messages(
    draft: dict, messages: list, context: SemanticLossContext, completed: WeakSet
) -> None:
    """Publish owned cache repairs before marking this input batch as handled."""
    cache = context.message_cache
    if cache is not None:
        cache.replace(messages)
        completed.add(cache.identity)
    draft["messages"] = messages
