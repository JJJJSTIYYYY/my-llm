"""Profile policies applied inside native decoders' existing content loops."""

from collections.abc import Iterator

from my_llm.basic.semantic_loss import report_loss
from my_llm.profile.resolution import ResolvedProfile


def feedback_reasoning(profile: ResolvedProfile | None, *, has_tool_calls: bool) -> bool:
    """Whether an assistant message's reasoning belongs in request history."""
    if profile is None:
        return True
    mode = profile.get("feedback_reasoning_content", "none")
    return mode == "always" or mode == "when_tool_calls" and has_tool_calls


def message_feedback(
    messages: list, profile: ResolvedProfile | None
) -> Iterator[tuple[int, dict, bool]]:
    """Yield message positions and feedback decisions without projecting history.

    Only when_tool_calls needs lookahead: consecutive coalescible assistant items
    belong to one turn, even when reasoning precedes a separate tool-call item.
    Original items and native identities remain unchanged.
    """
    if profile is None or profile.get("feedback_reasoning_content", "none") != "when_tool_calls":
        feedback = feedback_reasoning(profile, has_tool_calls=False)
        for index, message in enumerate(messages):
            yield index, message, feedback
        return
    pending = []
    has_calls = False
    for index, message in enumerate(messages):
        if pending and not (
            pending[-1][1].get("coalesce")
            and message.get("coalesce")
            and pending[-1][1]["role"] == message["role"] == "assistant"
            and pending[-1][1].get("protocol") == message.get("protocol")
        ):
            for offset, item in pending:
                yield offset, item, has_calls
            pending = []
            has_calls = False
        pending.append((index, message))
        has_calls = has_calls or any(block["type"] == "tool_use" for block in message["content"])
    for offset, item in pending:
        yield offset, item, has_calls


def input_content(block: dict, profile: ResolvedProfile | None, *, feedback: bool = True) -> bool:
    """Record unsupported image/reasoning content and let the decoder skip it."""
    kind = block["type"]
    if kind == "image" and profile is not None and not profile.get("supports_vision", False):
        report_loss(block, "model does not support image input")
        return False
    if kind == "thinking" and not feedback:
        report_loss(block, "profile excludes reasoning feedback from this message")
        return False
    return True
