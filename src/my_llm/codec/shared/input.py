"""Profile policies applied inside native decoders' existing content loops."""

from collections.abc import Iterable, Iterator

from my_llm.codec.shared.capabilities import content_capability, supports_content
from my_llm.codec.shared.conversion import select
from my_llm.profile.resolution import ResolvedProfile


def _keeps_reasoning(mode: str, has_tool_calls: bool) -> bool:
    """Apply the shared reasoning-feedback mode to one message or native group."""
    return mode == "always" or mode == "when_tool_calls" and has_tool_calls


class InputPolicy:
    """Resolve model capabilities once for one native-message filtering pass.

    This request-local policy caches configuration decisions, never message
    validity. Every original, translated or transformed value still passes through
    allows; unsupported values retain their source paths and semantic-loss policy.
    """

    __slots__ = ("profile", "supported", "feedback_mode", "requires_tool_calls")

    def __init__(self, profile: ResolvedProfile | None) -> None:
        """Snapshot the fixed model capabilities and reasoning-feedback mode."""
        self.profile = profile
        self.supported = {
            kind: supports_content(kind, profile)
            for kind in ("image", "thinking", "tool_use", "message_name")
        }
        self.supported["tool_result"] = self.supported["tool_use"]
        self.feedback_mode = (
            "always" if profile is None else profile.get("feedback_reasoning_content", "none")
        )
        self.requires_tool_calls = (
            self.supported["thinking"] and self.feedback_mode == "when_tool_calls"
        )

    def reasoning_feedback(self, *, has_tool_calls: bool = False) -> bool:
        """Whether reasoning is retained for this message or native item group."""
        return self.supported["thinking"] and _keeps_reasoning(self.feedback_mode, has_tool_calls)

    def allows(
        self, value: object, kind: str, *, feedback: bool = True, path: str | None = None
    ) -> bool:
        """Check one value, reporting unsupported capabilities only on failure."""
        if not self.supported.get(kind, True):
            return content_capability(value, kind, self.profile, path=path)
        return kind != "thinking" or feedback


def pad_reasoning(message: dict, field: str, *, feedback: bool, has_tool_calls: bool) -> None:
    """Fill an absent scalar reasoning field only when tool feedback is requested."""
    if (
        feedback
        and has_tool_calls
        and message.get("role") == "assistant"
        and message.get(field) is None
    ):
        message[field] = ""


def feedback_reasoning(profile: ResolvedProfile | None, *, has_tool_calls: bool) -> bool:
    """Whether an assistant message's reasoning belongs in request history."""
    if not supports_content("thinking", profile):
        return False
    if profile is None:
        return True
    mode = profile.get("feedback_reasoning_content", "none")
    return _keeps_reasoning(mode, has_tool_calls)


def _message_groups(messages: Iterable[dict]) -> Iterator[list[tuple[int, dict]]]:
    """Buffer only consecutive fragments belonging to one assistant turn."""
    pending = []
    for index, message in enumerate(messages):
        if pending and not (
            pending[-1][1].get("coalesce")
            and message.get("coalesce")
            and pending[-1][1]["role"] == message["role"] == "assistant"
            and pending[-1][1].get("protocol") == message.get("protocol")
        ):
            yield pending
            pending = []
        pending.append((index, message))
    if pending:
        yield pending


def message_feedback(
    messages: Iterable[dict], profile: ResolvedProfile | None, *, coalesce: bool = True
) -> Iterator[tuple[int, dict, bool]]:
    """Group history once for feedback and optional native message coalescing.

    Set coalesce=False when the target preserves separate native items and IDs.
    Original messages and their content remain borrowed until native rendering.
    """
    tool_feedback = (
        profile is not None
        and supports_content("thinking", profile)
        and profile.get("feedback_reasoning_content") == "when_tool_calls"
    )
    if not coalesce and not tool_feedback:
        feedback = feedback_reasoning(profile, has_tool_calls=False)
        for index, message in enumerate(messages):
            yield index, message, feedback
        return
    for group in _message_groups(messages):
        has_calls = tool_feedback and any(
            block["type"] == "tool_use" for _, message in group for block in message["content"]
        )
        feedback = feedback_reasoning(profile, has_tool_calls=has_calls)
        if coalesce and len(group) > 1:
            index, first = group[0]
            yield (
                index,
                {
                    **select(first, "protocol", "coalesce", "role", "name", "item"),
                    "content": [block for _, message in group for block in message["content"]],
                },
                feedback,
            )
        else:
            for index, message in group:
                yield index, message, feedback


def input_capability(
    value: object,
    kind: str,
    profile: ResolvedProfile | None,
    *,
    feedback: bool = True,
    path: str | None = None,
) -> bool:
    """Apply input capabilities to canonical or native values in their existing loops."""
    return content_capability(value, kind, profile, path=path) and (kind != "thinking" or feedback)


def input_content(block: dict, profile: ResolvedProfile | None, *, feedback: bool = True) -> bool:
    """Record unsupported content and let the native decoder skip it."""
    return input_capability(block, block["type"], profile, feedback=feedback)
