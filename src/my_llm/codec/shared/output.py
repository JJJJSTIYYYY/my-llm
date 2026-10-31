"""Output capability filtering shared by native and intermediate delivery."""

from collections.abc import Callable

from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.shared.capabilities import content_capability, supports_content
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Parameter


def _filters_output(profile: ResolvedProfile | None) -> bool:
    """Whether output declarations or model capabilities require content filtering."""
    return profile is not None and (
        profile.parameter("content", "output") is None
        or profile.parameter("reasoning_content", "output") is None
        or not supports_content("thinking", profile)
        or not supports_content("tool_use", profile)
    )


def _output_content(value: object, kind: str, profile: ResolvedProfile) -> bool:
    """Report unsupported output content and return whether it can be retained."""
    if kind in ("thinking", "tool_use", "tool_result") and not content_capability(
        value, kind, profile
    ):
        return False
    name = {
        "text": "content",
        "thinking": "reasoning_content",
    }.get(kind)
    if name is not None and profile.parameter(name, "output") is None:
        declaration = profile.parameters.get(name)
        report_loss(
            value,
            f"model does not support {name}"
            if isinstance(declaration, Parameter) and not declaration.supports
            else f"profile disables supported {name}",
        )
        return False
    return True


def _filter_items(values: list[dict], project: Callable[[dict], dict | None]) -> list[dict]:
    """Borrow unchanged items; allocate containers only after an actual change."""
    result = None
    for index, value in enumerate(values):
        projected = project(value)
        if projected is not value and result is None:
            result = values[:index]
        if result is not None and projected is not None:
            result.append(projected)
    return values if result is None else result


def _output_message(message: dict, profile: ResolvedProfile) -> dict:
    """Filter one message, borrowing its envelope when content is unchanged."""
    content = message["content"]
    filtered = _filter_items(
        content, lambda block: block if _output_content(block, block["type"], profile) else None
    )
    return message if filtered is content else {**message, "content": filtered}


def output_messages(values: list, profile: ResolvedProfile | None) -> list:
    """Apply output capability rules while preserving unchanged message containers."""
    if not _filters_output(profile):
        return values
    return _filter_items(values, lambda message: _output_message(message, profile))


def _output_event(event: dict, profile: ResolvedProfile) -> dict | None:
    """Filter an event and its final snapshots, dropping unsupported block events."""
    if "block" in event and not _output_content(event["block"], event["block"]["type"], profile):
        return None
    if event.get("type") == "delta":
        kind = {"text": "text", "thinking": "thinking", "signature": "thinking"}.get(
            event.get("field")
        )
        if kind is not None and not _output_content(event, kind, profile):
            return None
    result = event
    if "messages" in event:
        messages = output_messages(event["messages"], profile)
        if messages is not event["messages"]:
            result = {**event, "messages": messages}
    if "output_blocks" in event:
        blocks = _filter_items(
            event["output_blocks"],
            lambda entry: (
                entry if _output_content(entry["block"], entry["block"]["type"], profile) else None
            ),
        )
        if blocks is not event["output_blocks"]:
            if result is event:
                result = dict(event)
            result["output_blocks"] = blocks
    return result


def output_chunks(values: list, profile: ResolvedProfile | None) -> list:
    """Apply output capability rules to public events before protocol rendering."""
    if not _filters_output(profile):
        return values
    return _filter_items(values, lambda event: _output_event(event, profile))
