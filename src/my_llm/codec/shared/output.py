"""Output capability filtering shared by native and intermediate delivery."""

from my_llm.profile.resolution import ResolvedProfile
from my_llm.semantic_loss import originated, report_loss


def _filters_output(profile: ResolvedProfile | None) -> bool:
    return profile is not None and (
        profile.parameter("content", "output") is None
        or profile.parameter("reasoning_content", "output") is None
    )


def _output_block(block: dict, profile: ResolvedProfile) -> bool:
    kind = block.get("type")
    name = {
        "text": "content",
        "thinking": "reasoning_content",
        "redacted_thinking": "reasoning_content",
    }.get(kind)
    if name is not None and profile.parameter(name, "output") is None:
        report_loss(block, f"profile disables supported {name}")
        return False
    return True


def output_messages(values: list, profile: ResolvedProfile | None) -> list:
    if not _filters_output(profile):
        return values
    return [
        {**message, "content": [b for b in message.get("content", []) if _output_block(b, profile)]}
        for message in values
    ]


def output_chunks(values: list, profile: ResolvedProfile | None) -> list:
    if not _filters_output(profile):
        return values
    result = []
    for event in values:
        if "block" in event and not _output_block(event["block"], profile):
            continue
        if event.get("type") == "delta":
            kind = {"text": "text", "thinking": "thinking", "signature": "thinking"}.get(
                event.get("field")
            )
            if kind is not None:
                block = originated(
                    {"type": kind},
                    getattr(event, "origin_path", "$"),
                    getattr(event, "origin_value", event),
                )
                if not _output_block(block, profile):
                    continue
        if "messages" in event:
            event = {**event, "messages": output_messages(event["messages"], profile)}
        if "output_blocks" in event:
            event = {
                **event,
                "output_blocks": [
                    entry
                    for entry in event["output_blocks"]
                    if _output_block(entry["block"], profile)
                ],
            }
        result.append(event)
    return result
