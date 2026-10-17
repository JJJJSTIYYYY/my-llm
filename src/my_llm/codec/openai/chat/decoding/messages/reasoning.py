"""Native reasoning replay policy."""

from typing import Any

from my_llm.profile.configuration import merge_dicts


def thinking_enabled(profile: Any, parameters: dict) -> bool:
    if parameters.get("reasoning") is True or parameters.get("reasoning_effort", "none") != "none":
        return True
    body = merge_dicts(
        profile.get("extra_body", {}) if profile else {}, parameters.get("extra_body")
    )
    thinking = body.get("thinking", {})
    return (
        isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive")
    ) or body.get("enable_thinking") is True
