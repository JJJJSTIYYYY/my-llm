"""Native profile defaults and usage policy."""


def defaults() -> dict:
    return {
        "replay_reasoning_content_when_tool_calls": True,
        "supports_reasoning_effort": False,
        "content_path": "$.message.content",
        "reasoning_content_path": "$.message.thinking",
        "reasoning_enable_parameter": {},
        "reasoning_effort_parameter": {},
    }


USAGE_MODES = ("none", "native")
