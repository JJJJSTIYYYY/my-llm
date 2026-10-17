"""Native profile defaults and usage policy."""


def defaults() -> dict:
    return {
        "replay_reasoning_content_when_tool_calls": False,
        "supports_reasoning_effort": False,
        "content_path": "$.choices[0].delta.content",
        "reasoning_content_path": "$.choices[0].delta.reasoning_content",
        "reasoning_enable_parameter": {"reasoning_effort": "medium"},
        "reasoning_effort_parameter": {"reasoning_effort": "{effort}"},
    }


USAGE_MODES = ("none", "stream_options")
