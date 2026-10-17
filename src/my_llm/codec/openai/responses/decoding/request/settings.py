"""Native profile defaults and usage policy."""


def defaults() -> dict:
    return {
        "replay_reasoning_content_when_tool_calls": True,
        "supports_reasoning_effort": True,
        "content_path": "response.output_text.delta",
        "reasoning_content_path": "response.reasoning_summary_text.delta",
        "reasoning_enable_parameter": {"reasoning": {"summary": "detailed", "effort": "medium"}},
        "reasoning_effort_parameter": {"reasoning": {"summary": "detailed", "effort": "{effort}"}},
    }


USAGE_MODES = ("none", "native")
