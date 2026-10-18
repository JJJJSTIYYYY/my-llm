"""Declarative defaults for each supported standard protocol."""

from copy import deepcopy
from typing import Any

from my_llm.exceptions import UnsupportedProviderError
from my_llm.profile.configuration import Protocol

_PROTOCOL_DEFAULTS = {
    "openai_chat": {
        "replay_reasoning_content_when_tool_calls": False,
        "supports_reasoning_effort": False,
        "max_tokens_path": "$.max_completion_tokens",
        "content_path": "$.choices[0].delta.content",
        "reasoning_content_path": "$.choices[0].delta.reasoning_content",
        "reasoning_enable_parameter": {"reasoning_effort": "medium"},
        "reasoning_effort_parameter": {"reasoning_effort": "{effort}"},
    },
    "openai_responses": {
        "replay_reasoning_content_when_tool_calls": True,
        "supports_reasoning_effort": True,
        "max_tokens_path": "$.max_output_tokens",
        "content_path": "response.output_text.delta",
        "reasoning_content_path": "response.reasoning_summary_text.delta",
        "reasoning_enable_parameter": {"reasoning": {"summary": "detailed", "effort": "medium"}},
        "reasoning_effort_parameter": {"reasoning": {"summary": "detailed", "effort": "{effort}"}},
    },
    "anthropic": {
        "replay_reasoning_content_when_tool_calls": True,
        "supports_reasoning_effort": False,
        "max_tokens_path": "$.max_tokens",
        "content_path": "$.delta.text",
        "reasoning_content_path": "$.delta.thinking",
        "reasoning_enable_parameter": {},
        "reasoning_effort_parameter": {},
    },
    "gemini": {
        "replay_reasoning_content_when_tool_calls": True,
        "supports_reasoning_effort": False,
        "max_tokens_path": "$.config.max_output_tokens",
        "content_path": "$.candidates[0].content.parts[0].text",
        "reasoning_content_path": "$.candidates[0].content.parts[0].text",
        "reasoning_enable_parameter": {},
        "reasoning_effort_parameter": {},
    },
    "ollama": {
        "replay_reasoning_content_when_tool_calls": True,
        "supports_reasoning_effort": False,
        "max_tokens_path": "$.options.num_predict",
        "content_path": "$.message.content",
        "reasoning_content_path": "$.message.thinking",
        "reasoning_enable_parameter": {},
        "reasoning_effort_parameter": {},
    },
}


def protocol_defaults(protocol: Protocol) -> dict[str, Any]:
    """Return an independent standard configuration, without loading any codec."""
    try:
        return deepcopy(_PROTOCOL_DEFAULTS[protocol])
    except (KeyError, TypeError) as exc:
        raise UnsupportedProviderError(f"unsupported protocol: {protocol!r}") from exc
