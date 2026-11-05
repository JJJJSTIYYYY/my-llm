"""Protocol identifiers, paths and conversion rules for anthropic."""

# Protocol identity and parameter paths.
PROTOCOL = "anthropic"
TEXT_PATH = "$.delta.text"
REASONING_PATH = "$.delta.thinking"
FEEDBACK_PATH = "$.content_block"
MAX_TOKENS_PATH = "$.max_tokens"
REASONING_ENABLE_PATH = "$.thinking"
REASONING_EFFORT_PATH = "$.output_config.effort"

# Native reasoning defaults.
DEFAULT_REASONING = {"type": "enabled", "budget_tokens": 1024}

# Request structure and field projections.
REQUEST_ENVELOPE_FIELDS = {
    "input",
    "messages",
    "model",
    "previous_response_id",
    "stream",
    "system",
    "tools",
}
MESSAGE_PATH = ("messages",)
TOOL_PATH = ("tools",)
TOOL_SCHEMA_PATHS = (("input_schema",),)
OTHER_PROTOCOL_MESSAGE_FIELDS = {
    "images",
    "parts",
    "reasoning_content",
    "thinking",
    "tool_call_id",
    "tool_calls",
    "tool_name",
    "type",
}
SIGNATURE_PATH = ("signature",)
THINKING_PATH = ("thinking",)
IMAGE_SOURCE_FIELDS = ("type", "media_type", "data", "url")
MIDDLEWARE_TOOL_RESULT_ID_FIELDS = ("type", "tool_use_id")
TOOL_RESULT_ID_ERROR_CACHE_FIELDS = ("type", "tool_use_id", "is_error", "cache_control")
# Native declarations and middleware functions have independent field contracts.
NATIVE_FUNCTION_METADATA_FIELDS = ("name", "description", "strict")
MIDDLEWARE_FUNCTION_METADATA_FIELDS = ("name", "description", "strict")
TOOL_FIELDS = ("name", "description", "input_schema", "strict")

# Native content recognition and conversion.
MESSAGE_ROLES = ("user", "assistant", "system")
MESSAGE_BLOCK_TYPES = {
    "code_execution_tool_result",
    "document",
    "image",
    "redacted_thinking",
    "server_tool_use",
    "text",
    "thinking",
    "tool_result",
    "tool_use",
    "web_fetch_tool_result",
    "web_search_tool_result",
}
IMAGE_SOURCE_TYPES = ("url", "base64")
MIDDLEWARE_TEXT_IMAGE_TYPES = ("text", "image")
THINKING_REDACTED_THINKING_TYPES = ("thinking", "redacted_thinking")
MIDDLEWARE_TEXT_THINKING_TOOL_USE_TYPES = ("text", "thinking", "tool_use")
ENABLED_ADAPTIVE_THINKING_MODES = ("enabled", "adaptive")
MIDDLEWARE_TEXT_IMAGE_TOOL_USE_TYPES = ("text", "image", "tool_use")
CONTENT_BLOCK_TYPE_TO_FIELDS = {
    "text": ("text",),
    "thinking": ("thinking", "signature"),
    "redacted_thinking": ("data",),
    "tool_use": ("id", "name", "input"),
    "tool_result": ("tool_use_id",),
    "image": (),
}
TOOL_RESULT_NATIVE_PART_TYPES = frozenset(
    (
        *CONTENT_BLOCK_TYPE_TO_FIELDS,
        "document",
        "audio",
        "input_audio",
        "server_tool_use",
        "web_search_tool_result",
    )
)

# Native stream event types.
MESSAGE_START = "message_start"
MESSAGE_DELTA = "message_delta"
MESSAGE_STOP = "message_stop"
CONTENT_BLOCK_START = "content_block_start"
CONTENT_BLOCK_DELTA = "content_block_delta"
CONTENT_BLOCK_STOP = "content_block_stop"
TEXT_DELTA = "text_delta"
THINKING_DELTA = "thinking_delta"
SIGNATURE_DELTA = "signature_delta"
INPUT_JSON_DELTA = "input_json_delta"

# Stream routing, metadata and finish reasons.
THINKING_SIGNATURE_DELTA_TYPES = (THINKING_DELTA, SIGNATURE_DELTA)
STOP_METADATA_FIELDS = ("stop_reason", "stop_sequence")
RESPONSE_FIELD_TO_MIDDLEWARE_METADATA = (("id", "response_id"), ("model", "model"))
DELTA_TYPE_TO_MIDDLEWARE_FIELD = {
    TEXT_DELTA: "text",
    THINKING_DELTA: "reasoning",
    SIGNATURE_DELTA: "feedback_item",
    INPUT_JSON_DELTA: "arguments",
}
MIDDLEWARE_DELTA_FIELD_TO_BLOCK_TYPE = {
    "arguments": "tool_use",
    "reasoning": "thinking",
    "feedback_item": "thinking",
}
MIDDLEWARE_FIELD_TO_DELTA_VALUE_FIELD = {
    "arguments": "partial_json",
    "reasoning": "thinking",
    "feedback_item": "signature",
}
MIDDLEWARE_FIELD_TO_DELTA_FIELD = {"text": "text", "reasoning": "thinking"}
# (Profile parameter, native delta type, middleware value field, middleware block type).
CONTENT_REASONING_PARAMETER_DELTA_BLOCK_FIELDS = (
    ("content", TEXT_DELTA, "text", "text"),
    ("reasoning_content", THINKING_DELTA, "reasoning", "thinking"),
)
STOP_REASON_TO_MIDDLEWARE_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}
MIDDLEWARE_REASON_TO_STOP_REASON = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}
