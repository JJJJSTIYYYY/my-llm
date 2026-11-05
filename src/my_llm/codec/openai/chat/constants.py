"""Protocol identifiers, paths and conversion rules for openai/chat."""

# Protocol identity and parameter paths.
PROTOCOL = "openai_chat"
TEXT_PATH = "$.choices[0].delta.content"
REASONING_PATH = "$.choices[0].delta.reasoning_content"
MAX_TOKENS_PATH = "$.max_completion_tokens"
REASONING_EFFORT_PATH = "$.reasoning_effort"
CANDIDATE_COUNT_PATH = "$.n"
BODY_CANDIDATE_COUNT_PATH = "$.extra_body.n"

# Native reasoning defaults.
DEFAULT_REASONING_EFFORT = "medium"

# Request structure and field projections.
REQUEST_ENVELOPE_FIELDS = {"input", "messages", "model", "previous_response_id", "stream", "tools"}
MESSAGE_PATH = ("messages",)
TOOL_PATH = ("tools",)
BODY_CONTAINER = "extra_body"
BODY_CONTAINERS = (BODY_CONTAINER,)
MESSAGE_DELEGATED_PATHS = (("content",), ("tool_calls",), ("function_call",))
USAGE_MODES = ("none", "stream_options")
OTHER_PROTOCOL_MESSAGE_FIELDS = {"images", "parts", "thinking", "tool_name", "type"}
TOKEN_LIMIT_FIELDS = ("max_tokens", "max_completion_tokens")
FUNCTION_DESCRIPTION_PARAMETERS_FIELDS = ("description", "parameters")
RESPONSE_FIELDS = ("choices", "usage")
MESSAGE_IDENTITY_FIELDS = ("role", "name")
FUNCTION_DECLARATION_FIELDS = ("name", "description", "parameters", "strict")
TEXT_PART_FIELDS = ("type", "text")
MESSAGE_ROLE_NAME_TOOL_CALL_FIELDS = ("role", "name", "tool_call_id", "function_call")
TOOL_CALL_ID_TYPE_FIELDS = ("id", "type")
FUNCTION_CALL_FIELDS = ("name", "arguments")
NATIVE_FUNCTION_METADATA_FIELDS = ("name", "description", "strict")

# Native content recognition and conversion.
MESSAGE_ROLES = ("user", "assistant", "system", "developer", "tool")
TOOL_RESULT_ROLES = ("tool", "function")
MESSAGE_CONTENT_TYPES = {"file", "image_url", "input_audio", "refusal", "text"}
MIDDLEWARE_TEXT_IMAGE_THINKING_TOOL_USE_TOOL_RESULT_TYPES = (
    "text",
    "image",
    "tool_use",
    "tool_result",
    "thinking",
)
TOOL_RESULT_NATIVE_PART_TYPES = frozenset(("text", "image_url", "input_audio", "file", "refusal"))

# Stream routing, metadata and finish reasons.
# (Profile parameter, middleware block type, native response path).
CONTENT_REASONING_PARAMETER_BLOCK_PATHS = (
    ("content", "text", TEXT_PATH),
    ("reasoning_content", "thinking", REASONING_PATH),
)
MIDDLEWARE_FIELD_TO_DELTA_FIELD = {"text": "content", "reasoning": "reasoning_content"}
FINISH_REASONS = ("stop", "length", "tool_calls", "content_filter", "function_call")
RESPONSE_FIELD_TO_MIDDLEWARE_METADATA = (
    ("id", "response_id"),
    ("model", "model"),
    ("created", "created_at"),
)
CHUNK_OBJECT = "chat.completion.chunk"
