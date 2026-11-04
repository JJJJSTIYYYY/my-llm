"""Protocol identifiers, paths and conversion rules for ollama."""

# Protocol identity and parameter paths.
PROTOCOL = "ollama"
TEXT_PATH = "$.message.content"
REASONING_PATH = "$.message.thinking"
MAX_TOKENS_PATH = "$.options.num_predict"
REASONING_ENABLE_PATH = "$.think"
API_KEY_PATH = "$.extra_headers.Authorization"

# Request structure and field projections.
REQUEST_ENVELOPE_FIELDS = {"input", "messages", "model", "previous_response_id", "stream", "tools"}
MESSAGE_PATH = ("messages",)
TOOL_PATH = ("tools",)
TOOL_SCHEMA_PATHS = (("function", "parameters"),)
MESSAGE_IMAGE_PATHS = (("images",),)
OTHER_PROTOCOL_MESSAGE_FIELDS = {"parts", "reasoning_content", "type"}
RESPONSE_FIELDS = ("message", "done")
# Native declarations and middleware functions have independent field contracts.
NATIVE_FUNCTION_METADATA_FIELDS = ("name", "description")
MIDDLEWARE_FUNCTION_METADATA_FIELDS = ("name", "description")
MESSAGE_FIELDS = ("role", "content", "thinking", "images", "tool_name", "tool_call_id")
FUNCTION_DECLARATION_FIELDS = ("name", "description", "parameters")
FUNCTION_CALL_FIELDS = ("name", "arguments", "index")
PROMPT_EVAL_TOKEN_COUNT_FIELDS = ("prompt_eval_count", "eval_count")

# Native content recognition and conversion.
MESSAGE_ROLES = ("user", "assistant", "system", "tool")
USER_TOOL_MESSAGE_ROLES = ("user", "tool")
MIDDLEWARE_TEXT_TOOL_USE_TYPES = ("text", "tool_use")
MIDDLEWARE_TEXT_IMAGE_THINKING_TOOL_USE_TOOL_RESULT_TYPES = (
    "text",
    "image",
    "tool_use",
    "tool_result",
    "thinking",
)

# Stream routing, metadata and finish reasons.
RESPONSE_FIELD_TO_MIDDLEWARE_METADATA = (("model", "model"),)
# (Profile parameter, middleware block type, native response path).
CONTENT_REASONING_PARAMETER_BLOCK_PATHS = (
    ("content", "text", TEXT_PATH),
    ("reasoning_content", "thinking", REASONING_PATH),
)
MIDDLEWARE_FIELD_TO_MESSAGE_FIELD = {"text": "content", "reasoning": "thinking"}

# Supported image format signatures.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
WEBP_RIFF_SIGNATURE = b"RIFF"
WEBP_SIGNATURE = b"WEBP"
