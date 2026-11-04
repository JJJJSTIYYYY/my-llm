"""Protocol identifiers, paths and conversion rules for gemini."""

# Protocol identity and parameter paths.
PROTOCOL = "gemini"
TEXT_PATH = "$.candidates[0].content.parts[0].text"
FEEDBACK_PATH = "$.candidates[0].content.parts[0]"
MAX_TOKENS_PATH = "$.config.max_output_tokens"
REASONING_ENABLE_PATH = "$.config.thinking_config"
REASONING_EFFORT_PATH = "$.config.thinking_config.thinking_level"
HEADERS_PATH = "$.config.http_options.headers"
CANDIDATE_COUNT_PATH = "$.config.candidate_count"

# Native reasoning defaults.
DEFAULT_REASONING = {"include_thoughts": True, "thinking_budget": -1}

# Request structure and field projections.
BODY_CONTAINER = "config"
REQUEST_ENVELOPE_FIELDS = {
    "automatic_function_calling",
    BODY_CONTAINER,
    "contents",
    "http_options",
    "input",
    "messages",
    "model",
    "previous_response_id",
    "stream",
    "system_instruction",
    "tools",
}
MESSAGE_PATH = ("contents",)
TOOL_PATH = (BODY_CONTAINER, "tools")
BODY_CONTAINERS = (BODY_CONTAINER,)
INLINE_FILE_DATA_FIELDS = ("inline_data", "file_data")
RESPONSE_FIELDS = (
    "candidates",
    "usage_metadata",
    "prompt_feedback",
    "response_id",
    "model_version",
)
HTTP_OPTION_FIELDS = ("base_url", "headers")
MIDDLEWARE_TOOL_RESULT_ID_CONTENT_SIGNATURE_FIELDS = ("type", "tool_use_id", "content", "signature")
TEXT_PART_FIELDS = ("text", "thought", "thought_signature")
IMAGE_MEDIA_FIELDS = ("mime_type", "data", "file_uri")
NATIVE_FUNCTION_METADATA_FIELDS = ("name", "description")
FUNCTION_CALL_FIELDS = ("id", "name", "args")
FUNCTION_RESPONSE_FIELDS = ("id", "name", "response")
FUNCTION_DECLARATION_FIELDS = ("name", "description", "parameters", "parameters_json_schema")

# Native content recognition and conversion.
MESSAGE_ROLES = ("user", "model", "system", "developer")
ABSENT_OR_UNSPECIFIED_BLOCK_REASONS = (None, "BLOCK_REASON_UNSPECIFIED")
MIDDLEWARE_TEXT_THINKING_TOOL_USE_TYPES = ("text", "thinking", "tool_use")

# Stream routing, metadata and finish reasons.
ABSENT_OR_UNSPECIFIED_FINISH_REASONS = (None, "FINISH_REASON_UNSPECIFIED")
MAX_TOKENS_FINISH_REASON = "MAX_TOKENS"
STOP_AND_MAX_TOKENS_FINISH_REASONS = ("STOP", MAX_TOKENS_FINISH_REASON)
RESPONSE_FIELD_TO_MIDDLEWARE_METADATA = (("response_id", "response_id"), ("model_version", "model"))
# (Profile parameter, middleware block type, middleware value field).
CONTENT_REASONING_PARAMETER_BLOCK_FIELDS = (
    ("content", "text", "text"),
    ("reasoning_content", "thinking", "reasoning"),
)
MIDDLEWARE_REASON_TO_FINISH_REASON = {
    "length": MAX_TOKENS_FINISH_REASON,
    "content_filter": "SAFETY",
    "stop": "STOP",
    "tool_calls": "STOP",
}

# Gemini Schema to JSON Schema translation.
SCHEMA_OBJECT_KEYWORDS = ("properties", "$defs")
UNSPECIFIED_SCHEMA_TYPE = "TYPE_UNSPECIFIED"
SCHEMA_DEFS_PREFIX = "#/defs/"
JSON_SCHEMA_DEFS_PREFIX = "#/$defs/"
SCHEMA_TYPE_TO_JSON_SCHEMA_TYPE = {
    name.upper(): name
    for name in ("object", "array", "string", "number", "integer", "boolean", "null")
}
SCHEMA_KEYWORD_TO_JSON_SCHEMA_KEYWORD = {
    "any_of": "anyOf",
    "anyOf": "anyOf",
    "defs": "$defs",
    "ref": "$ref",
    "additional_properties": "additionalProperties",
    "additionalProperties": "additionalProperties",
    "min_items": "minItems",
    "max_items": "maxItems",
    "min_length": "minLength",
    "max_length": "maxLength",
    "min_properties": "minProperties",
    "max_properties": "maxProperties",
}
SCHEMA_COUNT_KEYWORDS = {
    "maxItems",
    "maxLength",
    "maxProperties",
    "minItems",
    "minLength",
    "minProperties",
}
SCHEMA_LITERAL_VALUE_KEYWORDS = {
    "default",
    "description",
    "enum",
    "format",
    "maximum",
    "minimum",
    "pattern",
    "required",
    "title",
}
SCHEMA_KEYWORDS = tuple(
    dict.fromkeys(
        (
            "type",
            "properties",
            "items",
            "example",
            *sorted(SCHEMA_LITERAL_VALUE_KEYWORDS),
            *sorted(SCHEMA_COUNT_KEYWORDS),
            "$defs",
            "$ref",
            "anyOf",
            "additionalProperties",
            *SCHEMA_KEYWORD_TO_JSON_SCHEMA_KEYWORD,
        )
    )
)
