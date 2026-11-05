"""Protocol identifiers, paths and conversion rules for openai/responses."""

# Protocol identity and parameter paths.
PROTOCOL = "openai_responses"
TEXT_PATH = "$.delta"
FEEDBACK_PATH = "$.item"
MAX_TOKENS_PATH = "$.max_output_tokens"
REASONING_ENABLE_PATH = "$.reasoning"
REASONING_EFFORT_PATH = "$.reasoning.effort"
OUTPUT_PATH = "$.response.output"

# Native reasoning defaults.
DEFAULT_REASONING = {"effort": "medium", "summary": "detailed"}

# Request structure and field projections.
REQUEST_ENVELOPE_FIELDS = {"input", "messages", "model", "previous_response_id", "stream", "tools"}
MESSAGE_PATH = ("input",)
TOOL_PATH = ("tools",)
BODY_CONTAINER = "extra_body"
BODY_CONTAINERS = (BODY_CONTAINER,)
MESSAGE_DELEGATED_PATHS = (("content",), ("output",))
OTHER_PROTOCOL_MESSAGE_FIELDS = {
    "images",
    "parts",
    "reasoning_content",
    "thinking",
    "tool_call_id",
    "tool_calls",
    "tool_name",
}
REASONING_FIELDS = ("effort", "summary", "generate_summary")
MIDDLEWARE_TOOL_RESULT_IDENTITY_FIELDS = ("type", "tool_use_id", "name", "item")
ITEM_HEADER_FIELDS = ("type", "id", "role", "status", "call_id", "name")
FUNCTION_CALL_ITEM_IDENTITY_FIELDS = ("id", "status", "name", "call_id")
ITEM_ID_STATUS_FIELDS = ("id", "status")
# Native declarations and middleware functions have independent field contracts.
NATIVE_FUNCTION_METADATA_FIELDS = ("name", "description", "strict")
MIDDLEWARE_FUNCTION_METADATA_FIELDS = ("name", "description", "strict")
TOOL_FIELDS = ("type", "name", "description", "parameters", "strict")
MESSAGE_FIELDS = ("type", "role", "id", "status")
FUNCTION_CALL_FIELDS = ("type", "id", "call_id", "name", "arguments", "status")
FUNCTION_OUTPUT_FIELDS = ("type", "id", "call_id", "status")
REASONING_ITEM_FIELDS = ("type", "id", "encrypted_content", "status")
SUMMARY_TEXT_PART_FIELDS = ("type", "text")

# Native content recognition and conversion.
MESSAGE_ROLES = ("user", "assistant", "system", "developer")
FUNCTION_REASONING_REFERENCE_ITEM_TYPES = {
    "function_call",
    "function_call_output",
    "item_reference",
    "reasoning",
}
MESSAGE_CONTENT_TYPES = {"input_file", "input_image", "input_text", "output_text", "refusal"}
INPUT_OUTPUT_PLAIN_TEXT_PART_TYPES = ("input_text", "output_text", "text")
INPUT_OUTPUT_TEXT_PART_TYPES = ("input_text", "output_text")
INPUT_TEXT_IMAGE_PART_TYPES = ("input_text", "input_image")
REASONING_TEXT_TYPES = ("summary_text", "reasoning_text")
MESSAGE_REASONING_FUNCTION_CALL_ITEM_TYPES = ("message", "reasoning", "function_call")
INPUT_ITEM_TYPE_TO_MIDDLEWARE_BLOCK_TYPE = {
    "reasoning": "thinking",
    "function_call": "tool_use",
    "function_call_output": "tool_result",
}
MIDDLEWARE_TEXT_IMAGE_TOOL_USE_TYPES = ("text", "image", "tool_use")
REASONING_FUNCTION_CALL_ITEM_TYPES = ("reasoning", "function_call")
TOOL_RESULT_NATIVE_PART_TYPES = frozenset(
    ("input_text", "input_image", "input_file", "input_audio", "output_text", "text", "refusal")
)
TEXT_IMAGE_PART_FIELDS = ("type", "text", "image_url")

# Native stream event types.
RESPONSE_FAILED = "response.failed"
RESPONSE_INCOMPLETE = "response.incomplete"
RESPONSE_CREATED = "response.created"
RESPONSE_IN_PROGRESS = "response.in_progress"
OUTPUT_ITEM_ADDED = "response.output_item.added"
OUTPUT_ITEM_DONE = "response.output_item.done"
FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
CONTENT_PART_ADDED = "response.content_part.added"
CONTENT_PART_DONE = "response.content_part.done"
REASONING_SUMMARY_PART_ADDED = "response.reasoning_summary_part.added"
REASONING_SUMMARY_PART_DONE = "response.reasoning_summary_part.done"
REASONING_SUMMARY_TEXT_DELTA = "response.reasoning_summary_text.delta"
REASONING_SUMMARY_TEXT_DONE = "response.reasoning_summary_text.done"
OUTPUT_TEXT_DELTA = "response.output_text.delta"
OUTPUT_TEXT_DONE = "response.output_text.done"
RESPONSE_COMPLETED = "response.completed"

# Stream routing, metadata and finish reasons.
MIDDLEWARE_REASON_TO_INCOMPLETE_REASON = {
    "length": "max_output_tokens",
    "content_filter": "content_filter",
}
STOP_AND_TOOL_CALLS_REASONS = ("stop", "tool_calls")
FAILED_INCOMPLETE_ERROR_EVENTS = (RESPONSE_FAILED, RESPONSE_INCOMPLETE, "error")
CREATED_IN_PROGRESS_EVENTS = (RESPONSE_CREATED, RESPONSE_IN_PROGRESS)
CONTENT_PART_ADDED_DONE_EVENTS = (CONTENT_PART_ADDED, CONTENT_PART_DONE)
MESSAGE_EVENT_PREFIXES = ("response.output_text", "response.content_part", "response.refusal")
REASONING_TEXT_DELTA = "response.reasoning_text.delta"
# (Profile parameter, native event types, middleware block type).
CONTENT_REASONING_PARAMETER_EVENT_BLOCK_TYPES = (
    ("content", (OUTPUT_TEXT_DELTA,), "text"),
    ("reasoning_content", (REASONING_SUMMARY_TEXT_DELTA, REASONING_TEXT_DELTA), "thinking"),
)
RESPONSE_FIELD_TO_MIDDLEWARE_METADATA = (
    ("id", "response_id"),
    ("model", "model"),
    ("created_at", "created_at"),
)
ITEM_FIELD_TO_MIDDLEWARE_METADATA = (
    ("id", "id"),
    ("type", "type"),
    ("status", "status"),
    ("name", "name"),
    ("call_id", "call_id"),
)
