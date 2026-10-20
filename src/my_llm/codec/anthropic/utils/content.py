"""Select recognized Anthropic content; application objects remain open."""

from my_llm.codec.shared.conversion import select

FIELDS = {
    "text": ("text",),
    "thinking": ("thinking", "signature"),
    "redacted_thinking": ("data",),
    "tool_use": ("id", "name", "input"),
    "server_tool_use": ("id", "name", "input"),
    "tool_result": ("tool_use_id", "is_error"),
    "image": (),
    "document": ("title", "context"),
    "web_search_tool_result": ("tool_use_id",),
    "web_fetch_tool_result": ("tool_use_id",),
    "code_execution_tool_result": ("tool_use_id",),
    "bash_code_execution_tool_result": ("tool_use_id",),
    "text_editor_code_execution_tool_result": ("tool_use_id",),
}
CITATION_FIELDS = (
    "type",
    "cited_text",
    "document_index",
    "document_title",
    "start_char_index",
    "end_char_index",
    "start_page_number",
    "end_page_number",
    "start_block_index",
    "end_block_index",
    "url",
    "title",
    "encrypted_index",
    "file_id",
    "search_result_index",
    "source",
)
RESULT_FIELDS = {
    "web_search_result": ("url", "title", "encrypted_content", "page_age"),
    "web_search_tool_result_error": ("error_code",),
    "web_fetch_tool_result_error": ("error_code",),
    "web_fetch_result": ("url", "retrieved_at"),
    "code_execution_result": ("stdout", "stderr", "return_code"),
    "code_execution_tool_result_error": ("error_code",),
    "code_execution_output": ("file_id",),
    "bash_code_execution_result": ("stdout", "stderr", "return_code"),
    "bash_code_execution_tool_result_error": ("error_code",),
    "text_editor_code_execution_view_result": (
        "content",
        "file_type",
        "num_lines",
        "start_line",
        "total_lines",
    ),
    "text_editor_code_execution_create_result": ("is_file_update",),
    "text_editor_code_execution_str_replace_result": (
        "lines",
        "new_lines",
        "new_start",
        "old_lines",
        "old_start",
    ),
    "text_editor_code_execution_tool_result_error": ("error_code", "error_message"),
}


def native_block(block: dict) -> dict | None:
    kind = block.get("type")
    fields = FIELDS.get(kind, RESULT_FIELDS.get(kind))
    if fields is None:
        return None
    value = select(block, "type", *fields)
    if "cache_control" in block:
        value["cache_control"] = select(block["cache_control"], "type", "ttl")
    if "citations" in block:
        citations = block["citations"]
        value["citations"] = (
            [select(item, *CITATION_FIELDS) for item in citations]
            if isinstance(citations, list)
            else select(citations, "enabled")
        )
    if kind in ("image", "document") and "source" in block:
        source = block["source"]
        value["source"] = select(source, "type", "media_type", "data", "url", "file_id")
        if source.get("type") == "content":
            value["source"]["content"] = _content(source.get("content"))
    if "content" in block and kind not in ("text_editor_code_execution_view_result",):
        value["content"] = (
            block["content"]
            if kind == "tool_result" and isinstance(block["content"], dict)
            else _content(block["content"])
        )
    return value


def _content(content):
    if isinstance(content, list):
        return [value for item in content if (value := native_block(item)) is not None]
    if isinstance(content, dict):
        return native_block(content)
    return content
