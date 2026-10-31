"""anthropic: native input to intermediate."""

from typing import Any

from anthropic.types import MessageParam, ToolParam

from my_llm.basic.data import _MISSING, capture, dictionary_views
from my_llm.basic.exceptions import MessageValidationError, ResponseFailedError
from my_llm.basic.semantic_loss import originated
from my_llm.codec.anthropic.content import intermediate_block, native_block
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.shared.conversion import mapping_list, select
from my_llm.codec.shared.stream import BlockRef, encode_batch, metadata, record
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "anthropic"
REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}


class AnthropicEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL
    message_type = MessageParam
    tool_type = ToolParam

    def _validation_messages(self, messages: list) -> list:
        """Project supported Anthropic message fields before native schema validation."""
        result = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                projected = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    value = native_block(block)
                    if value is None or (
                        value.get("type") == "image" and message.get("role") != "user"
                    ):
                        continue
                    if value.get("type") == "thinking" and not value.get("signature"):
                        if not isinstance(value.get("thinking"), str):
                            raise MessageValidationError("thinking must be a string")
                        # A signature is a target replay constraint, not a reason
                        # to reject supported source reasoning during SDK checks.
                        continue
                    projected.append(value)
                content = projected
            result.append({"role": message.get("role"), "content": content})
        return result

    def _validation_tools(self, tools: list) -> list:
        """Project supported Anthropic function declarations before native schema validation."""
        return [
            select(tool, "name", "description", "input_schema", "strict")
            for tool in tools
            if "input_schema" in tool
        ]

    def match_item(self, item: dict) -> bool:
        """Whether a native message or tool has a recognizable Anthropic shape."""
        if "input_schema" in item:
            return isinstance(item["input_schema"], dict) and isinstance(item.get("name"), str)
        if item.get("role") not in ("user", "assistant", "system") or "content" not in item:
            return False
        if any(
            key in item
            for key in {
                "parts",
                "type",
                "tool_calls",
                "tool_call_id",
                "reasoning_content",
                "thinking",
                "images",
                "tool_name",
            }
        ):
            return False
        content = item["content"]
        return (
            isinstance(content, str)
            or content is None
            or (
                mapping_list(content)
                and all(
                    block.get("type")
                    in {
                        "text",
                        "thinking",
                        "redacted_thinking",
                        "tool_use",
                        "tool_result",
                        "image",
                        "document",
                        "server_tool_use",
                        "web_search_tool_result",
                        "web_fetch_tool_result",
                        "code_execution_tool_result",
                    }
                    for block in content
                )
            )
        )

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Normalize supported Anthropic blocks and reasoning provenance into middleware messages."""
        result = []
        for index, message in enumerate(messages):
            path = f"$.messages[{index}].content"
            value = message.get("content")
            content = (
                [originated({"type": "text", "text": value}, path, value)]
                if isinstance(value, str)
                else [
                    block
                    for part_index, item in enumerate(dictionary_views(value or []))
                    if (block := intermediate_block(item, path=f"{path}[{part_index}]")) is not None
                ]
            )
            result.append({"protocol": PROTOCOL, "role": message.get("role"), "content": content})
        for message in result:
            if message.get("role") != "user":
                message["content"] = [
                    block for block in message["content"] if block.get("type") != "image"
                ]
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Project supported Anthropic function declarations into canonical tool groups."""
        return [
            {
                "functions": [
                    originated(
                        {
                            **select(tool, "name", "description", "strict"),
                            "parameters": originated(
                                tool.get("input_schema", {}),
                                f"$.tools[{index}].input_schema",
                                tool.get("input_schema", {}),
                            ),
                        },
                        f"$.tools[{index}]",
                        tool,
                    )
                ]
            }
            for index, tool in enumerate(tools)
            if "input_schema" in tool
        ]

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Translate native Anthropic events into middleware operation batches.

        Extract supported content, metadata and usage; represent block identity
        and protocol completion explicitly for StreamState. Provider failures
        propagate as stream errors. Tool-argument assembly and public tool-event
        delivery belong to StreamState rather than this protocol parser.
        """
        result = []
        for event_index, event in enumerate(chunks):
            if event.get("type") == "error" or event.get("error"):
                raise ResponseFailedError(
                    "provider generation failed", details=event.get("error", event)
                )
            kind = event.get("type")
            index = event.get("index", 0)
            ops = []
            if kind == "message_start":
                message = event["message"]
                ops.append(metadata(self._metadata(message)))
                if message.get("usage") is not None:
                    ops.append({"type": "usage", "usage": message["usage"]})
            elif kind == "content_block_start":
                block = intermediate_block(event["content_block"])
                if block is None or block["type"] not in (
                    "text",
                    "thinking",
                    "tool_use",
                ):
                    continue
                if block["type"] == "tool_use":
                    block["input"] = capture(block["input"])
                ops.append(
                    {
                        "type": "block_set",
                        "ref": BlockRef("content", index),
                        "block": block,
                    }
                )
            elif kind == "content_block_delta":
                delta = event["delta"]
                field = {
                    "text_delta": "text",
                    "thinking_delta": "thinking",
                    "signature_delta": "signature",
                    "input_json_delta": "arguments",
                }.get(delta.get("type"))
                if field:
                    value = (
                        self.parameter_value(
                            event,
                            profile,
                            "content" if field == "text" else "reasoning_content",
                            "$.delta.text" if field == "text" else "$.delta.thinking",
                            _MISSING,
                        )
                        if field in ("text", "thinking")
                        else delta.get("partial_json" if field == "arguments" else field, "")
                    )
                    if value is not None and value is not _MISSING:
                        ops.append(
                            {
                                "type": "block_delta",
                                "ref": BlockRef("content", index),
                                "field": field,
                                "kind": {"arguments": "tool_use", "signature": "thinking"}.get(
                                    field, field
                                ),
                                "value": value,
                            }
                        )
            elif kind == "content_block_stop":
                ops.append({"type": "block_stop", "ref": BlockRef("content", index)})
            elif kind == "message_delta":
                delta = event.get("delta", {})
                if delta.get("stop_reason"):
                    reason = delta["stop_reason"]
                    ops.append({"type": "reason", "reason": REASONS.get(reason, reason)})
                    ops.append(
                        metadata(
                            {
                                f"native_{key}": delta[key]
                                for key in ("stop_reason", "stop_sequence")
                                if isinstance(delta.get(key), (str, int, float))
                            }
                        )
                    )
                if event.get("usage") is not None:
                    ops.append({"type": "usage", "usage": event["usage"]})
            elif kind == "message_stop":
                ops.append({"type": "finish"})
            if not ops:
                continue
            for operation_index, operation in enumerate(ops):
                event_path = f"$.chunks[{parameters.get('chunk_offset', 0) + event_index}]"
                if "block" in operation and not hasattr(operation["block"], "origin_path"):
                    operation["block"] = originated(operation["block"], event_path, event)
                origin_path, origin_value = event_path, event
                if operation.get("type") == "block_set":
                    operation["block"] = originated(
                        operation["block"],
                        event_path + ".content_block",
                        event.get("content_block", event),
                    )
                elif operation.get("type") == "block_delta":
                    field = operation.get("field")
                    native_field = "partial_json" if field == "arguments" else field
                    origin_path, origin_value = (
                        event_path + f".delta.{native_field}",
                        operation.get("value"),
                    )
                ops[operation_index] = originated(operation, origin_path, origin_value)
            result.append(encode_batch(ops, PROTOCOL, usage=self.usage))
        return result

    def usage(self, value: dict | None) -> dict | None:
        """Normalize reported Anthropic token counters, retaining absent usage as None."""
        if value is None:
            return None
        cached = value.get("cache_read_input_tokens")
        written = value.get("cache_creation_input_tokens")
        return record(
            input_base=value.get("input_tokens"),
            input_add={"cache_read": cached, "cache_write": written},
            output_base=value.get("output_tokens"),
            cached=cached,
            cache_write=written,
        )

    def _metadata(self, event: dict) -> dict:
        """Project supported Anthropic response identity and model metadata."""
        return {
            "response_id" if key == "id" else key: event[key]
            for key in ("id", "model")
            if isinstance(event.get(key), (str, int, float))
        }
