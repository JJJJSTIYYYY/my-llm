"""anthropic: native input to intermediate."""

from typing import Any

from my_llm.codec.anthropic.utils.content import CITATION_FIELDS, native_block
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.shared.conversion import (
    dictionary_views,
    encode_functions,
    mapping_list,
    select,
)
from my_llm.codec.shared.stream import BlockRef, encode_batch, record
from my_llm.exceptions import ResponseFailedError
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

    def match_item(self, item: dict) -> bool:
        if "input_schema" in item:
            return isinstance(item["input_schema"], dict) and isinstance(item.get("name"), str)
        if item.get("role") not in ("user", "assistant", "system") or "content" not in item:
            return False
        if set(item) & {
            "parts",
            "type",
            "tool_calls",
            "tool_call_id",
            "reasoning_content",
            "thinking",
            "images",
            "tool_name",
        }:
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
        result = []
        for message in messages:
            value = message.get("content")
            content = (
                [{"type": "text", "text": value}]
                if isinstance(value, str)
                else [
                    block
                    for item in dictionary_views(value or [])
                    if (block := native_block(item)) is not None
                ]
            )
            result.append({"protocol": PROTOCOL, "role": message.get("role"), "content": content})
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return encode_functions(
            tools,
            lambda tool: (
                [
                    {
                        **select(tool, "name", "description", "strict"),
                        **(
                            {"cache_control": select(tool["cache_control"], "type", "ttl")}
                            if tool.get("cache_control") is not None
                            else {}
                        ),
                        "parameters": tool.get("input_schema", {}),
                    }
                ]
                if "input_schema" in tool
                else None
            ),
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event in chunks:
            if event.get("type") == "error" or event.get("error"):
                raise ResponseFailedError(
                    "provider generation failed", details=event.get("error", event)
                )
            kind = event.get("type")
            index = event.get("index", 0)
            ops = []
            if kind == "message_start":
                message = event["message"]
                ops.append({"type": "metadata", **self._metadata(message)})
                if message.get("usage") and (
                    profile is None or profile.get("usage_mode") != "none"
                ):
                    ops.append({"type": "usage", "usage": message["usage"]})
            elif kind == "content_block_start":
                block = native_block(event["content_block"])
                if block is None:
                    continue
                ops.append({"type": "block_set", "ref": BlockRef("content", index), "block": block})
            elif kind == "content_block_delta":
                delta = event["delta"]
                if delta.get("type") == "citations_delta":
                    ops.append(
                        {
                            "type": "block_annotation",
                            "ref": BlockRef("content", index),
                            "field": "citations",
                            "value": select(delta["citation"], *CITATION_FIELDS),
                        }
                    )
                field = {
                    "text_delta": "text",
                    "thinking_delta": "thinking",
                    "signature_delta": "signature",
                    "input_json_delta": "arguments",
                }.get(delta.get("type"))
                if field:
                    ops.append(
                        {
                            "type": "block_delta",
                            "ref": BlockRef("content", index),
                            "field": field,
                            "kind": {"arguments": "tool_use", "signature": "thinking"}.get(
                                field, field
                            ),
                            "value": delta.get(
                                "partial_json" if field == "arguments" else field, ""
                            ),
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
                        {
                            "type": "metadata",
                            "native_stop_reason": reason,
                            "native_stop_sequence": delta.get("stop_sequence"),
                        }
                    )
                if event.get("usage") and (profile is None or profile.get("usage_mode") != "none"):
                    ops.append({"type": "usage", "usage": event["usage"]})
            elif kind == "message_stop":
                ops.append({"type": "finish"})
            if not ops:
                continue
            result.append(encode_batch(ops, PROTOCOL, usage=self.usage))
        return result

    def usage(self, value: dict | None) -> dict | None:
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
        return {key: event[key] for key in ("id", "model") if key in event}
