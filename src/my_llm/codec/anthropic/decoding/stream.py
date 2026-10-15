"""Anthropic streamed response decoding."""

import json
from typing import Any

from my_llm.codec.utils.common import plain, read
from my_llm.codec.utils.native import final_calls
from my_llm.codec.utils.validation import validate_tool_call
from my_llm.profile.utils.resolution import ResolvedProfile
from my_llm.types import MessageChunk
from my_llm.types.exceptions import IncompleteStreamError, ResponseFailedError

_DELTA_FIELDS = {
    "text_delta": "text",
    "thinking_delta": "thinking",
    "signature_delta": "signature",
    "input_json_delta": "partial_json",
}
_FIELD_BLOCKS = {
    "text": "text",
    "thinking": "thinking",
    "signature": "thinking",
    "partial_json": "tool_use",
}
_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class AnthropicDecoder:
    """Accumulate content blocks, signed thinking and partial tool JSON."""

    def __init__(self, profile: ResolvedProfile) -> None:
        self.profile = profile
        self.blocks: dict[int, dict[str, Any]] = {}
        self.fragments: dict[int, list[str]] = {}
        self.text_fragments: dict[tuple[int, str], list[str]] = {}
        self.active: set[int] = set()
        self.reason: str | None = None
        self.stopped = False
        self.closed = False
        self.response_id: str | None = None
        self.usage: dict[str, Any] = {}

    def feed(self, event: Any) -> list[MessageChunk]:
        kind = read(event, "type")
        if kind == "error":
            raise ResponseFailedError("Anthropic stream error", details=plain(read(event, "error")))
        if kind == "ping":
            return []
        if self.stopped or self.closed:
            raise IncompleteStreamError("Anthropic event after message_stop")
        if kind == "message_start":
            self._start_message(event)
        elif kind == "content_block_start":
            return self._start_block(event)
        elif kind == "content_block_delta":
            return self._update_block(event)
        elif kind == "content_block_stop":
            self._stop_block(event)
        elif kind == "message_delta":
            self._update_message(event)
        elif kind == "message_stop":
            self.stopped = True
        return []

    def _start_message(self, event: Any) -> None:
        message = read(event, "message")
        self.response_id = read(message, "id")
        self.usage.update(plain(read(message, "usage")) or {})

    def _start_block(self, event: Any) -> list[MessageChunk]:
        index = read(event, "index")
        if not isinstance(index, int) or index < 0 or index in self.blocks:
            raise IncompleteStreamError("invalid Anthropic content block index")
        block = plain(read(event, "content_block"))
        if block.get("type") not in ("text", "thinking", "redacted_thinking", "tool_use"):
            raise IncompleteStreamError("only text, thinking and local tool blocks are supported")
        self.blocks[index] = block
        self.active.add(index)
        if block["type"] == "text" and block.get("text"):
            return [MessageChunk(content=block["text"])]
        if block["type"] == "thinking" and block.get("thinking"):
            return [MessageChunk(reasoning_content=block["thinking"])]
        return []

    def _update_block(self, event: Any) -> list[MessageChunk]:
        index = read(event, "index")
        if index not in self.active:
            raise IncompleteStreamError("delta without an active content block")
        delta = read(event, "delta")
        field = _DELTA_FIELDS.get(read(delta, "type"))
        if field is None or self.blocks[index]["type"] != _FIELD_BLOCKS[field]:
            raise IncompleteStreamError("invalid Anthropic block delta")
        fragment = read(delta, field, "")
        if field == "partial_json":
            self.fragments.setdefault(index, []).append(fragment)
        else:
            self.text_fragments.setdefault((index, field), []).append(fragment)
            if field == "text":
                return [MessageChunk(content=fragment)]
            if field == "thinking":
                return [MessageChunk(reasoning_content=fragment)]
        return []

    def _stop_block(self, event: Any) -> None:
        index = read(event, "index")
        if index not in self.active:
            raise IncompleteStreamError("stop without an active content block")
        self.active.remove(index)
        block = self.blocks[index]
        for field in ("text", "thinking", "signature"):
            if (index, field) in self.text_fragments:
                block[field] = block.get(field, "") + "".join(
                    self.text_fragments.pop((index, field))
                )
        if block["type"] == "tool_use":
            self._complete_tool_input(index, block)

    def _complete_tool_input(self, index: int, block: dict[str, Any]) -> None:
        arguments = "".join(self.fragments.get(index, [])) or json.dumps(block.get("input"))
        validate_tool_call(
            {
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": arguments,
                },
            },
            streamed=True,
        )
        block["input"] = json.loads(arguments)

    def _update_message(self, event: Any) -> None:
        self.reason = read(read(event, "delta"), "stop_reason") or self.reason
        self.usage.update(plain(read(event, "usage")) or {})

    def finish(self) -> MessageChunk:
        """Require the terminal event and closed blocks before publishing calls."""
        if self.closed or not self.stopped or not self.reason or self.active:
            raise IncompleteStreamError("Anthropic stream ended without complete message_stop")
        self.closed = True
        output = [self.blocks[i] for i in sorted(self.blocks)]
        calls = [
            {
                "id": b["id"],
                "type": "function",
                "function": {
                    "name": b["name"],
                    "arguments": json.dumps(b["input"], ensure_ascii=False),
                },
            }
            for b in output
            if b["type"] == "tool_use"
        ]
        if bool(calls) != (self.reason == "tool_use"):
            raise IncompleteStreamError("Anthropic tool calls and stop_reason disagree")
        return MessageChunk(
            tool_calls=final_calls(calls),
            finish_reason=_FINISH_REASONS.get(self.reason, self.reason),
            response_id=self.response_id,
            response_output=output,
            usage=self.usage if self.profile.get("usage_mode", "none") != "none" else None,
        )
