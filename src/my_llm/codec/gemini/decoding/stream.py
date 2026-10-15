"""Gemini streamed response decoding."""

from typing import Any

from my_llm.codec.utils.common import plain, read
from my_llm.codec.utils.native import complete_call, final_calls
from my_llm.profile.utils.resolution import ResolvedProfile
from my_llm.types import MessageChunk
from my_llm.types.exceptions import IncompleteStreamError, ResponseFailedError


class GeminiDecoder:
    """Keep signed model parts for replay and complete calls for local execution."""

    def __init__(self, profile: ResolvedProfile) -> None:
        self.profile = profile
        self.calls = []
        self.output = []
        self.reason: str | None = None
        self.usage = None
        self.response_id = None
        self.closed = False

    def feed(self, event: Any) -> list[MessageChunk]:
        if self.closed:
            raise IncompleteStreamError("Gemini decoder was already finalized")
        if read(event, "error"):
            raise ResponseFailedError("Gemini stream error", details=plain(read(event, "error")))
        feedback = read(event, "prompt_feedback")
        if read(feedback, "block_reason") not in (None, "BLOCK_REASON_UNSPECIFIED"):
            raise ResponseFailedError("Gemini blocked the prompt", details=plain(feedback))
        if read(event, "usage_metadata") is not None:
            self.usage = plain(read(event, "usage_metadata"))
        self.response_id = read(event, "response_id") or self.response_id
        candidates = read(event, "candidates", []) or []
        if not candidates:
            return []
        if len(candidates) != 1 or read(candidates[0], "index", 0) not in (None, 0):
            raise IncompleteStreamError("expected a single Gemini candidate")
        candidate = candidates[0]
        parts = read(read(candidate, "content"), "parts", []) or []
        if self.reason and parts:
            raise IncompleteStreamError("Gemini content after finish_reason")
        chunks = []
        for part in parts:
            chunk = self._decode_part(part)
            if chunk is not None:
                chunks.append(chunk)
        self._set_finish_reason(candidate)
        return chunks

    def _decode_part(self, part: Any) -> MessageChunk | None:
        native = plain(part)
        chunk = None
        if read(part, "function_call") is not None:
            native = self._decode_function_part(part, native)
        elif read(part, "text") is not None:
            if read(part, "thought", False):
                chunk = MessageChunk(reasoning_content=read(part, "text"))
            else:
                chunk = MessageChunk(content=read(part, "text"))
        elif not read(part, "thought_signature"):
            raise IncompleteStreamError("unsupported Gemini output part")
        self.output.append({"type": "gemini_part", "part": native})
        return chunk

    def _decode_function_part(self, part: Any, native: dict[str, Any]) -> dict[str, Any]:
        call = read(part, "function_call")
        if read(call, "partial_args") or read(call, "will_continue"):
            raise IncompleteStreamError("unexpected partial Gemini function arguments")
        unified = complete_call(
            read(call, "name"),
            {} if read(call, "args") is None else read(call, "args"),
            read(call, "id"),
        )
        self.calls.append(unified)
        return {**native, "function_call": {**plain(call), "id": unified["id"]}}

    def _set_finish_reason(self, candidate: Any) -> None:
        reason = read(candidate, "finish_reason")
        if reason in (None, "FINISH_REASON_UNSPECIFIED"):
            return
        if reason:
            reason = getattr(reason, "value", reason)
            if self.reason and self.reason != reason:
                raise IncompleteStreamError("conflicting Gemini finish reasons")
            if reason not in ("STOP", "MAX_TOKENS"):
                raise ResponseFailedError("Gemini generation failed", details=plain(candidate))
            self.reason = reason

    def finish(self) -> MessageChunk:
        """Require native completion and publish tools, signatures and usage."""
        if self.closed or self.reason is None:
            raise IncompleteStreamError("Gemini stream ended without finish_reason")
        self.closed = True
        if self.calls and self.reason != "STOP":
            raise IncompleteStreamError("Gemini tool calls ended incompletely")
        return MessageChunk(
            tool_calls=final_calls(self.calls),
            finish_reason=self.reason,
            response_id=self.response_id,
            response_output=self.output,
            usage=self.usage if self.profile.get("usage_mode", "none") != "none" else None,
        )
