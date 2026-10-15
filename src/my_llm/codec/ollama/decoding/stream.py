"""Ollama streamed response decoding."""

from typing import Any

from my_llm.codec.utils.common import read
from my_llm.codec.utils.native import complete_call, final_calls
from my_llm.profile.utils.resolution import ResolvedProfile
from my_llm.types import MessageChunk
from my_llm.types.exceptions import IncompleteStreamError, ResponseFailedError


class OllamaDecoder:
    """Translate thinking/content and buffer complete native tool objects."""

    def __init__(self, profile: ResolvedProfile) -> None:
        self.profile = profile
        self.calls = []
        self.done = False
        self.reason = None
        self.usage = None
        self.closed = False

    def feed(self, event: Any) -> list[MessageChunk]:
        if self.done or self.closed:
            raise IncompleteStreamError("Ollama event after done")
        if read(event, "error"):
            raise ResponseFailedError("Ollama stream error", details=read(event, "error"))
        message = read(event, "message")
        for call in read(message, "tool_calls", []) or []:
            function = read(call, "function")
            self.calls.append(
                complete_call(read(function, "name"), read(function, "arguments"), read(call, "id"))
            )
        if read(event, "done", False):
            self.done = True
            self.reason = read(event, "done_reason") or "stop"
            self.usage = {
                key: read(event, key)
                for key in (
                    "prompt_eval_count",
                    "eval_count",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_duration",
                    "eval_duration",
                )
                if read(event, key) is not None
            }
        content = read(message, "content") or ""
        reasoning = read(message, "thinking") or ""
        return (
            [MessageChunk(content=content, reasoning_content=reasoning)]
            if content or reasoning
            else []
        )

    def finish(self) -> MessageChunk:
        """Reject truncated streams and publish complete tools and native usage."""
        if self.closed or not self.done:
            raise IncompleteStreamError("Ollama stream ended without done")
        self.closed = True
        if self.calls and self.reason != "stop":
            raise IncompleteStreamError("Ollama tool calls ended incompletely")
        return MessageChunk(
            tool_calls=final_calls(self.calls),
            finish_reason=self.reason,
            usage=self.usage if self.profile.get("usage_mode", "none") != "none" else None,
        )
