"""OpenAI Responses conversion and native reasoning/tool replay."""

from __future__ import annotations

from typing import Any

from openai import Omit

from my_llm.codec.common import (
    path_value,
    plain,
    read,
    request_parameters,
    text_delta,
)
from my_llm.codec.messages import encode_messages
from my_llm.codec.tools import encode_tools
from my_llm.codec.validation import validate_tool_call
from my_llm.exceptions import (
    IncompleteStreamError,
    MessageValidationError,
    ResponseFailedError,
    ResponseIncompleteError,
)
from my_llm.profiles import ResolvedProfile
from my_llm.types import MessageChunk


class ResponsesCodec:
    """Advanced adapter for encoding requests and decoding Responses streams.

    Most applications should use MyLLM.generate(). Direct callers provide a
    resolved profile and request-owned message dictionaries; encode() does not
    isolate mutable input. Create a fresh decoder for each response stream.
    """

    def __init__(self, profile: ResolvedProfile) -> None:
        """Use a validated profile returned by resolve_profiles()."""
        self.profile = profile

    def encode(
        self, messages: Any, *, tools: Any, previous_response_id: Any = None, **parameters: Any
    ) -> dict[str, Any]:
        """Encode messages/tools in either input format as SDK request keywords.

        parameters accepts generation reasoning, token and extra-field options.
        previous_response_id permits tool results whose calls are in server-side
        history. Invalid input raises ConfigurationError subclasses.
        """
        has_previous = previous_response_id is not None and not isinstance(
            previous_response_id, Omit
        )
        if has_previous and (not isinstance(previous_response_id, str) or not previous_response_id):
            raise MessageValidationError("previous_response_id must be a non-empty string")
        history = encode_messages(messages, self.profile, allow_external_calls=has_previous)
        encoded_tools = encode_tools(tools, self.profile)
        kwargs = request_parameters(self.profile, **parameters)
        kwargs.update(model=self.profile.model, input=history, stream=True)
        if not isinstance(encoded_tools, Omit):
            kwargs["tools"] = encoded_tools
        if has_previous:
            kwargs["previous_response_id"] = previous_response_id
        return kwargs

    def decoder(self) -> ResponsesDecoder:
        """Return a fresh stateful decoder for one Responses stream."""
        return ResponsesDecoder(self.profile)


class ResponsesDecoder:
    """Incrementally decode one stream and retain native output for replay."""

    def __init__(self, profile: ResolvedProfile) -> None:
        """Set delta selectors and usage settings from the resolved profile."""
        self.profile = profile
        self.items: dict[int, dict[str, Any]] = {}
        self.completed: Any = None
        self.closed = False

    def _extract(self, event: Any, selector: str | Omit) -> str:
        if isinstance(selector, Omit):
            return ""
        if selector.startswith("$"):
            return text_delta(path_value(event, selector))
        return text_delta(read(event, "delta")) if read(event, "type") == selector else ""

    def feed(self, event: Any) -> list[MessageChunk]:
        """Read one SDK event/dictionary and return any text/reasoning chunks.

        Native items and tool arguments are buffered. Failed/incomplete events
        raise ResponseFailedError/ResponseIncompleteError with native details.
        """
        kind = read(event, "type")
        if kind in ("response.failed", "error", "response.incomplete"):
            response = read(event, "response")
            details = plain(
                read(response, "incomplete_details")
                if kind == "response.incomplete"
                else read(response, "error", event)
            )
            error = (
                ResponseIncompleteError if kind == "response.incomplete" else ResponseFailedError
            )
            status = "incomplete" if kind == "response.incomplete" else "failed"
            partial = {"response_status": status, "finish_reason": status}
            if read(response, "id") is not None:
                partial["response_id"] = read(response, "id")
            raise error(
                f"Responses stream reported {kind}", details=details, partial_message=partial
            )
        if self.completed is not None:
            raise IncompleteStreamError("received events after response.completed")
        if kind == "response.completed":
            response = read(event, "response")
            if response is None or read(response, "status", "completed") != "completed":
                raise IncompleteStreamError("invalid response.completed event")
            self.completed = response
            output = read(response, "output", []) or []
            for index, item in enumerate(output):
                self.items[index] = plain(item)
            return []
        if kind in ("response.output_item.added", "response.output_item.done"):
            item = plain(read(event, "item"))
            index = read(event, "output_index")
            if not isinstance(index, int) or index < 0 or not isinstance(item, dict):
                raise IncompleteStreamError("invalid output item event")
            if item.get("type") not in ("message", "reasoning", "function_call"):
                raise IncompleteStreamError("provider returned a native cloud tool item")
            self.items[index] = item
        elif kind in (
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ):
            index = read(event, "output_index")
            if index not in self.items or self.items[index].get("type") != "function_call":
                raise IncompleteStreamError("function arguments refer to an unknown output item")
            item = self.items[index]
            if kind.endswith(".delta"):
                item["arguments"] = item.get("arguments", "") + text_delta(read(event, "delta"))
            else:
                item["arguments"] = text_delta(read(event, "arguments"))
        content = self._extract(event, self.profile.values["content_path"])
        reasoning = self._extract(event, self.profile.values["reasoning_content_path"])
        return (
            [MessageChunk(content=content, reasoning_content=reasoning)]
            if content or reasoning
            else []
        )

    def finish(self) -> MessageChunk:
        """Return final tools, status, usage and native Responses output items.

        Raises IncompleteStreamError for missing completion, invalid tool calls,
        unsupported output items or a repeated finish() call.
        """
        if self.closed:
            raise IncompleteStreamError("decoder was already finalized")
        self.closed = True
        if self.completed is None:
            raise IncompleteStreamError("Responses stream ended without response.completed")
        output = [self.items[index] for index in sorted(self.items)]
        calls = []
        for item in output:
            if item.get("type") == "function_call":
                if item.get("status", "completed") != "completed":
                    raise IncompleteStreamError("function call output item is incomplete")
                calls.append(
                    validate_tool_call(
                        {
                            "id": item.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": item.get("arguments"),
                            },
                        },
                        streamed=True,
                    )
                )
            elif item.get("type") not in ("message", "reasoning"):
                raise IncompleteStreamError("provider returned a native cloud tool item")
        if len({call["id"] for call in calls}) != len(calls):
            raise IncompleteStreamError("duplicate tool call IDs")
        return MessageChunk(
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            usage=plain(read(self.completed, "usage"))
            if self.profile.get("usage_mode", "none") == "native"
            else None,
            response_id=read(self.completed, "id"),
            response_output=output,
            response_status="completed",
        )
