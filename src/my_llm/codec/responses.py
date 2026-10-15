"""OpenAI Responses conversion and native reasoning/tool replay."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from openai import Omit

from ..exceptions import (
    IncompleteStreamError,
    MessageValidationError,
    ResponseFailedError,
    ResponseIncompleteError,
)
from ..profiles import ResolvedProfile
from ..types import MessageChunk
from .common import (
    path_value,
    plain,
    read,
    request_parameters,
    text_delta,
    validate_messages,
    validate_tool_call,
    validate_tools,
)


def _content(content: Any, role: str) -> Any:
    if isinstance(content, str):
        return content
    if content is None:
        return []
    result = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            if not isinstance(block.get("text"), str):
                raise MessageValidationError("text content blocks require text")
            result.append(
                {
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": block["text"],
                    **({"annotations": []} if role == "assistant" else {}),
                }
            )
        elif kind == "image_url" and role == "user":
            image = block.get("image_url", {})
            if not isinstance(image, dict) or not isinstance(image.get("url"), str):
                raise MessageValidationError("image_url needs a URL")
            result.append(
                {
                    "type": "input_image",
                    "image_url": image["url"],
                    "detail": image.get("detail", "auto"),
                }
            )
        else:
            raise MessageValidationError(f"unsupported Responses content block: {kind!r}")
    return result


class ResponsesCodec:
    def __init__(self, profile: ResolvedProfile) -> None:
        self.profile = profile

    def encode(
        self, messages: Any, *, tools: Any, previous_response_id: Any = None, **parameters: Any
    ) -> dict[str, Any]:
        history = []
        has_previous = previous_response_id is not None and not isinstance(
            previous_response_id, Omit
        )
        if has_previous and (not isinstance(previous_response_id, str) or not previous_response_id):
            raise MessageValidationError("previous_response_id must be a non-empty string")
        for message in validate_messages(messages, allow_external_calls=has_previous):
            role = message["role"]
            if role == "tool":
                output = message["content"]
                if not isinstance(output, str):
                    raise MessageValidationError("Responses tool output must be a string")
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": output,
                    }
                )
                continue
            if role == "assistant" and message.get("response_output"):
                # Native items preserve reasoning IDs/encrypted content and original function items.
                for item in message["response_output"]:
                    if item.get("type") not in ("message", "reasoning", "function_call"):
                        raise MessageValidationError(
                            "only message, reasoning and function_call replay items are supported"
                        )
                    history.append(deepcopy(item))
                continue
            content = message.get("content")
            if content is not None:
                history.append({"role": role, "content": _content(content, role)})
            for call in message.get("tool_calls", []):
                history.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
        encoded_tools = validate_tools(tools, self.profile)
        kwargs = request_parameters(self.profile, **parameters)
        kwargs.update(model=self.profile.model, input=history, stream=True)
        if not isinstance(encoded_tools, Omit):
            kwargs["tools"] = encoded_tools
        if has_previous:
            kwargs["previous_response_id"] = previous_response_id
        return kwargs

    def decoder(self) -> ResponsesDecoder:
        return ResponsesDecoder(self.profile)


class ResponsesDecoder:
    def __init__(self, profile: ResolvedProfile) -> None:
        self.profile = profile
        self.items: dict[int, dict[str, Any]] = {}
        self.completed: Any = None
        self.content = ""
        self.reasoning = ""
        self.closed = False

    def _extract(self, event: Any, selector: str | Omit) -> str:
        if isinstance(selector, Omit):
            return ""
        if selector.startswith("$"):
            return text_delta(path_value(event, selector))
        return text_delta(read(event, "delta")) if read(event, "type") == selector else ""

    def feed(self, event: Any) -> list[MessageChunk]:
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
        self.content += content
        self.reasoning += reasoning
        return (
            [MessageChunk(content=content, reasoning_content=reasoning)]
            if content or reasoning
            else []
        )

    def finish(self) -> MessageChunk:
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
            response_output=deepcopy(output),
            response_status="completed",
        )
