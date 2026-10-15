"""Recognize Chat messages and Responses items without copying the input history."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from my_llm.codec.utils.common import RESPONSE_BLOCKS
from my_llm.types import Message
from my_llm.types.exceptions import MessageValidationError


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    message: Message
    response_items: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class _AssistantItems:
    items: list[dict[str, Any]] = field(default_factory=list)
    content: list[dict[str, Any]] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def add(self, item: dict[str, Any]) -> None:
        kind = item.get("type")
        if kind == "function_call":
            if item.get("status") not in (None, "completed"):
                raise MessageValidationError("cannot replay an incomplete function call")
            self.calls.append(
                {
                    "id": item.get("call_id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments")},
                }
            )
        elif kind == "reasoning":
            self.reasoning.extend(_reasoning_text(item))
        else:
            content = item.get("content")
            if isinstance(content, str):
                self.content.append({"type": "output_text", "text": content, "annotations": []})
            elif isinstance(content, list) and all(isinstance(block, dict) for block in content):
                self.content.extend(content)
            else:
                raise MessageValidationError("Responses messages require text or content blocks")
        self.items.append(item)

    def finish(self) -> NormalizedMessage:
        message: Message = {"role": "assistant", "content": self.content or None}
        if self.calls:
            message["tool_calls"] = self.calls
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        return NormalizedMessage(message, self.items)


def normalize_messages(messages: Any) -> list[NormalizedMessage]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise MessageValidationError("messages must be a non-empty list")
    result = []
    assistant: _AssistantItems | None = None
    for item in messages:
        if not isinstance(item, dict):
            raise MessageValidationError("messages and Responses items must be dictionaries")
        native = _is_response_item(item)
        if native and _is_assistant_item(item):
            if assistant is None:
                assistant = _AssistantItems()
            assistant.add(item)
            continue
        if assistant is not None:
            result.append(assistant.finish())
            assistant = None
        result.append(_normalize_item(item) if native else NormalizedMessage(item))
    if assistant is not None:
        result.append(assistant.finish())
    return result


def _is_response_item(item: dict[str, Any]) -> bool:
    if "type" in item:
        if any(key in item for key in ("tool_calls", "tool_call_id", "response_output")):
            raise MessageValidationError("message mixes Responses and Chat Completions fields")
        return True
    if any(
        key in item
        for key in (
            "tool_calls",
            "tool_call_id",
            "response_output",
            "reasoning_content",
            "name",
            "refusal",
            "audio",
            "function_call",
        )
    ):
        return False
    if any(key in item for key in ("id", "status", "phase")):
        return True
    content = item.get("content")
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") in RESPONSE_BLOCKS for block in content
    )


def _is_assistant_item(item: dict[str, Any]) -> bool:
    return item.get("type") in ("function_call", "reasoning") or (
        item.get("type") in (None, "message") and item.get("role") == "assistant"
    )


def _normalize_item(item: dict[str, Any]) -> NormalizedMessage:
    kind = item.get("type")
    if kind in (None, "message"):
        if item.get("role") not in ("system", "user", "assistant"):
            raise MessageValidationError("invalid Responses message role")
        return NormalizedMessage({"role": item["role"], "content": item.get("content")}, [item])
    if kind == "function_call_output":
        return NormalizedMessage(
            {"role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output")},
            [item],
        )
    if kind == "item_reference":
        if not isinstance(item.get("id"), str) or not item["id"]:
            raise MessageValidationError("item_reference requires an ID")
        return NormalizedMessage({"role": "assistant", "content": None}, [item])
    raise MessageValidationError(f"unsupported Responses input item: {kind!r}")


def _reasoning_text(item: dict[str, Any]) -> list[str]:
    parts = item.get("summary") or item.get("content") or []
    if not isinstance(parts, list) or any(
        not isinstance(part, dict) or not isinstance(part.get("text"), str) for part in parts
    ):
        raise MessageValidationError("reasoning summary/content must contain text blocks")
    return [part["text"] for part in parts]
