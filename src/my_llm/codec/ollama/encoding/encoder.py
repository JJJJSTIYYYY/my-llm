"""ollama: native input to intermediate."""

from typing import Any
from uuid import uuid4

from ollama import Message, Tool

from my_llm.basic.data import capture
from my_llm.basic.exceptions import ResponseFailedError
from my_llm.basic.semantic_loss import originated
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.ollama.encoding.content import image_data, media_type, tool_block
from my_llm.codec.ollama.schema import validate_ollama_values
from my_llm.codec.shared.conversion import (
    allocate_call_id,
    associate_tool_result,
    blocks,
    mapping_list,
    select,
    text_result,
)
from my_llm.codec.shared.stream import BlockRef, encode_batch, metadata, record
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "ollama"


class OllamaEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL
    message_type = Message
    tool_type = Tool

    def validate_values(self, values: list, schema: Any, *, filtered: bool = False) -> None:
        validate_ollama_values(values, schema, filtered=filtered)

    def _validation_messages(self, messages: list) -> list:
        result = []
        for message in messages:
            value = select(
                message, "role", "content", "thinking", "images", "tool_name", "tool_call_id"
            )
            if message.get("role") == "tool" and not isinstance(
                message.get("content"), (str, type(None))
            ):
                value["content"] = text_result(message.get("content"))
            if message.get("role") not in ("user", "tool"):
                value.pop("images", None)
            if message.get("tool_calls") is not None:
                value["tool_calls"] = [
                    {"function": select(call.get("function", {}), "name", "arguments", "index")}
                    for call in message["tool_calls"]
                    if isinstance(call, dict)
                ]
            result.append(value)
        return result

    def _validation_tools(self, tools: list) -> list:
        return [
            {
                "type": "function",
                "function": select(tool.get("function", {}), "name", "description", "parameters"),
            }
            for tool in tools
            if tool.get("type") == "function"
        ]

    def match_item(self, item: dict) -> bool:
        if item.get("type") == "function":
            return isinstance(item.get("function"), dict)
        if item.get("role") not in ("user", "assistant", "system", "tool"):
            return False
        if any(key in item for key in {"parts", "type", "reasoning_content"}):
            return False
        calls = item.get("tool_calls") or []
        if not mapping_list(calls) or any(
            not isinstance(call.get("function"), dict)
            or (
                call["function"].get("arguments") is not None
                and not isinstance(call["function"].get("arguments"), dict)
            )
            for call in calls
        ):
            return False
        return isinstance(item.get("content"), str) or item.get("content") is None

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        pending = []
        result = []
        id_prefix = f"{uuid4().hex}_" if parameters.get("enable_message_cache", False) else ""
        # Reserve explicit IDs first so ID-less calls cannot shadow a later native ID.
        used_ids = {
            call["id"]
            for message in messages
            for call in (message.get("tool_calls") or [])
            if call.get("id")
        }
        for index, message in enumerate(messages):
            native_content = message.get("content")
            if message.get("role") == "tool" and not isinstance(native_content, (str, type(None))):
                native_content = text_result(native_content)
            content = [
                originated(block, f"$.messages[{index}].content", message.get("content"))
                for block in blocks(native_content)
            ]
            for image_index, native_image in enumerate(message.get("images") or []):
                image = image_data(native_image)
                content.append(
                    originated(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type(image),
                                "data": image,
                            },
                        },
                        f"$.messages[{index}].images[{image_index}]",
                        native_image,
                    )
                )
            if message.get("thinking"):
                content.append(
                    originated(
                        {"type": "thinking", "thinking": message["thinking"], "mark": "raw"},
                        f"$.messages[{index}].thinking",
                        message["thinking"],
                    )
                )
            for call_index, call in enumerate(message.get("tool_calls") or []):
                call_id = call.get("id") or allocate_call_id(
                    used_ids, f"call_{id_prefix}{index}_{call_index}"
                )
                block = tool_block(call, call_id=call_id)
                content.append(
                    originated(block, f"$.messages[{index}].tool_calls[{call_index}]", call)
                )
                pending.append(block)
            role = message.get("role")
            if role == "tool":
                name = message.get("tool_name")
                call_id = associate_tool_result(pending, message.get("tool_call_id"), name)
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "name": name,
                        "content": content if message.get("images") else native_content,
                    }
                ]
                role = "user"
            result.append({"protocol": PROTOCOL, "role": role, "content": content})
        for message in result:
            if message.get("role") != "user":
                message["content"] = [
                    block for block in message["content"] if block.get("type") != "image"
                ]
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return [
            {
                "functions": [
                    originated(
                        {
                            **select(tool["function"], "name", "description"),
                            **(
                                {
                                    "parameters": originated(
                                        tool["function"]["parameters"],
                                        f"$.tools[{index}].function.parameters",
                                        tool["function"]["parameters"],
                                    )
                                }
                                if isinstance(tool["function"].get("parameters"), dict)
                                else {}
                            ),
                        },
                        f"$.tools[{index}].function",
                        tool["function"],
                    )
                ]
            }
            for index, tool in enumerate(tools)
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict)
        ]

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event_index, event in enumerate(chunks):
            if event.get("type") == "error" or event.get("error"):
                raise ResponseFailedError(
                    "provider generation failed", details=event.get("error", event)
                )
            if not any(key in event for key in ("message", "done")):
                continue
            ops = [
                metadata(
                    {"model": event["model"]}
                    if isinstance(event.get("model"), (str, int, float))
                    else {}
                )
            ]
            message = event.get("message", {})
            for name, kind, default in (
                ("content", "text", "$.message.content"),
                ("reasoning_content", "thinking", "$.message.thinking"),
            ):
                value = self.parameter_value(
                    event, profile, name, default, use_default=not event.get("done")
                )
                if value:
                    ops.append(
                        {
                            "type": "block_delta",
                            "ref": BlockRef(kind),
                            "field": kind,
                            "kind": kind,
                            "value": value,
                        }
                    )
            for call in message.get("tool_calls") or []:
                block = tool_block(call)
                block["input"] = capture(block["input"])
                ops.append({"type": "block_append", "block": block})
            if event.get("done"):
                ops.append(
                    {
                        "type": "usage",
                        "usage": select(event, "prompt_eval_count", "eval_count"),
                    }
                )
                ops.append({"type": "finish", "reason": event.get("done_reason", "stop")})
            for operation_index, operation in enumerate(ops):
                event_path = f"$.chunks[{parameters.get('chunk_offset', 0) + event_index}]"
                if "block" in operation and not hasattr(operation["block"], "origin_path"):
                    operation["block"] = originated(operation["block"], event_path, event)
                origin_path, origin_value = event_path, event
                if operation.get("type") == "block_delta" and operation.get("field") in (
                    "text",
                    "thinking",
                ):
                    name = "content" if operation["field"] == "text" else "reasoning_content"
                    native_field = "content" if name == "content" else "thinking"
                    binding = profile.parameter(name, "output") if profile is not None else None
                    suffix = binding.path[1:] if binding is not None else f".message.{native_field}"
                    origin_path, origin_value = event_path + suffix, operation.get("value")
                ops[operation_index] = originated(operation, origin_path, origin_value)
            result.append(encode_batch(ops, PROTOCOL, usage=self.usage))
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        return record(
            input_base=value.get("prompt_eval_count"),
            output_base=value.get("eval_count"),
        )
