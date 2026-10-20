"""ollama: native input to intermediate."""

from copy import deepcopy
from typing import Any

from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.ollama.encoding.content import image_data, media_type, tool_block
from my_llm.codec.shared.conversion import (
    allocate_call_id,
    associate_tool_result,
    blocks,
    encode_functions,
    mapping_list,
    select,
)
from my_llm.codec.shared.stream import BlockRef, encode_batch, metadata, record
from my_llm.exceptions import ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "ollama"


class OllamaEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL

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
        # Reserve explicit IDs first so ID-less calls cannot shadow a later native ID.
        used_ids = {
            call["id"]
            for message in messages
            for call in (message.get("tool_calls") or [])
            if call.get("id")
        }
        for index, message in enumerate(messages):
            content = blocks(message.get("content"))
            content.extend(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type(image), "data": image},
                }
                for native_image in (message.get("images") or [])
                for image in [image_data(native_image)]
            )
            if message.get("thinking"):
                content.append({"type": "thinking", "thinking": message["thinking"]})
            for call_index, call in enumerate(message.get("tool_calls") or []):
                call_id = call.get("id") or allocate_call_id(used_ids, f"call_{index}_{call_index}")
                block = tool_block(call, call_id=call_id)
                content.append(block)
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
                        "content": message.get("content"),
                    }
                ]
                role = "user"
            result.append({"protocol": PROTOCOL, "role": role, "content": content})
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return encode_functions(
            tools,
            lambda tool: (
                [select(tool["function"], "name", "description", "parameters")]
                if tool.get("type") == "function" and "function" in tool
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
                            "value": deepcopy(value),
                        }
                    )
            for call in message.get("tool_calls") or []:
                block = tool_block(call)
                block["input"] = deepcopy(block["input"])
                ops.append({"type": "block_append", "block": block})
            if event.get("done"):
                if profile is None or profile.get("usage_mode") != "none":
                    ops.append(
                        {
                            "type": "usage",
                            "usage": select(event, "prompt_eval_count", "eval_count"),
                        }
                    )
                ops.append({"type": "finish", "reason": event.get("done_reason", "stop")})
            result.append(encode_batch(ops, PROTOCOL, usage=self.usage))
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        return record(
            input_base=value.get("prompt_eval_count"),
            output_base=value.get("eval_count"),
        )
