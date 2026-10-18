"""ollama: native input to intermediate."""

from typing import Any

from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.ollama.encoding.content import media_type, tool_block
from my_llm.codec.shared.conversion import (
    allocate_call_id,
    associate_tool_result,
    blocks,
    dictionaries,
    encode_functions,
    extras,
    mapping_list,
    remember,
    residual,
)
from my_llm.codec.shared.stream import BlockRef, batch, record
from my_llm.exceptions import ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "ollama"
CALL = {"id": True, "type": True, "function": {"name": True, "arguments": True, "index": True}}


class OllamaEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL

    def match_item(self, item: dict) -> bool:
        if item.get("type") == "function":
            return isinstance(item.get("function"), dict)
        if item.get("role") not in ("user", "assistant", "system", "tool"):
            return False
        if set(item) & {"parts", "type", "reasoning_content"}:
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
        messages = dictionaries(messages)
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
                for image in (message.get("images") or [])
            )
            # Appending preserves the original indices of content-block extensions.
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
            result.append(
                remember(
                    {
                        **extras(
                            message,
                            "role",
                            "content",
                            "images",
                            "thinking",
                            "tool_calls",
                            "tool_name",
                            "tool_call_id",
                        ),
                        "role": role,
                        "content": content,
                    },
                    [message],
                    PROTOCOL,
                    preserve=residual(
                        message,
                        {
                            "role": True,
                            "content": True,
                            "tool_calls": [CALL],
                            "tool_call_id": True,
                            "reasoning_content": True,
                            "thinking": True,
                            "images": True,
                            "tool_name": True,
                        },
                    ),
                )
            )
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return encode_functions(
            tools,
            "ollama",
            lambda tool: (
                [tool["function"]]
                if tool.get("type") == "function" and "function" in tool
                else None
            ),
            consumed={
                "type": True,
                "function": {"name": True, "description": True, "parameters": True, "strict": True},
            },
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event in dictionaries(chunks):
            if event.get("type") == "error" or event.get("error"):
                raise ResponseFailedError(
                    "provider generation failed", details=event.get("error", event)
                )
            if not any(key in event for key in ("message", "done")):
                result.append(batch(event, [{"type": "opaque", "value": event}], PROTOCOL))
                continue
            ops = [{"type": "metadata", "model": event.get("model")}]
            message = event.get("message", {})
            for field, kind in (("content", "text"), ("thinking", "thinking")):
                if message.get(field):
                    ops.append(
                        {
                            "type": "block_delta",
                            "ref": BlockRef(kind),
                            "field": kind,
                            "kind": kind,
                            "value": message[field],
                        }
                    )
            for call in message.get("tool_calls") or []:
                ops.append({"type": "block_append", "block": tool_block(call)})
            if event.get("done"):
                if profile is None or profile.get("usage_mode") != "none":
                    ops.append(
                        {
                            "type": "usage",
                            "usage": {
                                k: v
                                for k, v in event.items()
                                if k.endswith(("_count", "_duration"))
                            },
                        }
                    )
                ops.append({"type": "finish", "reason": event.get("done_reason", "stop")})
            for operation in ops:
                if operation["type"] == "usage":
                    operation["usage"] = self.usage(operation["usage"])
            result.append(batch(event, ops, PROTOCOL, self._consumed_fields(event)))
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        return record(
            value,
            "ollama",
            input_base=value.get("prompt_eval_count"),
            output_base=value.get("eval_count"),
        )

    def _consumed_fields(self, event: dict) -> dict:
        return {
            "model": True,
            "done": True,
            "done_reason": True,
            "message": {"content": True, "thinking": True, "tool_calls": [CALL]},
            **{key: True for key in event if key.endswith(("_count", "_duration"))},
        }
