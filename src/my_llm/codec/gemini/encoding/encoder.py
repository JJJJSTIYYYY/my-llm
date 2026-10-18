"""gemini: native input to intermediate."""

from typing import Any

from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.gemini.encoding.content import encode_part, json_schema, part_fields
from my_llm.codec.shared.conversion import (
    allocate_call_id,
    associate_tool_result,
    dictionaries,
    encode_functions,
    extras,
    mapping_list,
    remember,
    residual,
)
from my_llm.codec.shared.stream import batch, record
from my_llm.exceptions import ConfigurationError, ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "gemini"


class GeminiEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL

    def match_item(self, item: dict) -> bool:
        if "function_declarations" in item:
            return mapping_list(item["function_declarations"])
        return item.get("role", "user") in (
            "user",
            "model",
            "system",
            "developer",
        ) and mapping_list(item.get("parts"))

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        pending: list[dict] = []
        messages = dictionaries(messages)
        # Reserve IDs from later messages too, before synthesizing any missing IDs.
        used_ids = {
            part["function_call"]["id"]
            for message in messages
            for part in message["parts"]
            if part.get("function_call", {}).get("id")
        }
        for index, message in enumerate(messages):
            content = []
            for part_index, part in enumerate(message["parts"]):
                call = part.get("function_call")
                call_id = (
                    call.get("id") or allocate_call_id(used_ids, f"call_{index}_{part_index}")
                    if call is not None
                    else ""
                )
                block = encode_part(part, call_id)
                if block.get("type") == "tool_use":
                    pending.append(block)
                elif block.get("type") == "tool_result":
                    block["tool_use_id"] = associate_tool_result(
                        pending, block.get("tool_use_id"), block.get("name")
                    )
                content.append(block)
            result.append(
                remember(
                    {
                        **extras(message, "role", "parts"),
                        "role": "assistant"
                        if message.get("role") == "model"
                        else message.get("role", "user"),
                        "content": content,
                    },
                    [message],
                    PROTOCOL,
                    preserve=residual(message, {"role": True, "parts": [part_fields]}),
                )
            )
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return encode_functions(
            tools,
            "gemini",
            lambda tool: (
                [
                    {
                        **extras(f, "parameters_json_schema"),
                        "parameters": f["parameters_json_schema"]
                        if "parameters_json_schema" in f
                        else json_schema(f.get("parameters", {})),
                    }
                    for f in tool["function_declarations"]
                ]
                if "function_declarations" in tool
                else None
            ),
            consumed={
                "function_declarations": [
                    {
                        "name": True,
                        "description": True,
                        "parameters": True,
                        "parameters_json_schema": True,
                    }
                ]
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
            if not any(
                key in event
                for key in (
                    "candidates",
                    "usage_metadata",
                    "prompt_feedback",
                    "response_id",
                    "model_version",
                )
            ):
                result.append(batch(event, [{"type": "opaque", "value": event}], PROTOCOL))
                continue
            if event.get("prompt_feedback", {}).get("block_reason") not in (
                None,
                "BLOCK_REASON_UNSPECIFIED",
            ):
                raise ResponseFailedError(
                    "Gemini blocked the prompt", details=event["prompt_feedback"]
                )
            ops = [
                {
                    "type": "metadata",
                    "id": event.get("response_id"),
                    "model": event.get("model_version"),
                }
            ]
            if event.get("usage_metadata") is not None and (
                profile is None or profile.get("usage_mode") != "none"
            ):
                ops.append({"type": "usage", "usage": event["usage_metadata"]})
            for candidate in event.get("candidates", []):
                if candidate.get("index", 0) not in (None, 0):
                    raise ConfigurationError("only one generated candidate is supported")
                # GenerateContent Parts have no stable cross-chunk block address.
                # Append each part explicitly, including empty signed text parts.
                for part in candidate.get("content", {}).get("parts", []):
                    block = encode_part(part, "")
                    ops.append({"type": "block_append", "block": block})
                reason = candidate.get("finish_reason")
                if reason not in (None, "FINISH_REASON_UNSPECIFIED"):
                    if reason not in ("STOP", "MAX_TOKENS"):
                        raise ResponseFailedError("Gemini generation failed", details=candidate)
                    ops.append(
                        {
                            "type": "finish",
                            "reason": "length" if reason == "MAX_TOKENS" else "stop",
                            "allow_tail": ["metadata", "usage"],
                        }
                    )
            for operation in ops:
                if operation["type"] == "usage":
                    operation["usage"] = self.usage(operation["usage"])
            result.append(batch(event, ops, PROTOCOL, self._consumed_fields(event)))
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        thoughts = value.get("thoughts_token_count")
        return record(
            value,
            "gemini",
            input_base=value.get("prompt_token_count"),
            input_add={"tools": value.get("tool_use_prompt_token_count")},
            output_base=value.get("candidates_token_count"),
            output_add={"thinking": thoughts},
            total=value.get("total_token_count"),
            cached=value.get("cached_content_token_count"),
            reasoning=thoughts,
        )

    def _consumed_fields(self, event: dict) -> dict:
        return {
            "response_id": True,
            "model_version": True,
            "prompt_feedback": {"block_reason": True},
            "candidates": [
                {"index": True, "finish_reason": True, "content": {"parts": [part_fields]}}
            ],
        }
