"""gemini: native input to intermediate."""

from copy import deepcopy
from typing import Any

from google.genai import types

from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.gemini.encoding.content import encode_part, json_schema
from my_llm.codec.shared.conversion import (
    _MISSING,
    allocate_call_id,
    associate_tool_result,
    encode_functions,
    mapping_list,
    select,
)
from my_llm.codec.shared.stream import encode_batch, metadata, record
from my_llm.exceptions import ConfigurationError, ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Omit

PROTOCOL = "gemini"
TEXT_PATH = "$.candidates[0].content.parts[0].text"


class GeminiEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL
    message_type = types.Content
    tool_type = types.Tool

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
                if block is None:
                    continue
                if block.get("type") == "tool_use":
                    pending.append(block)
                elif block.get("type") == "tool_result":
                    block["tool_use_id"] = associate_tool_result(
                        pending, block.get("tool_use_id"), block.get("name")
                    )
                content.append(block)
            result.append(
                {
                    "protocol": PROTOCOL,
                    "role": "assistant"
                    if message.get("role") == "model"
                    else message.get("role", "user"),
                    "content": content,
                }
            )
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return encode_functions(
            tools,
            lambda tool: (
                [
                    {
                        **select(f, "name", "description", "response_json_schema", "behavior"),
                        "parameters": f["parameters_json_schema"]
                        if "parameters_json_schema" in f
                        else json_schema(f.get("parameters", {})),
                    }
                    for f in tool["function_declarations"]
                ]
                if "function_declarations" in tool
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
                continue
            if event.get("prompt_feedback", {}).get("block_reason") not in (
                None,
                "BLOCK_REASON_UNSPECIFIED",
            ):
                raise ResponseFailedError(
                    "Gemini blocked the prompt", details=event["prompt_feedback"]
                )
            ops = [
                metadata(
                    {
                        target: event[native]
                        for native, target in (
                            ("response_id", "response_id"),
                            ("model_version", "model"),
                        )
                        if isinstance(event.get(native), (str, int, float))
                    }
                )
            ]
            if event.get("usage_metadata") is not None and (
                profile is None or profile.get("usage_mode") != "none"
            ):
                ops.append({"type": "usage", "usage": event["usage_metadata"]})
            seen_text = set()
            has_parts = False
            for candidate in event.get("candidates", []):
                if candidate.get("index", 0) not in (None, 0):
                    raise ConfigurationError("only one generated candidate is supported")
                # GenerateContent Parts have no stable cross-chunk block address.
                # Append each part explicitly, including empty signed text parts.
                native_parts = candidate.get("content", {}).get("parts", [])
                has_parts = has_parts or bool(native_parts)
                for part in native_parts:
                    block = encode_part(part, "")
                    if block is not None:
                        # A standard text path uses each native part's thought flag.
                        # An overridden path addresses one exact SDK-root parameter.
                        if block["type"] in ("text", "thinking"):
                            name = "reasoning_content" if block["type"] == "thinking" else "content"
                            seen_text.add(name)
                            if profile is not None:
                                parameter = profile.parameter(name, "output")
                                if parameter is None or parameter.path != TEXT_PATH:
                                    continue
                        for key in ("input", "content", "part_metadata"):
                            if key in block:
                                block[key] = deepcopy(block[key])
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
            if profile is not None:
                for name, kind, field in (
                    ("content", "text", "text"),
                    ("reasoning_content", "thinking", "thinking"),
                ):
                    parameter = profile.parameter(name, "output")
                    if parameter is None:
                        continue
                    if parameter.path != TEXT_PATH:
                        value = self.parameter_value(
                            event, profile, name, TEXT_PATH, _MISSING, use_default=has_parts
                        )
                    elif name not in seen_text and has_parts:
                        value = parameter.default_value
                    else:
                        continue
                    if value is not _MISSING and not isinstance(value, Omit):
                        # Put configured text before a terminal finish operation.
                        position = next(
                            (index for index, op in enumerate(ops) if op["type"] == "finish"),
                            len(ops),
                        )
                        ops.insert(
                            position,
                            {
                                "type": "block_append",
                                "block": {"type": kind, field: deepcopy(value)},
                            },
                        )
            result.append(encode_batch(ops, PROTOCOL, usage=self.usage))
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        thoughts = value.get("thoughts_token_count")
        return record(
            input_base=value.get("prompt_token_count"),
            input_add={"tools": value.get("tool_use_prompt_token_count")},
            output_base=value.get("candidates_token_count"),
            output_add={"thinking": thoughts},
            total=value.get("total_token_count"),
            cached=value.get("cached_content_token_count"),
            reasoning=thoughts,
        )
