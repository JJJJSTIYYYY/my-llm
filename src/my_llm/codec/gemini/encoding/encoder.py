"""gemini: native input to intermediate."""

from typing import Any

from google.genai import types

from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.gemini.encoding.content import encode_part, image_part, json_schema
from my_llm.codec.shared.conversion import (
    _MISSING,
    allocate_call_id,
    associate_tool_result,
    mapping_list,
    select,
    text_result,
)
from my_llm.codec.shared.stream import encode_batch, metadata, record
from my_llm.exceptions import ConfigurationError, ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Omit
from my_llm.semantic_loss import originated

PROTOCOL = "gemini"
TEXT_PATH = "$.candidates[0].content.parts[0].text"


class GeminiEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL
    message_type = types.Content
    tool_type = types.Tool

    def _validation_messages(self, messages: list) -> list:
        result = []
        for message in messages:
            parts = []
            for part in message.get("parts", []):
                if not isinstance(part, dict):
                    continue
                value = select(part, "text", "thought", "thought_signature")
                if "function_call" in part:
                    value["function_call"] = select(part["function_call"], "id", "name", "args")
                if "function_response" in part:
                    value["function_response"] = select(
                        part["function_response"], "id", "name", "response"
                    )
                    if "parts" in part["function_response"]:
                        value["function_response"]["parts"] = [
                            image
                            for child in part["function_response"]["parts"] or []
                            if isinstance(child, dict) and (image := image_part(child)) is not None
                        ]
                    response = value["function_response"].get("response", {})
                    if not isinstance(response, dict):
                        value["function_response"]["response"] = {"result": text_result(response)}
                for field in ("inline_data", "file_data"):
                    media = part.get(field)
                    if (
                        message.get("role", "user") == "user"
                        and isinstance(media, dict)
                        and str(media.get("mime_type", "")).startswith("image/")
                    ):
                        value[field] = select(media, "mime_type", "data", "file_uri")
                if value:
                    parts.append(value)
            result.append({**select(message, "role"), "parts": parts})
        return result

    def _validation_tools(self, tools: list) -> list:
        return [
            {
                "function_declarations": [
                    select(function, "name", "description", "parameters", "parameters_json_schema")
                    for function in tool["function_declarations"] or []
                ]
            }
            for tool in tools
            if "function_declarations" in tool
        ]

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
                block = encode_part(part, call_id, path=f"$.messages[{index}].parts[{part_index}]")
                if block is None:
                    continue
                if block.get("type") == "tool_use":
                    pending.append(block)
                elif block.get("type") == "tool_result":
                    block["tool_use_id"] = associate_tool_result(
                        pending, block.get("tool_use_id"), block.get("name")
                    )
                content.append(originated(block, f"$.messages[{index}].parts[{part_index}]", part))
            result.append(
                {
                    "protocol": PROTOCOL,
                    "role": "assistant"
                    if message.get("role") == "model"
                    else message.get("role", "user"),
                    "content": content,
                }
            )
        for message in result:
            if message.get("role") != "user":
                message["content"] = [
                    block for block in message["content"] if block.get("type") != "image"
                ]
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for index, tool in enumerate(tools):
            functions = []
            for function_index, function in enumerate(tool.get("function_declarations") or []):
                path = f"$.tools[{index}].function_declarations[{function_index}]"
                value = select(function, "name", "description")
                if function.get("parameters_json_schema") is not None:
                    value["parameters"] = originated(
                        function["parameters_json_schema"],
                        f"{path}.parameters_json_schema",
                        function["parameters_json_schema"],
                    )
                elif function.get("parameters") is not None:
                    value["parameters"] = originated(
                        json_schema(function["parameters"]),
                        f"{path}.parameters",
                        function["parameters"],
                    )
                functions.append(originated(value, path, function))
            if functions:
                result.append({"functions": functions})
        return result

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event_index, event in enumerate(chunks):
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
            if event.get("usage_metadata") is not None:
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
                for part_index, part in enumerate(native_parts):
                    block = encode_part(part, "")
                    if block is not None and block["type"] in ("text", "thinking", "tool_use"):
                        # A standard text path uses each native part's thought flag.
                        # An overridden path addresses one exact SDK-root parameter.
                        if block["type"] in ("text", "thinking"):
                            name = "reasoning_content" if block["type"] == "thinking" else "content"
                            seen_text.add(name)
                            if profile is not None:
                                parameter = profile.parameter(name, "output")
                                if parameter is not None and parameter.path != TEXT_PATH:
                                    continue
                        ops.append(
                            {
                                "type": "block_append",
                                "block": originated(
                                    block,
                                    f"$.chunks[{parameters.get('chunk_offset', 0) + event_index}].candidates[0].content.parts[{part_index}]",
                                    part,
                                ),
                            }
                        )
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
                                "block": {"type": kind, field: value},
                            },
                        )
            for operation_index, operation in enumerate(ops):
                event_path = f"$.chunks[{parameters.get('chunk_offset', 0) + event_index}]"
                if "block" in operation and not hasattr(operation["block"], "origin_path"):
                    operation["block"] = originated(operation["block"], event_path, event)
                ops[operation_index] = originated(operation, event_path, event)
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
