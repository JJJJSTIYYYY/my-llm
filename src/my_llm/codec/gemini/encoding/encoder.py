"""gemini: native input to intermediate."""

from typing import Any
from uuid import uuid4

from google.genai import types

import my_llm.codec.gemini.constants as constants
from my_llm.basic.data import _MISSING, capture
from my_llm.basic.exceptions import ConfigurationError, ResponseFailedError
from my_llm.basic.semantic_loss import originated
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.gemini.encoding.content import encode_part, image_part, json_schema
from my_llm.codec.shared.conversion import (
    allocate_call_id,
    mapping_list,
    select,
    text_result,
)
from my_llm.codec.shared.stream import (
    encode_batch,
    metadata,
    optional_usage,
    project_metadata,
    record,
)
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Omit, OutputParameter, Parameter


class GeminiEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = constants.PROTOCOL
    message_type = types.Content
    tool_type = types.Tool

    def _validation_messages(self, messages: list) -> list:
        """Project supported Gemini message fields before native schema validation."""
        result = []
        for message in messages:
            parts = []
            for part in message.get("parts", []):
                if not isinstance(part, dict):
                    continue
                value = select(part, *constants.TEXT_PART_FIELDS)
                if "function_call" in part:
                    value["function_call"] = select(
                        part["function_call"], *constants.FUNCTION_CALL_FIELDS
                    )
                if "function_response" in part:
                    value["function_response"] = select(
                        part["function_response"], *constants.FUNCTION_RESPONSE_FIELDS
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
                for field in constants.INLINE_FILE_DATA_FIELDS:
                    media = part.get(field)
                    if (
                        message.get("role", "user") == "user"
                        and isinstance(media, dict)
                        and str(media.get("mime_type", "")).startswith("image/")
                    ):
                        value[field] = select(media, *constants.IMAGE_MEDIA_FIELDS)
                if value:
                    parts.append(value)
            result.append({**select(message, "role"), "parts": parts})
        return result

    def _validation_tools(self, tools: list) -> list:
        """Project supported Gemini function declarations before native schema validation."""
        return [
            {
                "function_declarations": [
                    select(function, *constants.FUNCTION_DECLARATION_FIELDS)
                    for function in tool["function_declarations"] or []
                ]
            }
            for tool in tools
            if "function_declarations" in tool
        ]

    def match_item(self, item: dict) -> bool:
        """Whether a native message or tool has a recognizable Gemini shape."""
        if "function_declarations" in item:
            return mapping_list(item["function_declarations"])
        return item.get("role", "user") in constants.MESSAGE_ROLES and mapping_list(
            item.get("parts")
        )

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Normalize Gemini parts while preserving explicit function-response IDs.

        Map model to assistant, allocate noncolliding IDs for ID-less calls, and
        retain missing result IDs for the shared history validator to discard.
        Preserve supported signatures and source provenance. Request preparation
        repairs incomplete and duplicate associations in its structural scan.
        """
        result = []
        binding = profile.parameters.get("feedback_reasoning_item") if profile is not None else None
        custom_feedback = isinstance(binding, Parameter) and binding.path != constants.FEEDBACK_PATH
        id_prefix = f"{uuid4().hex}_" if parameters.get("enable_message_cache", False) else ""
        # Reserve IDs from later messages too, before synthesizing any missing IDs.
        used_ids = {
            part["function_call"]["id"]
            for message in messages
            for part in message["parts"]
            if part.get("function_call", {}).get("id")
        }
        for index, message in enumerate(messages):
            role = "assistant" if message.get("role") == "model" else message.get("role", "user")
            content = []
            for part_index, part in enumerate(message["parts"]):
                call = part.get("function_call")
                call_id = (
                    call.get("id")
                    or allocate_call_id(used_ids, f"call_{id_prefix}{index}_{part_index}")
                    if call is not None
                    else ""
                )
                block = encode_part(part, call_id, path=f"$.messages[{index}].parts[{part_index}]")
                if block is None or block.get("type") == "image" and role != "user":
                    continue
                if block["type"] == "thinking" and custom_feedback:
                    feedback_item = self.parameter_value(
                        {"candidates": [{"content": {"parts": [part]}}]},
                        profile,
                        "feedback_reasoning_item",
                        constants.FEEDBACK_PATH,
                        _MISSING,
                    )
                    block.pop("feedback_item", None)
                    if feedback_item is not _MISSING:
                        block["feedback_item"] = feedback_item
                content.append(originated(block, f"$.messages[{index}].parts[{part_index}]", part))
            result.append(
                {
                    "protocol": constants.PROTOCOL,
                    "role": role,
                    "content": content,
                }
            )
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Project supported Gemini function declarations into canonical tool groups."""
        result = []
        for index, tool in enumerate(tools):
            functions = []
            for function_index, function in enumerate(tool.get("function_declarations") or []):
                path = f"$.tools[{index}].function_declarations[{function_index}]"
                value = select(function, *constants.NATIVE_FUNCTION_METADATA_FIELDS)
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
        """Translate native Gemini events into middleware operation batches.

        Extract supported content, metadata and usage; represent block identity
        and protocol completion explicitly for StreamState. Provider failures
        propagate as stream errors. Tool-argument assembly and public tool-event
        delivery belong to StreamState rather than this protocol parser.
        """
        result = []
        feedback_parameter = (
            profile.parameters.get("feedback_reasoning_item") if profile is not None else None
        )
        custom_feedback = isinstance(feedback_parameter, OutputParameter) and (
            feedback_parameter.path != constants.FEEDBACK_PATH or feedback_parameter.event_type
        )
        for event_index, event in enumerate(chunks):
            if event.get("type") == "error" or event.get("error"):
                raise ResponseFailedError(
                    "provider generation failed", details=event.get("error", event)
                )
            if not any(key in event for key in constants.RESPONSE_FIELDS) and not any(
                self.parameter_matches_event(event, profile, name, native=False)
                for name in ("content", "reasoning_content", "feedback_reasoning_item")
            ):
                continue
            if (
                event.get("prompt_feedback", {}).get("block_reason")
                not in constants.ABSENT_OR_UNSPECIFIED_BLOCK_REASONS
            ):
                raise ResponseFailedError(
                    "Gemini blocked the prompt", details=event["prompt_feedback"]
                )
            ops = [
                metadata(project_metadata(event, constants.RESPONSE_FIELD_TO_MIDDLEWARE_METADATA))
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
                    if (
                        block is not None
                        and block["type"] in constants.MIDDLEWARE_TEXT_THINKING_TOOL_USE_TYPES
                    ):
                        if block["type"] == "tool_use":
                            block["input"] = capture(block["input"])
                        # A standard text path uses each native part's thought flag.
                        # An overridden path addresses one exact SDK-root parameter.
                        if block["type"] == "thinking" and custom_feedback:
                            block.pop("feedback_item", None)
                        if block["type"] in ("text", "thinking"):
                            name = "reasoning_content" if block["type"] == "thinking" else "content"
                            seen_text.add(name)
                            if profile is not None:
                                parameter = profile.parameters.get(name)
                                if isinstance(parameter, OutputParameter) and (
                                    parameter.path != constants.TEXT_PATH or parameter.event_type
                                ):
                                    if block["type"] == "text":
                                        continue
                                    block["reasoning"] = None
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
                if reason not in constants.ABSENT_OR_UNSPECIFIED_FINISH_REASONS:
                    if reason not in constants.STOP_AND_MAX_TOKENS_FINISH_REASONS:
                        raise ResponseFailedError("Gemini generation failed", details=candidate)
                    ops.append(
                        {
                            "type": "finish",
                            "reason": "length"
                            if reason == constants.MAX_TOKENS_FINISH_REASON
                            else "stop",
                            "allow_tail": ["metadata", "usage"],
                        }
                    )
            if profile is not None:
                for name, kind, field in constants.CONTENT_REASONING_PARAMETER_BLOCK_FIELDS:
                    parameter = profile.parameters.get(name)
                    if not isinstance(parameter, OutputParameter):
                        continue
                    if not self.parameter_matches_event(event, profile, name):
                        continue
                    if parameter.path != constants.TEXT_PATH or parameter.event_type:
                        value = self.parameter_value(
                            event,
                            profile,
                            name,
                            constants.TEXT_PATH,
                            _MISSING,
                            use_default=has_parts or bool(parameter.event_type),
                            event=True,
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
                                "block": {
                                    "type": kind,
                                    field: value,
                                },
                            },
                        )
            if custom_feedback:
                feedback_item = self.parameter_value(
                    event,
                    profile,
                    "feedback_reasoning_item",
                    constants.FEEDBACK_PATH,
                    _MISSING,
                    use_default=has_parts or bool(feedback_parameter.event_type),
                    event=True,
                )
                if feedback_item is not _MISSING and not isinstance(feedback_item, Omit):
                    position = next(
                        (index for index, op in enumerate(ops) if op["type"] == "finish"), len(ops)
                    )
                    ops.insert(
                        position,
                        {
                            "type": "block_append",
                            "block": {
                                "type": "thinking",
                                "reasoning": None,
                                "feedback_item": capture(feedback_item),
                            },
                        },
                    )
            for operation_index, operation in enumerate(ops):
                event_path = f"$.chunks[{parameters.get('chunk_offset', 0) + event_index}]"
                if "block" in operation and not hasattr(operation["block"], "origin_path"):
                    operation["block"] = originated(operation["block"], event_path, event)
                ops[operation_index] = originated(operation, event_path, event)
            result.append(encode_batch(ops, constants.PROTOCOL, usage=self.usage))
        return result

    @optional_usage
    def usage(self, value: dict | None) -> dict | None:
        """Normalize reported Gemini token counters, retaining absent usage as None."""
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
