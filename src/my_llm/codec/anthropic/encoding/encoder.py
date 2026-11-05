"""anthropic: native input to intermediate."""

from typing import Any

from anthropic.types import MessageParam, ToolParam

import my_llm.codec.anthropic.constants as constants
from my_llm.basic.data import _MISSING, capture, dictionary_views
from my_llm.basic.exceptions import MessageValidationError, ResponseFailedError
from my_llm.basic.semantic_loss import originated
from my_llm.codec.anthropic.content import intermediate_block, native_block
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.shared.conversion import mapping_list, select
from my_llm.codec.shared.stream import (
    BlockRef,
    encode_batch,
    metadata,
    optional_usage,
    project_metadata,
    record,
)
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import OutputParameter, Parameter


class AnthropicEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = constants.PROTOCOL
    message_type = MessageParam
    tool_type = ToolParam

    def _validation_messages(self, messages: list) -> list:
        """Project supported Anthropic message fields before native schema validation."""
        result = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                projected = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result" and (
                        not isinstance(block.get("tool_use_id"), str) or not block["tool_use_id"]
                    ):
                        continue  # Discarded by the shared canonical history validator.
                    value = native_block(block)
                    if value is None or (
                        value.get("type") == "image" and message.get("role") != "user"
                    ):
                        continue
                    if value.get("type") == "thinking" and not value.get("signature"):
                        if not isinstance(value.get("thinking"), str):
                            raise MessageValidationError("thinking must be a string")
                        # A signature is a target replay constraint, not a reason
                        # to reject supported source reasoning during SDK checks.
                        continue
                    projected.append(value)
                content = projected
            result.append({"role": message.get("role"), "content": content})
        return result

    def _validation_tools(self, tools: list) -> list:
        """Project supported Anthropic function declarations before native schema validation."""
        return [select(tool, *constants.TOOL_FIELDS) for tool in tools if "input_schema" in tool]

    def match_item(self, item: dict) -> bool:
        """Whether a native message or tool has a recognizable Anthropic shape."""
        if "input_schema" in item:
            return isinstance(item["input_schema"], dict) and isinstance(item.get("name"), str)
        if item.get("role") not in constants.MESSAGE_ROLES or "content" not in item:
            return False
        if any(key in item for key in constants.OTHER_PROTOCOL_MESSAGE_FIELDS):
            return False
        content = item["content"]
        return (
            isinstance(content, str)
            or content is None
            or (
                mapping_list(content)
                and all(block.get("type") in constants.MESSAGE_BLOCK_TYPES for block in content)
            )
        )

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Normalize supported Anthropic blocks and reasoning provenance into middleware messages."""
        result = []
        binding = profile.parameters.get("feedback_reasoning_item") if profile is not None else None
        custom_feedback = isinstance(binding, Parameter) and binding.path != constants.FEEDBACK_PATH
        for index, message in enumerate(messages):
            role = message.get("role")
            path = f"$.messages[{index}].content"
            value = message.get("content")
            content = []
            if isinstance(value, str):
                content.append(originated({"type": "text", "text": value}, path, value))
            else:
                for part_index, item in enumerate(dictionary_views(value or [])):
                    block = intermediate_block(item, path=f"{path}[{part_index}]")
                    if block is None or role != "user" and block.get("type") == "image":
                        continue
                    if block["type"] == "thinking" and custom_feedback:
                        feedback_item = self.parameter_value(
                            {"content_block": item},
                            profile,
                            "feedback_reasoning_item",
                            constants.FEEDBACK_PATH,
                            _MISSING,
                        )
                        block.pop("feedback_item", None)
                        if feedback_item is not _MISSING:
                            block["feedback_item"] = feedback_item
                    content.append(block)
            result.append({"protocol": constants.PROTOCOL, "role": role, "content": content})
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Project supported Anthropic function declarations into canonical tool groups."""
        return [
            {
                "functions": [
                    originated(
                        {
                            **select(tool, *constants.NATIVE_FUNCTION_METADATA_FIELDS),
                            "parameters": originated(
                                tool.get("input_schema", {}),
                                f"$.tools[{index}].input_schema",
                                tool.get("input_schema", {}),
                            ),
                        },
                        f"$.tools[{index}]",
                        tool,
                    )
                ]
            }
            for index, tool in enumerate(tools)
            if "input_schema" in tool
        ]

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Translate native Anthropic events into middleware operation batches.

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
        reasoning_parameter = (
            profile.parameters.get("reasoning_content") if profile is not None else None
        )
        custom_reasoning = isinstance(reasoning_parameter, OutputParameter) and (
            reasoning_parameter.path != constants.REASONING_PATH or reasoning_parameter.event_type
        )
        chunk_offset = parameters.get("chunk_offset", 0)
        for event_index, event in enumerate(chunks):
            if event.get("type") == "error" or event.get("error"):
                raise ResponseFailedError(
                    "provider generation failed", details=event.get("error", event)
                )
            kind = event.get("type")
            index = event.get("index", 0)
            ops = []
            if kind == constants.MESSAGE_START:
                message = event.get("message")
                if not isinstance(message, dict):
                    message = {}
                ops.append(metadata(self._metadata(message)))
                if message.get("usage") is not None:
                    ops.append({"type": "usage", "usage": message["usage"]})
            elif kind == constants.CONTENT_BLOCK_START:
                block = intermediate_block(event["content_block"])
                if (
                    block is None
                    or block["type"] not in constants.MIDDLEWARE_TEXT_THINKING_TOOL_USE_TYPES
                ):
                    continue
                if block["type"] == "tool_use":
                    block["input"] = capture(block["input"])
                elif block["type"] == "thinking" and custom_feedback:
                    block.pop("feedback_item", None)
                ops.append(
                    {
                        "type": "block_set",
                        "ref": BlockRef("content", index),
                        "block": block,
                        **(
                            {"feedback_text_field": "thinking"}
                            if event["content_block"]["type"] == "thinking"
                            and not custom_feedback
                            and not custom_reasoning
                            else {}
                        ),
                    }
                )
            elif kind == constants.CONTENT_BLOCK_DELTA:
                delta = event["delta"]
                field = constants.DELTA_TYPE_TO_MIDDLEWARE_FIELD.get(delta.get("type"))
                if field in ("arguments", "feedback_item") and not (
                    field == "feedback_item" and custom_feedback
                ):
                    value = delta.get("partial_json" if field == "arguments" else "signature", "")
                    if value is not None and value is not _MISSING:
                        ops.append(
                            {
                                "type": "block_delta",
                                "ref": BlockRef("content", index),
                                "field": field,
                                "kind": constants.MIDDLEWARE_DELTA_FIELD_TO_BLOCK_TYPE.get(
                                    field, field
                                ),
                                "value": value,
                                **(
                                    {"path": constants.SIGNATURE_PATH}
                                    if field == "feedback_item"
                                    else {}
                                ),
                            }
                        )
                if field == "reasoning" and custom_reasoning and not custom_feedback:
                    native_reasoning = delta.get("thinking")
                    if native_reasoning is not None:
                        ops.append(
                            {
                                "type": "block_delta",
                                "ref": BlockRef("content", index),
                                "field": "feedback_item",
                                "path": constants.THINKING_PATH,
                                "kind": "thinking",
                                "value": native_reasoning,
                            }
                        )
            elif kind == constants.CONTENT_BLOCK_STOP:
                ops.append({"type": "block_stop", "ref": BlockRef("content", index)})
            elif kind == constants.MESSAGE_DELTA:
                delta = event.get("delta", {})
                if delta.get("stop_reason"):
                    reason = delta["stop_reason"]
                    ops.append(
                        {
                            "type": "reason",
                            "reason": constants.STOP_REASON_TO_MIDDLEWARE_REASON.get(
                                reason, reason
                            ),
                        }
                    )
                    ops.append(
                        metadata(
                            {
                                f"native_{key}": delta[key]
                                for key in constants.STOP_METADATA_FIELDS
                                if isinstance(delta.get(key), (str, int, float))
                            }
                        )
                    )
                if event.get("usage") is not None:
                    ops.append({"type": "usage", "usage": event["usage"]})
            elif kind == constants.MESSAGE_STOP:
                ops.append({"type": "finish"})
            for (
                name,
                delta_type,
                field,
                block_kind,
            ) in constants.CONTENT_REASONING_PARAMETER_DELTA_BLOCK_FIELDS:
                if not self.parameter_matches_event(
                    event,
                    profile,
                    name,
                    native=kind == constants.CONTENT_BLOCK_DELTA
                    and event.get("delta", {}).get("type") == delta_type,
                ):
                    continue
                value = self.parameter_value(
                    event,
                    profile,
                    name,
                    constants.TEXT_PATH if field == "text" else constants.REASONING_PATH,
                    _MISSING,
                )
                if value is not None and value is not _MISSING:
                    ops.append(
                        {
                            "type": "block_delta",
                            "ref": BlockRef("content", index),
                            "field": field,
                            "kind": block_kind,
                            "value": value,
                        }
                    )
            if custom_feedback and self.parameter_matches_event(
                event,
                profile,
                "feedback_reasoning_item",
                native=(
                    kind == constants.CONTENT_BLOCK_START
                    and event["content_block"].get("type")
                    in constants.THINKING_REDACTED_THINKING_TYPES
                    or kind == constants.CONTENT_BLOCK_DELTA
                    and event["delta"].get("type") in constants.THINKING_SIGNATURE_DELTA_TYPES
                    or kind == constants.CONTENT_BLOCK_STOP
                ),
            ):
                feedback_item = self.parameter_value(
                    event,
                    profile,
                    "feedback_reasoning_item",
                    constants.FEEDBACK_PATH,
                    _MISSING,
                    event=True,
                )
                if feedback_item is not _MISSING:
                    position = len(ops) - 1 if kind == constants.CONTENT_BLOCK_STOP else len(ops)
                    ops.insert(
                        position,
                        {
                            "type": "block_update",
                            "ref": BlockRef("content", index),
                            "block": {"type": "thinking", "feedback_item": capture(feedback_item)},
                        },
                    )
            if not ops:
                continue
            event_path = f"$.chunks[{chunk_offset + event_index}]"
            for operation_index, operation in enumerate(ops):
                if "block" in operation and not hasattr(operation["block"], "origin_path"):
                    operation["block"] = originated(operation["block"], event_path, event)
                origin_path, origin_value = event_path, event
                if operation.get("type") == "block_set":
                    operation["block"] = originated(
                        operation["block"],
                        event_path + ".content_block",
                        event.get("content_block", event),
                    )
                elif operation.get("type") == "block_delta":
                    field = operation.get("field")
                    native_field = constants.MIDDLEWARE_FIELD_TO_DELTA_VALUE_FIELD.get(field, field)
                    if (
                        field == "feedback_item"
                        and operation.get("path") == constants.THINKING_PATH
                    ):
                        native_field = "thinking"
                    origin_path, origin_value = (
                        event_path + f".delta.{native_field}",
                        operation.get("value"),
                    )
                ops[operation_index] = originated(operation, origin_path, origin_value)
            result.append(encode_batch(ops, constants.PROTOCOL, usage=self.usage))
        return result

    @optional_usage
    def usage(self, value: dict | None) -> dict | None:
        """Normalize reported Anthropic token counters, retaining absent usage as None."""
        if value is None:
            return None
        cached = value.get("cache_read_input_tokens")
        written = value.get("cache_creation_input_tokens")
        return record(
            input_base=value.get("input_tokens"),
            input_add={"cache_read": cached, "cache_write": written},
            output_base=value.get("output_tokens"),
            cached=cached,
            cache_write=written,
        )

    def _metadata(self, event: dict) -> dict:
        """Project supported Anthropic response identity and model metadata."""
        return project_metadata(event, constants.RESPONSE_FIELD_TO_MIDDLEWARE_METADATA)
