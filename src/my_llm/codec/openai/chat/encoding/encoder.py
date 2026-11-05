"""openai/chat: native input to intermediate."""

from typing import Any
from uuid import uuid4

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

import my_llm.codec.openai.chat.constants as constants
from my_llm.basic.exceptions import ConfigurationError, ResponseFailedError
from my_llm.basic.semantic_loss import originated
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.openai.chat.encoding.content import (
    encode_content,
    tool_block,
)
from my_llm.codec.shared.conversion import (
    mapping_list,
    select,
    tool_result_content,
)
from my_llm.codec.shared.stream import (
    BlockRef,
    encode_batch,
    feedback_operation,
    metadata,
    optional_usage,
    project_metadata,
    record,
    tool_operations,
)
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Parameter


class ChatCompletionsEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = constants.PROTOCOL
    message_type = ChatCompletionMessageParam
    tool_type = ChatCompletionToolParam

    def _validation_messages(self, messages: list) -> list:
        """Project supported Chat Completions message fields before native schema validation."""
        result = []
        for message in messages:
            if message.get("role") in constants.TOOL_RESULT_ROLES and (
                not isinstance(message.get("tool_call_id"), str) or not message["tool_call_id"]
            ):
                continue  # The canonical history validator discards this entire message.
            value = select(message, *constants.MESSAGE_ROLE_NAME_TOOL_CALL_FIELDS)
            content = message.get("content")
            if message.get("role") in constants.TOOL_RESULT_ROLES:
                content = tool_result_content(
                    content, native_types=constants.TOOL_RESULT_NATIVE_PART_TYPES
                )
            value["content"] = (
                [
                    select(block, *constants.TEXT_PART_FIELDS)
                    if block.get("type") == "text"
                    else {"type": "image_url", "image_url": block.get("image_url")}
                    for block in content
                    if isinstance(block, dict)
                    and (
                        block.get("type") == "text"
                        or (block.get("type") == "image_url" and message.get("role") == "user")
                    )
                ]
                if isinstance(content, list)
                else content
            )
            if message.get("tool_calls") is not None:
                value["tool_calls"] = [
                    {
                        **select(call, *constants.TOOL_CALL_ID_TYPE_FIELDS),
                        "function": select(
                            call.get("function", {}), *constants.FUNCTION_CALL_FIELDS
                        ),
                    }
                    for call in message["tool_calls"]
                    if isinstance(call, dict) and call.get("type", "function") == "function"
                ]
            result.append(value)
        return result

    def _validation_tools(self, tools: list) -> list:
        """Project supported Chat Completions function declarations before native schema validation."""
        return [
            {
                "type": "function",
                "function": select(
                    tool.get("function", {}), *constants.FUNCTION_DECLARATION_FIELDS
                ),
            }
            for tool in tools
            if tool.get("type") == "function"
        ]

    def match_item(self, item: dict) -> bool:
        """Whether a native message or tool has a recognizable Chat Completions shape."""
        if item.get("type") == "function":
            return isinstance(item.get("function"), dict)
        if item.get("role") not in constants.MESSAGE_ROLES:
            return False
        if any(key in item for key in constants.OTHER_PROTOCOL_MESSAGE_FIELDS):
            return False
        calls = item.get("tool_calls") or []
        if not mapping_list(calls) or any(
            not isinstance(call.get("function"), dict)
            or isinstance(call["function"].get("arguments"), dict)
            for call in calls
        ):
            return False
        content = item.get("content")
        return (
            isinstance(content, str)
            or content is None
            or (
                mapping_list(content)
                and all(block.get("type") in constants.MESSAGE_CONTENT_TYPES for block in content)
            )
        )

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Normalize Chat messages and legacy function rounds into middleware history.

        Parse function arguments as JSON objects, turn role=tool/function messages
        into user-role tool_result blocks and attach source provenance. Unsupported
        source fields/content are omitted. Legacy call IDs are unique per batch
        when caching is enabled; association validation happens in the decoder.
        """
        result = []
        legacy_call_id = (
            f"legacy_function_call_{uuid4().hex}"
            if parameters.get("enable_message_cache", False)
            else "legacy_function_call"
        )
        feedback_binding = (
            profile.parameters.get("feedback_reasoning_item") if profile is not None else None
        )
        custom_feedback = (
            isinstance(feedback_binding, Parameter)
            and feedback_binding.path != constants.REASONING_PATH
        )
        for index, message in enumerate(messages):
            path = f"$.messages[{index}]"
            role = message.get("role")
            native_content = message.get("content")
            if role in constants.TOOL_RESULT_ROLES:
                native_content = tool_result_content(
                    native_content, native_types=constants.TOOL_RESULT_NATIVE_PART_TYPES
                )
            content = (
                []
                if role in constants.TOOL_RESULT_ROLES
                else encode_content(
                    native_content, path=f"{path}.content", allow_images=role == "user"
                )
            )
            reasoning = message.get("reasoning_content")
            feedback = (
                self.parameter_value(
                    {"choices": [{"delta": message}]},
                    profile,
                    "feedback_reasoning_item",
                    constants.REASONING_PATH,
                )
                if custom_feedback
                else reasoning
            )
            if reasoning is not None or feedback is not None:
                content.append(
                    originated(
                        {"type": "thinking", "reasoning": reasoning, "feedback_item": feedback},
                        f"{path}.reasoning_content",
                        message.get("reasoning_content"),
                    )
                )
            content.extend(
                originated(tool_block(call), f"{path}.tool_calls[{call_index}]", call)
                for call_index, call in enumerate(message.get("tool_calls") or [])
                if call.get("type", "function") == "function"
            )
            if message.get("function_call"):
                content.append(tool_block(message["function_call"], call_id=legacy_call_id))
            if role in constants.TOOL_RESULT_ROLES:
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id"),
                        "content": encode_content(native_content, path=f"{path}.content")
                        if isinstance(native_content, list)
                        else native_content,
                        **select(message, "name"),
                    }
                ]
                role = "user"
            value = {
                "protocol": constants.PROTOCOL,
                "role": role,
                "content": content,
                **select(message, "name"),
            }
            result.append(value)
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Project supported Chat Completions function declarations into canonical tool groups."""
        return [
            {
                "functions": [
                    originated(
                        {
                            **select(tool["function"], *constants.NATIVE_FUNCTION_METADATA_FIELDS),
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
        """Translate native Chat Completions events into middleware operation batches.

        Extract supported content, metadata and usage; represent block identity
        and protocol completion explicitly for StreamState. Provider failures
        propagate as stream errors. Tool-argument assembly and public tool-event
        delivery belong to StreamState rather than this protocol parser.
        """
        result = []
        chunk_offset = parameters.get("chunk_offset", 0)
        for event_index, event in enumerate(chunks):
            if event.get("type") == "error" or event.get("error"):
                raise ResponseFailedError(
                    "provider generation failed", details=event.get("error", event)
                )
            custom_event = any(
                self.parameter_matches_event(event, profile, name, native=False)
                for name in ("content", "reasoning_content", "feedback_reasoning_item")
            )
            if not any(key in event for key in constants.RESPONSE_FIELDS) and not custom_event:
                continue
            ops = [metadata(self._metadata(event))]
            if event.get("usage") is not None:
                ops.append({"type": "usage", "usage": event["usage"]})
            for choice in event.get("choices") or ([{}] if custom_event else []):
                if choice.get("index", 0) != 0:
                    raise ConfigurationError("only one generated candidate is supported")
                delta = choice.get("delta", {})
                for name, kind, default in constants.CONTENT_REASONING_PARAMETER_BLOCK_PATHS:
                    if not self.parameter_matches_event(
                        event, profile, name, native=bool(event.get("choices"))
                    ):
                        continue
                    value = self.parameter_value(
                        event,
                        profile,
                        name,
                        default,
                        use_default=choice.get("finish_reason") is None,
                        event=True,
                    )
                    if value:
                        ops.append(
                            {
                                "type": "block_delta",
                                "ref": BlockRef(kind),
                                "field": "text" if kind == "text" else "reasoning",
                                "value": value,
                                "kind": kind,
                            }
                        )
                feedback = None
                if self.parameter_matches_event(
                    event, profile, "feedback_reasoning_item", native=bool(event.get("choices"))
                ):
                    feedback = self.parameter_value(
                        event,
                        profile,
                        "feedback_reasoning_item",
                        constants.REASONING_PATH,
                        use_default=choice.get("finish_reason") is None,
                    )
                if feedback is not None:
                    ops.append(feedback_operation(BlockRef("thinking"), feedback))
                for call in delta.get("tool_calls") or []:
                    if call.get("type", "function") == "function":
                        function = call.get("function", {})
                        ops.extend(
                            tool_operations(
                                BlockRef("tool_use", call.get("index", 0)),
                                call_id=call.get("id"),
                                name=function.get("name"),
                                arguments=function.get("arguments"),
                            )
                        )
                if choice.get("finish_reason") is not None:
                    ops.append(
                        {
                            "type": "finish",
                            "reason": choice["finish_reason"],
                            "allow_tail": ["metadata", "usage"],
                        }
                    )
            event_path = f"$.chunks[{chunk_offset + event_index}]"
            for operation_index, operation in enumerate(ops):
                if "block" in operation and not hasattr(operation["block"], "origin_path"):
                    operation["block"] = originated(operation["block"], event_path, event)
                origin_path, origin_value = event_path, event
                if operation.get("type") == "block_delta" and operation.get("field") in (
                    "text",
                    "reasoning",
                ):
                    name = "content" if operation["field"] == "text" else "reasoning_content"
                    binding = profile.parameter(name, "output") if profile is not None else None
                    suffix = (
                        binding.path[1:] if binding is not None else f".choices[0].delta.{name}"
                    )
                    origin_path, origin_value = event_path + suffix, operation.get("value")
                ops[operation_index] = originated(operation, origin_path, origin_value)
            result.append(encode_batch(ops, constants.PROTOCOL, usage=self.usage))
        return result

    @optional_usage
    def usage(self, value: dict | None) -> dict | None:
        """Normalize reported Chat Completions token counters, retaining absent usage as None."""
        if value is None:
            return None
        return record(
            input_base=value.get("prompt_tokens"),
            output_base=value.get("completion_tokens"),
            total=value.get("total_tokens"),
            cached=(value.get("prompt_tokens_details") or {}).get("cached_tokens"),
            reasoning=(value.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        )

    def _metadata(self, event: dict) -> dict:
        """Project supported Chat response identity, model and creation metadata."""
        return project_metadata(event, constants.RESPONSE_FIELD_TO_MIDDLEWARE_METADATA)
