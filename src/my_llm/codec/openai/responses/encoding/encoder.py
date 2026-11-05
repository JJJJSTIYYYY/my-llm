"""openai/responses: native input to intermediate."""

from typing import Any

from openai.types.responses import ResponseInputItemParam, ToolParam

import my_llm.codec.openai.responses.constants as constants
from my_llm.basic.data import capture
from my_llm.basic.exceptions import ResponseFailedError, ResponseIncompleteError
from my_llm.basic.semantic_loss import originated
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.openai.responses.encoding.content import (
    encode_content,
)
from my_llm.codec.shared.conversion import (
    arguments,
    mapping_list,
    select,
    tool_result_content,
)
from my_llm.codec.shared.stream import (
    BlockRef,
    encode_batch,
    metadata,
    optional_usage,
    project_metadata,
    record,
    tool_operations,
)
from my_llm.profile.resolution import ResolvedProfile


class ResponsesEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = constants.PROTOCOL
    message_type = ResponseInputItemParam
    tool_type = ToolParam

    def _validation_messages(self, messages: list) -> list:
        """Project supported Responses message fields before native schema validation."""
        result = []
        for message in messages:
            kind = message.get("type", "message")
            if kind == "function_call_output" and (
                not isinstance(message.get("call_id"), str) or not message["call_id"]
            ):
                continue  # Association repair owns missing result IDs, not SDK shape validation.
            if kind == "message":
                value = select(message, *constants.MESSAGE_FIELDS)
                content = message.get("content")
                value["content"] = (
                    [
                        {
                            **select(block, *constants.TEXT_IMAGE_PART_FIELDS),
                            **({"annotations": []} if block.get("type") == "output_text" else {}),
                            **({"detail": "auto"} if block.get("type") == "input_image" else {}),
                        }
                        for block in content
                        if isinstance(block, dict)
                        and (
                            block.get("type") in constants.INPUT_OUTPUT_TEXT_PART_TYPES
                            or (
                                block.get("type") == "input_image" and message.get("role") == "user"
                            )
                        )
                    ]
                    if isinstance(content, list)
                    else content
                )
            elif kind == "function_call":
                value = select(message, *constants.FUNCTION_CALL_FIELDS)
            elif kind == "function_call_output":
                value = select(message, *constants.FUNCTION_OUTPUT_FIELDS)
                output = tool_result_content(
                    message.get("output"), native_types=constants.TOOL_RESULT_NATIVE_PART_TYPES
                )
                value["output"] = (
                    [
                        select(block, *constants.TEXT_IMAGE_PART_FIELDS)
                        for block in output
                        if isinstance(block, dict)
                        and block.get("type") in constants.INPUT_TEXT_IMAGE_PART_TYPES
                    ]
                    if isinstance(output, list)
                    else output
                )
            elif kind == "reasoning":
                value = select(message, *constants.REASONING_ITEM_FIELDS)
                value["summary"] = [
                    select(part, *constants.SUMMARY_TEXT_PART_FIELDS)
                    for part in message.get("summary", [])
                    if isinstance(part, dict) and part.get("type") == "summary_text"
                ]
            else:
                continue
            result.append(value)
        return result

    def _validation_tools(self, tools: list) -> list:
        """Project supported Responses function declarations before native schema validation."""
        return [
            select(tool, *constants.TOOL_FIELDS) for tool in tools if tool.get("type") == "function"
        ]

    def match_item(self, item: dict) -> bool:
        """Whether a native message or tool has a recognizable Responses shape."""
        kind = item.get("type")
        if kind == "function":
            return "function" not in item and isinstance(item.get("name"), str)
        if kind in constants.FUNCTION_REASONING_REFERENCE_ITEM_TYPES:
            return True
        if kind not in (None, "message") or item.get("role") not in constants.MESSAGE_ROLES:
            return False
        if any(key in item for key in constants.OTHER_PROTOCOL_MESSAGE_FIELDS):
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
        """Normalize Responses input items into ordered middleware messages.

        Preserve native item identity, split reasoning and function items into
        canonical blocks, and parse function arguments as JSON objects. Mark
        compatible assistant fragments for downstream grouping while retaining
        their original item metadata. The decoder checks call/result associations.
        """
        result = []
        for index, item in enumerate(messages):
            path = f"$.messages[{index}]"
            kind = item.get("type", "message")
            role = item.get("role", "assistant") if kind == "message" else "assistant"
            identity = select(item, *constants.ITEM_ID_STATUS_FIELDS)
            item_identity = {"item": identity} if identity else {}
            if kind == "function_call":
                content = [
                    {
                        "type": "tool_use",
                        "id": item.get("call_id"),
                        "name": item.get("name"),
                        "input": arguments(item.get("arguments", {})),
                        **item_identity,
                    }
                ]
            elif kind == "function_call_output":
                role = "user"
                output = tool_result_content(
                    item.get("output"), native_types=constants.TOOL_RESULT_NATIVE_PART_TYPES
                )
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": item.get("call_id"),
                        "content": encode_content(output, path=f"{path}.output")
                        if isinstance(output, list)
                        else output,
                        **item_identity,
                    }
                ]
            elif kind == "reasoning":
                reasoning = "\n".join(
                    part.get("text", "")
                    for parts in (item.get("summary", []), item.get("content") or [])
                    for part in parts
                    if part.get("type") in constants.REASONING_TEXT_TYPES
                )
                content = [
                    {
                        "type": "thinking",
                        "reasoning": reasoning or None,
                        "feedback_item": self.parameter_value(
                            {"item": item},
                            profile,
                            "feedback_reasoning_item",
                            constants.FEEDBACK_PATH,
                        ),
                        **item_identity,
                    }
                ]
            elif kind == "message":
                content = [
                    originated(
                        {
                            **block,
                            **item_identity,
                        },
                        block.origin_path,
                        block.origin_value,
                    )
                    for block in encode_content(
                        item.get("content"), path=f"{path}.content", allow_images=role == "user"
                    )
                ]
            else:
                continue
            if kind != "message":
                content = [originated(block, path, item) for block in content]
            result.append(
                {
                    "protocol": constants.PROTOCOL,
                    "role": role,
                    "content": content,
                    **(item_identity if kind == "message" else {}),
                }
            )
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Project supported Responses function declarations into canonical tool groups."""
        return [
            {
                "functions": [
                    originated(
                        {
                            **select(tool, *constants.NATIVE_FUNCTION_METADATA_FIELDS),
                            **(
                                {
                                    "parameters": originated(
                                        tool["parameters"],
                                        f"$.tools[{index}].parameters",
                                        tool["parameters"],
                                    )
                                }
                                if isinstance(tool.get("parameters"), dict)
                                else {}
                            ),
                        },
                        f"$.tools[{index}]",
                        tool,
                    )
                ]
            }
            for index, tool in enumerate(tools)
            if tool.get("type") == "function"
        ]

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Translate native Responses events into middleware operation batches.

        Extract supported content, metadata and usage; represent block identity
        and protocol completion explicitly for StreamState. Provider failures
        propagate as stream errors. Tool-argument assembly and public tool-event
        delivery belong to StreamState rather than this protocol parser.
        """
        result = []
        chunk_offset = parameters.get("chunk_offset", 0)
        for event_index, event in enumerate(chunks):
            event_path = f"$.chunks[{chunk_offset + event_index}]"
            kind = event.get("type")
            if kind in constants.FAILED_INCOMPLETE_ERROR_EVENTS or event.get("error"):
                error = (
                    ResponseIncompleteError
                    if kind == constants.RESPONSE_INCOMPLETE
                    else ResponseFailedError
                )
                response = event.get("response", {})
                raise error(
                    "provider generation failed",
                    details=response.get(
                        "incomplete_details", response.get("error", event.get("error", event))
                    ),
                )
            kind = event.get("type", "")
            index = event.get("output_index", 0)
            ops = []
            if kind in constants.CREATED_IN_PROGRESS_EVENTS:
                ops.extend(self._response_metadata(event.get("response", {})))
                if kind == constants.RESPONSE_CREATED:
                    ops.append({"type": "start"})
            elif (
                kind == constants.OUTPUT_ITEM_ADDED
                and event.get("item", {}).get("type") == "function_call"
            ):
                item = event["item"]
                ops.extend(
                    tool_operations(
                        BlockRef("tool_use", index),
                        call_id=item.get("call_id"),
                        name=item.get("name"),
                        arguments=item.get("arguments", ""),
                    )
                )
            elif kind == constants.FUNCTION_CALL_ARGUMENTS_DELTA:
                ops.extend(
                    tool_operations(
                        BlockRef("tool_use", index),
                        arguments=event.get("delta", ""),
                    )
                )
            elif (
                kind == constants.OUTPUT_ITEM_DONE
                and event.get("item", {}).get("type") == "function_call"
            ):
                message = self._messages([event["item"]], profile)[0]
                block = message["content"][0]
                ops.append(
                    {
                        "type": "block_set",
                        "ref": BlockRef("tool_use", index),
                        "block": block,
                    }
                )
            elif kind in constants.CONTENT_PART_ADDED_DONE_EVENTS:
                part = event.get("part", {})
                if part.get("type") == "output_text":
                    ops.append(
                        {
                            "type": "block_set",
                            "ref": BlockRef("text", index, event.get("content_index", 0)),
                            "block": encode_content([part])[0],
                        }
                    )
            elif kind == constants.REASONING_SUMMARY_PART_ADDED and event.get("summary_index", 0):
                ops.append(
                    {
                        "type": "block_delta",
                        "ref": BlockRef("thinking", index),
                        "kind": "thinking",
                        "field": "reasoning",
                        "value": "\n",
                    }
                )
            elif (
                kind == constants.OUTPUT_ITEM_DONE
                and event.get("item", {}).get("type") == "reasoning"
            ):
                block = self._messages([event["item"]], profile)[0]["content"][0]
                block["feedback_item"] = capture(block["feedback_item"])
                ops.append(
                    {
                        "type": "block_set",
                        "ref": BlockRef("thinking", index),
                        "block": block,
                    }
                )
            elif kind == constants.RESPONSE_COMPLETED:
                response = event["response"]
                output = response.get("output", [])
                ops.extend(
                    [
                        *self._response_metadata(response),
                        metadata(
                            {
                                "response_items": {
                                    str(i): self._item_metadata(item)
                                    for i, item in enumerate(output)
                                    if item.get("type")
                                    in constants.MESSAGE_REASONING_FUNCTION_CALL_ITEM_TYPES
                                }
                            }
                        ),
                        self._snapshot(
                            output,
                            profile=profile,
                            path=f"{event_path}.response.output",
                        ),
                        {
                            "type": "finish",
                            "reason": "tool_calls"
                            if any(i.get("type") == "function_call" for i in output)
                            else "stop",
                            "allow_tail": ["usage"],
                        },
                    ]
                )
                if response.get("usage") is not None:
                    ops.append({"type": "usage", "usage": response["usage"]})
            for (
                name,
                event_types,
                block_kind,
            ) in constants.CONTENT_REASONING_PARAMETER_EVENT_BLOCK_TYPES:
                if not self.parameter_matches_event(
                    event, profile, name, native=kind in event_types
                ):
                    continue
                value = self.parameter_value(event, profile, name, constants.TEXT_PATH, event=True)
                if value is not None:
                    ops.append(
                        {
                            "type": "block_delta",
                            "ref": BlockRef(
                                block_kind,
                                index,
                                0 if block_kind == "thinking" else event.get("content_index", 0),
                            ),
                            "kind": block_kind,
                            "field": "text" if block_kind == "text" else "reasoning",
                            "value": value,
                        }
                    )
            if self.parameter_matches_event(
                event, profile, "feedback_reasoning_item", native=False
            ):
                feedback_item = self.parameter_value(
                    event, profile, "feedback_reasoning_item", constants.FEEDBACK_PATH
                )
                if feedback_item is not None:
                    ops.append(
                        {
                            "type": "block_update",
                            "ref": BlockRef("thinking", index),
                            "block": {
                                "type": "thinking",
                                "feedback_item": capture(feedback_item),
                            },
                        }
                    )
            if "output_index" in event:
                item = event.get("item") or {}
                item_id = item.get("id") or event.get("item_id")
                if item_id:
                    header = self._item_metadata(item) or {"id": item_id}
                    if not item:
                        if "reasoning_summary" in kind or "reasoning_text" in kind:
                            header["type"] = "reasoning"
                        elif "function_call_arguments" in kind:
                            header["type"] = "function_call"
                        elif kind.startswith(constants.MESSAGE_EVENT_PREFIXES):
                            header["type"] = "message"
                    ops.insert(
                        0,
                        metadata(header, path=("response_items", str(index))),
                    )
            for operation_index, operation in enumerate(ops):
                if "block" in operation and not hasattr(operation["block"], "origin_path"):
                    operation["block"] = originated(operation["block"], event_path, event)
                origin_path, origin_value = event_path, event
                if operation.get("type") == "block_delta":
                    origin_path, origin_value = event_path + ".delta", operation.get("value")
                elif operation.get("type") == "block_set" and "part" in event:
                    operation["block"] = originated(
                        operation["block"], event_path + ".part", event["part"]
                    )
                elif operation.get("type") == "block_set" and "item" in event:
                    operation["block"] = originated(
                        operation["block"], event_path + ".item", event["item"]
                    )
                ops[operation_index] = originated(operation, origin_path, origin_value)
            result.append(encode_batch(ops, constants.PROTOCOL, usage=self.usage))
        return result

    @optional_usage
    def usage(self, value: dict | None) -> dict | None:
        """Normalize reported Responses token counters, retaining absent usage as None."""
        if value is None:
            return None
        return record(
            input_base=value.get("input_tokens"),
            output_base=value.get("output_tokens"),
            total=value.get("total_tokens"),
            cached=(value.get("input_tokens_details") or {}).get("cached_tokens"),
            cache_write=(value.get("input_tokens_details") or {}).get("cache_write_tokens"),
            reasoning=(value.get("output_tokens_details") or {}).get("reasoning_tokens"),
        )

    def _metadata(self, event: dict) -> dict:
        """Project recognized response metadata into a middleware metadata operation."""
        return project_metadata(event, constants.RESPONSE_FIELD_TO_MIDDLEWARE_METADATA)

    def _item_metadata(self, item: dict) -> dict:
        """Record native output-item headers for identity-preserving stream rendering."""
        return project_metadata(item, constants.ITEM_FIELD_TO_MIDDLEWARE_METADATA)

    def _snapshot(
        self,
        output: list,
        *,
        profile: ResolvedProfile | None = None,
        path: str = constants.OUTPUT_PATH,
    ) -> dict:
        """Address recognized final blocks with their live item/part coordinates."""
        blocks = []
        for index, item in enumerate(output):
            encoded = self._messages([item], profile)
            if not encoded:
                continue
            message = encoded[0]
            for part, block in enumerate(message["content"]):
                kind = block["type"]
                origin_path = f"{path}[{index}]"
                if kind == "thinking":
                    block["feedback_item"] = capture(block["feedback_item"])
                    blocks.append(
                        {
                            "ref": BlockRef(kind, index),
                            "block": originated(block, origin_path, item),
                        }
                    )
                else:
                    origin_value = item
                    if item.get("type") == "message":
                        origin_path += block.origin_path.removeprefix("$.messages[0]")
                        origin_value = block.origin_value
                    blocks.append(
                        {
                            "ref": BlockRef(kind, index, part),
                            "block": originated(block, origin_path, origin_value),
                        }
                    )
        return {"type": "snapshot", "blocks": blocks}

    def _response_metadata(self, response: dict) -> list[dict]:
        """Project response-level and indexed output-item metadata from a snapshot."""
        return [metadata(self._metadata(response))]
