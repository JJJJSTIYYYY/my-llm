"""openai/responses: native input to intermediate."""

from typing import Any

from openai.types.responses import ResponseInputItemParam, ToolParam

from my_llm.basic.exceptions import ResponseFailedError, ResponseIncompleteError
from my_llm.basic.semantic_loss import originated
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.openai.responses.encoding.content import (
    encode_content,
    tool_operations,
)
from my_llm.codec.shared.conversion import (
    arguments,
    mapping_list,
    select,
    tool_result_content,
)
from my_llm.codec.shared.stream import BlockRef, encode_batch, metadata, record
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_responses"
TOOL_RESULT_PARTS = frozenset(
    ("input_text", "input_image", "input_file", "input_audio", "output_text", "text", "refusal")
)


class ResponsesEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL
    message_type = ResponseInputItemParam
    tool_type = ToolParam

    def _validation_messages(self, messages: list) -> list:
        result = []
        for message in messages:
            kind = message.get("type", "message")
            if kind == "message":
                value = select(message, "type", "role", "id", "status")
                content = message.get("content")
                value["content"] = (
                    [
                        {
                            **select(block, "type", "text", "image_url"),
                            **({"annotations": []} if block.get("type") == "output_text" else {}),
                            **({"detail": "auto"} if block.get("type") == "input_image" else {}),
                        }
                        for block in content
                        if isinstance(block, dict)
                        and (
                            block.get("type") in ("input_text", "output_text")
                            or (
                                block.get("type") == "input_image" and message.get("role") == "user"
                            )
                        )
                    ]
                    if isinstance(content, list)
                    else content
                )
            elif kind == "function_call":
                value = select(message, "type", "id", "call_id", "name", "arguments", "status")
            elif kind == "function_call_output":
                value = select(message, "type", "id", "call_id", "status")
                output = tool_result_content(message.get("output"), native_types=TOOL_RESULT_PARTS)
                value["output"] = (
                    [
                        select(block, "type", "text", "image_url")
                        for block in output
                        if isinstance(block, dict)
                        and block.get("type") in ("input_text", "input_image")
                    ]
                    if isinstance(output, list)
                    else output
                )
            elif kind == "reasoning":
                value = select(message, "type", "id", "encrypted_content", "status")
                value["summary"] = [
                    select(part, "type", "text")
                    for part in message.get("summary", [])
                    if isinstance(part, dict) and part.get("type") == "summary_text"
                ]
            else:
                continue
            result.append(value)
        return result

    def _validation_tools(self, tools: list) -> list:
        return [
            select(tool, "type", "name", "description", "parameters", "strict")
            for tool in tools
            if tool.get("type") == "function"
        ]

    def match_item(self, item: dict) -> bool:
        kind = item.get("type")
        if kind == "function":
            return "function" not in item and isinstance(item.get("name"), str)
        if kind in {"function_call", "function_call_output", "reasoning", "item_reference"}:
            return True
        if kind not in (None, "message") or item.get("role") not in (
            "user",
            "assistant",
            "system",
            "developer",
        ):
            return False
        if any(
            key in item
            for key in {
                "parts",
                "tool_calls",
                "tool_call_id",
                "reasoning_content",
                "thinking",
                "images",
                "tool_name",
            }
        ):
            return False
        content = item.get("content")
        return (
            isinstance(content, str)
            or content is None
            or (
                mapping_list(content)
                and all(
                    block.get("type")
                    in {"input_text", "output_text", "input_image", "input_file", "refusal"}
                    for block in content
                )
            )
        )

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for index, item in enumerate(messages):
            path = f"$.messages[{index}]"
            kind = item.get("type", "message")
            role = item.get("role", "assistant") if kind == "message" else "assistant"
            if kind == "function_call":
                content = [
                    {
                        "type": "tool_use",
                        "id": item.get("call_id"),
                        "name": item.get("name"),
                        "input": arguments(item.get("arguments", {})),
                        **({"item_id": item["id"]} if "id" in item else {}),
                        **select(item, "status"),
                    }
                ]
            elif kind == "function_call_output":
                role = "user"
                output = tool_result_content(item.get("output"), native_types=TOOL_RESULT_PARTS)
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": item.get("call_id"),
                        "content": encode_content(output, path=f"{path}.output")
                        if isinstance(output, list)
                        else output,
                        **select(item, "id", "status"),
                    }
                ]
            elif kind == "reasoning":
                summary = [
                    select(part, "type", "text")
                    for part in item.get("summary", [])
                    if part.get("type") == "summary_text"
                ]
                content = [
                    {
                        "type": "thinking",
                        "thinking": "".join(part.get("text", "") for part in summary),
                        "summary": summary,
                        **select(item, "id", "status", "encrypted_content"),
                    }
                ]
            elif kind == "message":
                content = [
                    originated(
                        {
                            **block,
                            **({"item_id": item["id"]} if "id" in item else {}),
                            **({"item_status": item["status"]} if "status" in item else {}),
                        },
                        block.origin_path,
                        block.origin_value,
                    )
                    for block in encode_content(item.get("content"), path=f"{path}.content")
                ]
            else:
                continue
            if kind != "message":
                content = [originated(block, path, item) for block in content]
            result.append(
                {
                    "protocol": PROTOCOL,
                    "coalesce": True,
                    "role": role,
                    "content": content,
                    **(select(item, "id", "status") if kind == "message" else {}),
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
        return [
            {
                "functions": [
                    originated(
                        {
                            **select(tool, "name", "description", "strict"),
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
        result = []
        for event_index, event in enumerate(chunks):
            kind = event.get("type")
            if kind in ("response.failed", "response.incomplete", "error") or event.get("error"):
                error = (
                    ResponseIncompleteError
                    if kind == "response.incomplete"
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
            if kind in ("response.created", "response.in_progress"):
                ops.extend(self._response_metadata(event.get("response", {})))
                if kind == "response.created":
                    ops.append({"type": "start"})
            elif (
                kind == "response.output_item.added"
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
            elif kind == "response.function_call_arguments.delta":
                ops.extend(
                    tool_operations(
                        BlockRef("tool_use", index),
                        arguments=event.get("delta", ""),
                    )
                )
            elif (
                kind == "response.output_item.done"
                and event.get("item", {}).get("type") == "function_call"
            ):
                message = self._messages([event["item"]])[0]
                block = message["content"][0]
                ops.append(
                    {
                        "type": "block_set",
                        "ref": BlockRef("tool_use", index),
                        "block": block,
                    }
                )
            elif kind in ("response.content_part.added", "response.content_part.done"):
                part = event.get("part", {})
                if part.get("type") == "output_text":
                    ops.append(
                        {
                            "type": "block_set",
                            "ref": BlockRef("text", index, event.get("content_index", 0)),
                            "block": encode_content([part])[0],
                        }
                    )
            elif kind in (
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_part.done",
            ):
                part = event["part"]
                ops.append(
                    {
                        "type": "block_set",
                        "ref": BlockRef("thinking", index, event.get("summary_index", 0)),
                        "block": {
                            "type": "thinking",
                            "thinking": part.get("text", ""),
                            "summary": [select(part, "type", "text")],
                        },
                    }
                )
            elif kind == "response.completed":
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
                                    if item.get("type") in ("message", "reasoning", "function_call")
                                }
                            }
                        ),
                        self._snapshot(
                            output,
                            path=f"$.chunks[{parameters.get('chunk_offset', 0) + event_index}].response.output",
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
            for name, event_type, block_kind in (
                ("content", "response.output_text.delta", "text"),
                ("reasoning_content", "response.reasoning_summary_text.delta", "thinking"),
            ):
                if kind != event_type:
                    continue
                value = self.parameter_value(event, profile, name, "$.delta")
                if value is not None:
                    ops.append(
                        {
                            "type": "block_delta",
                            "ref": BlockRef(
                                block_kind,
                                index,
                                event.get(
                                    "summary_index"
                                    if block_kind == "thinking"
                                    else "content_index",
                                    0,
                                ),
                            ),
                            "kind": block_kind,
                            "field": "text" if block_kind == "text" else "thinking",
                            "value": value,
                        }
                    )
            if "output_index" in event:
                item = event.get("item") or {}
                item_id = item.get("id") or event.get("item_id")
                if item_id:
                    header = self._item_metadata(item) or {"id": item_id}
                    if not item:
                        if "reasoning_summary" in kind:
                            header["type"] = "reasoning"
                        elif "function_call_arguments" in kind:
                            header["type"] = "function_call"
                        elif kind.startswith(
                            ("response.output_text", "response.content_part", "response.refusal")
                        ):
                            header["type"] = "message"
                    ops.insert(
                        0,
                        metadata(header, path=("response_items", str(index))),
                    )
            for operation_index, operation in enumerate(ops):
                event_path = f"$.chunks[{parameters.get('chunk_offset', 0) + event_index}]"
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
            result.append(encode_batch(ops, PROTOCOL, usage=self.usage))
        return result

    def usage(self, value: dict | None) -> dict | None:
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
        return {
            "response_id" if key == "id" else key: event[key]
            for key in ("id", "model", "created_at")
            if isinstance(event.get(key), (str, int, float))
        }

    def _item_metadata(self, item: dict) -> dict:
        return {
            key: item[key]
            for key in ("id", "type", "status", "name", "call_id")
            if isinstance(item.get(key), str)
        }

    def _snapshot(
        self,
        output: list,
        *,
        path: str = "$.response.output",
    ) -> dict:
        """Address recognized final blocks with their live item/part coordinates."""
        blocks = []
        for index, item in enumerate(output):
            encoded = self._messages([item])
            if not encoded:
                continue
            message = encoded[0]
            for part, block in enumerate(message["content"]):
                kind = block["type"]
                origin_path = f"{path}[{index}]"
                if kind == "thinking":
                    for summary_index, summary in enumerate(block.get("summary") or [{}]):
                        blocks.append(
                            {
                                "ref": BlockRef(kind, index, summary_index),
                                "block": originated(
                                    {
                                        **block,
                                        "thinking": summary.get("text", ""),
                                        "summary": [summary] if summary else [],
                                    },
                                    origin_path,
                                    item,
                                ),
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
        return [metadata(self._metadata(response))]
