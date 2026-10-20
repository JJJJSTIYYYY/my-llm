"""openai/responses: native input to intermediate."""

from copy import deepcopy
from typing import Any

from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.openai.responses.encoding.content import (
    encode_content,
    tool_operations,
)
from my_llm.codec.openai.responses.utils.metadata import response_parameters
from my_llm.codec.shared.conversion import (
    arguments,
    dictionary_views,
    encode_functions,
    mapping_list,
    select,
)
from my_llm.codec.shared.stream import BlockRef, encode_batch, metadata, record
from my_llm.exceptions import ResponseFailedError, ResponseIncompleteError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_responses"


class ResponsesEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL

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
        for item in messages:
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
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": item.get("call_id"),
                        "content": encode_content(item["output"])
                        if isinstance(item.get("output"), list)
                        else item.get("output"),
                        **select(item, "id", "status"),
                    }
                ]
            elif kind == "reasoning":
                summary = [select(part, "type", "text") for part in item.get("summary", [])]
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
                    {
                        **block,
                        **({"item_id": item["id"]} if "id" in item else {}),
                        **({"item_status": item["status"]} if "status" in item else {}),
                    }
                    for block in encode_content(item.get("content"))
                ]
            elif kind == "item_reference":
                content = [{"type": "item_reference", **select(item, "id")}]
            else:
                continue
            result.append(
                {
                    "protocol": PROTOCOL,
                    "coalesce": kind != "item_reference",
                    "role": role,
                    "content": content,
                    **(select(item, "id", "status") if kind == "message" else {}),
                }
            )
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return encode_functions(
            tools,
            lambda tool: (
                [select(tool, "name", "description", "parameters", "strict")]
                if tool.get("type") == "function"
                else None
            ),
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event in chunks:
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
                message = next(dictionary_views(self._messages([event["item"]])))
                block = message["content"][0]
                ops.append(
                    {
                        "type": "block_set",
                        "ref": BlockRef("tool_use", index),
                        "block": deepcopy(block),
                    }
                )
            elif kind in ("response.content_part.added", "response.content_part.done"):
                part = event.get("part", {})
                if part.get("type") in ("output_text", "refusal"):
                    ops.append(
                        {
                            "type": "block_set",
                            "ref": BlockRef(
                                "refusal" if part["type"] == "refusal" else "text",
                                index,
                                event.get("content_index", 0),
                            ),
                            "block": deepcopy(encode_content([part])[0]),
                        }
                    )
            elif kind == "response.refusal.delta":
                ops.append(
                    {
                        "type": "block_delta",
                        "ref": BlockRef("refusal", index, event.get("content_index", 0)),
                        "kind": "refusal",
                        "field": "refusal",
                        "value": event["delta"],
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
                        self._snapshot(output),
                        {
                            "type": "finish",
                            "reason": "tool_calls"
                            if any(i.get("type") == "function_call" for i in output)
                            else "stop",
                            "allow_tail": ["usage"],
                        },
                    ]
                )
                if response.get("usage") is not None and (
                    profile is None or profile.get("usage_mode") != "none"
                ):
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
                            "value": deepcopy(value),
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

    def _snapshot(self, output: list) -> dict:
        """Address recognized final blocks with their live item/part coordinates."""
        blocks = []
        for index, item in enumerate(output):
            encoded = self._messages([item])
            if not encoded:
                continue
            message = encoded[0]
            for part, block in enumerate(message.get("content", [])):
                block = deepcopy(block)
                kind = block["type"]
                if kind == "thinking":
                    for summary_index, summary in enumerate(block.get("summary") or [{}]):
                        blocks.append(
                            {
                                "ref": BlockRef(kind, index, summary_index),
                                "block": {
                                    **block,
                                    "thinking": summary.get("text", ""),
                                    "summary": [summary] if summary else [],
                                },
                            }
                        )
                else:
                    blocks.append({"ref": BlockRef(kind, index, part), "block": block})
        return {"type": "snapshot", "blocks": blocks}

    def _response_metadata(self, response: dict) -> list[dict]:
        return [
            metadata(self._metadata(response)),
            metadata(response_parameters(response), path=("response_parameters",)),
        ]
