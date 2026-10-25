"""openai/chat: native input to intermediate."""

from typing import Any

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from my_llm.basic.exceptions import ConfigurationError, ResponseFailedError
from my_llm.basic.semantic_loss import originated
from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.openai.chat.encoding.content import (
    encode_content,
    tool_block,
    tool_operations,
)
from my_llm.codec.shared.conversion import (
    mapping_list,
    select,
    tool_result_content,
)
from my_llm.codec.shared.stream import BlockRef, encode_batch, metadata, record
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_chat"
TOOL_RESULT_PARTS = frozenset(("text", "image_url", "input_audio", "file", "refusal"))


class ChatCompletionsEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL
    message_type = ChatCompletionMessageParam
    tool_type = ChatCompletionToolParam

    def _validation_messages(self, messages: list) -> list:
        result = []
        for message in messages:
            value = select(message, "role", "name", "tool_call_id", "function_call")
            content = message.get("content")
            if message.get("role") in ("tool", "function"):
                content = tool_result_content(content, native_types=TOOL_RESULT_PARTS)
            value["content"] = (
                [
                    select(block, "type", "text")
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
                        **select(call, "id", "type"),
                        "function": select(call.get("function", {}), "name", "arguments"),
                    }
                    for call in message["tool_calls"]
                    if isinstance(call, dict) and call.get("type", "function") == "function"
                ]
            result.append(value)
        return result

    def _validation_tools(self, tools: list) -> list:
        return [
            {
                "type": "function",
                "function": select(
                    tool.get("function", {}), "name", "description", "parameters", "strict"
                ),
            }
            for tool in tools
            if tool.get("type") == "function"
        ]

    def match_item(self, item: dict) -> bool:
        if item.get("type") == "function":
            return isinstance(item.get("function"), dict)
        if item.get("role") not in ("user", "assistant", "system", "developer", "tool"):
            return False
        if any(key in item for key in {"parts", "type", "thinking", "images", "tool_name"}):
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
                and all(
                    block.get("type") in {"text", "image_url", "file", "input_audio", "refusal"}
                    for block in content
                )
            )
        )

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for index, message in enumerate(messages):
            path = f"$.messages[{index}]"
            role = message.get("role")
            native_content = message.get("content")
            if role in ("tool", "function"):
                native_content = tool_result_content(native_content, native_types=TOOL_RESULT_PARTS)
            content = (
                []
                if role in ("tool", "function")
                else encode_content(native_content, path=f"{path}.content")
            )
            if message.get("reasoning_content"):
                content.append(
                    originated(
                        {
                            "type": "thinking",
                            "thinking": message["reasoning_content"],
                            "mark": "raw",
                        },
                        f"{path}.reasoning_content",
                        message["reasoning_content"],
                    )
                )
            content.extend(
                originated(tool_block(call), f"{path}.tool_calls[{call_index}]", call)
                for call_index, call in enumerate(message.get("tool_calls") or [])
                if call.get("type", "function") == "function"
            )
            if message.get("function_call"):
                content.append(tool_block(message["function_call"], call_id="legacy_function_call"))
            if role in ("tool", "function"):
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id") or "legacy_function_call",
                        "content": encode_content(native_content, path=f"{path}.content")
                        if isinstance(native_content, list)
                        else native_content,
                        **select(message, "name"),
                    }
                ]
                role = "user"
            value = {
                "protocol": PROTOCOL,
                "role": role,
                "content": content,
                **select(message, "name"),
            }
            result.append(value)
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
                            **select(tool["function"], "name", "description", "strict"),
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
        result = []
        for event_index, event in enumerate(chunks):
            if event.get("type") == "error" or event.get("error"):
                raise ResponseFailedError(
                    "provider generation failed", details=event.get("error", event)
                )
            if not any(key in event for key in ("choices", "usage")):
                continue
            ops = [metadata(self._metadata(event))]
            if event.get("usage") is not None:
                ops.append({"type": "usage", "usage": event["usage"]})
            for choice in event.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise ConfigurationError("only one generated candidate is supported")
                delta = choice.get("delta", {})
                for name, kind, default in (
                    ("content", "text", "$.choices[0].delta.content"),
                    ("reasoning_content", "thinking", "$.choices[0].delta.reasoning_content"),
                ):
                    value = self.parameter_value(
                        event,
                        profile,
                        name,
                        default,
                        use_default=choice.get("finish_reason") is None,
                    )
                    if value:
                        ops.append(
                            {
                                "type": "block_delta",
                                "ref": BlockRef(kind),
                                "field": "text" if kind == "text" else "thinking",
                                "value": value,
                                "kind": kind,
                            }
                        )
                for call in delta.get("tool_calls") or []:
                    if call.get("type", "function") == "function":
                        ops.extend(
                            tool_operations(BlockRef("tool_use", call.get("index", 0)), call)
                        )
                if choice.get("finish_reason") is not None:
                    ops.append(
                        {
                            "type": "finish",
                            "reason": choice["finish_reason"],
                            "allow_tail": ["metadata", "usage"],
                        }
                    )
            for operation_index, operation in enumerate(ops):
                event_path = f"$.chunks[{parameters.get('chunk_offset', 0) + event_index}]"
                if "block" in operation and not hasattr(operation["block"], "origin_path"):
                    operation["block"] = originated(operation["block"], event_path, event)
                origin_path, origin_value = event_path, event
                if operation.get("type") == "block_delta" and operation.get("field") in (
                    "text",
                    "thinking",
                ):
                    name = "content" if operation["field"] == "text" else "reasoning_content"
                    binding = profile.parameter(name, "output") if profile is not None else None
                    suffix = (
                        binding.path[1:] if binding is not None else f".choices[0].delta.{name}"
                    )
                    origin_path, origin_value = event_path + suffix, operation.get("value")
                ops[operation_index] = originated(operation, origin_path, origin_value)
            result.append(encode_batch(ops, PROTOCOL, usage=self.usage))
        return result

    def usage(self, value: dict | None) -> dict | None:
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
        return {
            target: event[native]
            for native, target in (
                ("id", "response_id"),
                ("model", "model"),
                ("created", "created_at"),
            )
            if isinstance(event.get(native), (str, int, float))
        }
