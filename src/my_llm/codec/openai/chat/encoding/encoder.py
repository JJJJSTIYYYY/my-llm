"""openai/chat: native input to intermediate."""

from copy import deepcopy
from typing import Any

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.openai.chat.encoding.content import (
    encode_content,
    tool_block,
    tool_operations,
)
from my_llm.codec.openai.chat.utils.content import annotations, text_blocks
from my_llm.codec.shared.conversion import (
    encode_functions,
    mapping_list,
    select,
)
from my_llm.codec.shared.stream import BlockRef, encode_batch, metadata, record
from my_llm.exceptions import ConfigurationError, ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_chat"


class ChatCompletionsEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL
    message_type = ChatCompletionMessageParam
    tool_type = ChatCompletionToolParam

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
        for message in messages:
            role = message.get("role")
            native_content = message.get("content")
            content = [] if role in ("tool", "function") else encode_content(native_content)
            if message.get("reasoning_content"):
                content.append({"type": "thinking", "thinking": message["reasoning_content"]})
            if message.get("refusal") is not None:
                content.append({"type": "refusal", "refusal": message["refusal"]})
            content.extend(tool_block(call) for call in message.get("tool_calls") or [])
            if message.get("function_call"):
                content.append(tool_block(message["function_call"], call_id="legacy_function_call"))
            if role in ("tool", "function"):
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id") or "legacy_function_call",
                        "content": text_blocks(native_content)
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
            if message.get("annotations"):
                value["annotations"] = annotations(message["annotations"])
            if message.get("audio") is not None:
                value["audio"] = select(message["audio"], "id", "expires_at", "data", "transcript")
            result.append(value)
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return encode_functions(
            tools,
            lambda tool: (
                [select(tool["function"], "name", "description", "parameters", "strict")]
                if tool.get("type") == "function" and "function" in tool
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
            if not any(key in event for key in ("choices", "usage")):
                continue
            ops = [metadata(self._metadata(event))]
            if event.get("usage") is not None and (
                profile is None or profile.get("usage_mode") != "none"
            ):
                ops.append({"type": "usage", "usage": event["usage"]})
            for choice in event.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise ConfigurationError("only one generated candidate is supported")
                delta = choice.get("delta", {})
                if delta.get("refusal") is not None:
                    ops.append(
                        {
                            "type": "block_delta",
                            "ref": BlockRef("refusal"),
                            "field": "refusal",
                            "kind": "refusal",
                            "value": delta["refusal"],
                        }
                    )
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
                                "value": deepcopy(value),
                                "kind": kind,
                            }
                        )
                for call in delta.get("tool_calls") or []:
                    ops.extend(tool_operations(BlockRef("tool_use", call.get("index", 0)), call))
                if choice.get("finish_reason") is not None:
                    ops.append(
                        {
                            "type": "finish",
                            "reason": choice["finish_reason"],
                            "allow_tail": ["metadata", "usage"],
                        }
                    )
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
            "response_id" if key == "id" else key: event[key]
            for key in ("id", "model", "created")
            if isinstance(event.get(key), (str, int, float))
        }
