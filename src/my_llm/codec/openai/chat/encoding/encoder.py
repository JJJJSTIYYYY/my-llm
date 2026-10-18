"""openai/chat: native input to intermediate."""

from typing import Any

from my_llm.codec.base.encoder import BaseEncoder
from my_llm.codec.openai.chat.encoding.content import (
    CALL,
    content_fields,
    encode_content,
    tool_block,
    tool_operations,
)
from my_llm.codec.shared.conversion import (
    dictionaries,
    encode_functions,
    extras,
    mapping_list,
    remember,
    residual,
)
from my_llm.codec.shared.options import path_value
from my_llm.codec.shared.stream import BlockRef, batch, encode_batch, record
from my_llm.exceptions import ConfigurationError, ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "openai_chat"


class ChatCompletionsEncoder(BaseEncoder):
    """Convert native messages, tools, events and usage to intermediate values."""

    __slots__ = ()
    protocol = PROTOCOL

    def match_item(self, item: dict) -> bool:
        if item.get("type") == "function":
            return isinstance(item.get("function"), dict)
        if item.get("role") not in ("user", "assistant", "system", "developer", "tool"):
            return False
        if set(item) & {"parts", "type", "thinking", "images", "tool_name"}:
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
        for message in dictionaries(messages):
            content = encode_content(message.get("content"))
            # Appending preserves the original indices of content-block extensions.
            if message.get("reasoning_content"):
                content.append({"type": "thinking", "thinking": message["reasoning_content"]})
            if message.get("refusal") is not None:
                content.append({"type": "refusal", "refusal": message["refusal"]})
            content.extend(tool_block(c) for c in (message.get("tool_calls") or []))
            role = message.get("role")
            if role == "tool":
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id"),
                        "content": message.get("content"),
                        "name": message.get("name"),
                    }
                ]
                role = "user"
            result.append(
                remember(
                    {
                        **extras(
                            message,
                            "content",
                            "reasoning_content",
                            "refusal",
                            "tool_calls",
                            "tool_call_id",
                        ),
                        "role": role,
                        "content": content,
                    },
                    [message],
                    PROTOCOL,
                    preserve=residual(
                        message,
                        {
                            "role": True,
                            "content": [content_fields]
                            if isinstance(message.get("content"), list)
                            else True,
                            "tool_calls": [CALL],
                            "tool_call_id": True,
                            "reasoning_content": True,
                            "refusal": True,
                            "thinking": True,
                            "images": True,
                            "tool_name": True,
                        },
                    ),
                )
            )
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return encode_functions(
            tools,
            "openai_chat",
            lambda tool: (
                [tool["function"]]
                if tool.get("type") == "function" and "function" in tool
                else None
            ),
            consumed={
                "type": True,
                "function": {"name": True, "description": True, "parameters": True, "strict": True},
            },
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event in dictionaries(chunks):
            if event.get("type") == "error" or event.get("error"):
                raise ResponseFailedError(
                    "provider generation failed", details=event.get("error", event)
                )
            if not any(key in event for key in ("choices", "usage")):
                result.append(batch(event, [{"type": "opaque", "value": event}], PROTOCOL))
                continue
            ops = [{"type": "metadata", **self._metadata(event)}]
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
                for field, kind, default in (
                    ("content_path", "text", "$.choices[0].delta.content"),
                    ("reasoning_content_path", "thinking", "$.choices[0].delta.reasoning_content"),
                ):
                    value = path_value(event, profile.values[field] if profile else default)
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
                    ops.extend(tool_operations(BlockRef("tool_use", call.get("index", 0)), call))
                if choice.get("finish_reason") is not None:
                    ops.append(
                        {
                            "type": "finish",
                            "reason": choice["finish_reason"],
                            "allow_tail": ["metadata", "usage"],
                        }
                    )
            result.append(
                encode_batch(
                    event, ops, PROTOCOL, consumed=self._consumed_fields(event), usage=self.usage
                )
            )
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        return record(
            value,
            "openai_chat",
            input_base=value.get("prompt_tokens"),
            output_base=value.get("completion_tokens"),
            total=value.get("total_tokens"),
            cached=(value.get("prompt_tokens_details") or {}).get("cached_tokens"),
            reasoning=(value.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        )

    def _metadata(self, event: dict) -> dict:
        return {key: event[key] for key in ("id", "model", "created") if key in event}

    def _consumed_fields(self, event: dict) -> dict:
        return {
            **dict.fromkeys(("id", "model", "created"), True),
            "usage": {},
            "choices": [
                {
                    "index": True,
                    "finish_reason": True,
                    "delta": {
                        "content": True,
                        "reasoning_content": True,
                        "refusal": True,
                        "tool_calls": [{**CALL, "index": True}],
                    },
                }
            ],
        }
