"""anthropic: intermediate to native output."""

from copy import deepcopy
from typing import Any

from my_llm.codec.anthropic.decoding.content import foreign_block
from my_llm.codec.anthropic.utils.content import CITATION_FIELDS, native_block
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.shared.conversion import (
    blocks,
    coalesce_messages,
    decode_functions,
    json_arguments,
    project_message,
    select,
    select_present,
    source_protocol,
)
from my_llm.codec.shared.stream import counts, events
from my_llm.exceptions import ResponseFailedError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "anthropic"
REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class AnthropicDecoder(BaseDecoder):
    """Translate recognized intermediate values into protocol dictionaries."""

    __slots__ = ()
    protocol = PROTOCOL

    body_overlay = "extra_body"

    def assemble_request(
        self,
        messages: list,
        tools: list | None,
        profile: ResolvedProfile,
        *,
        request: dict,
        **parameters: Any,
    ) -> dict:
        result = request
        system = [m for m in messages if m.get("role") in ("system", "developer")]
        history = [m for m in messages if m.get("role") not in ("system", "developer")]
        result.update(model=profile.model, stream=True, messages=history)
        if system:
            result["system"] = [
                block
                for message in system
                for block in (
                    [{"type": "text", "text": message["content"]}]
                    if isinstance(message["content"], str)
                    else message["content"]
                )
            ]
        if tools is not None:
            result["tools"] = tools
        return result

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for message in coalesce_messages(messages):
            if source_protocol(message) != PROTOCOL:
                message = project_message(message, foreign_block)
            content = [
                value
                for block in blocks(message.get("content"))
                if (value := native_block(block)) is not None
            ]
            result.append({"role": message["role"], "content": content})
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return decode_functions(
            tools,
            lambda functions: [
                {
                    **select_present(f, "name", "description", "strict"),
                    **(
                        {
                            "cache_control": None
                            if f["cache_control"] is None
                            else select(f["cache_control"], "type", "ttl")
                        }
                        if "cache_control" in f
                        else {}
                    ),
                    "input_schema": f.get("parameters") or {"type": "object"},
                }
                for f in functions
            ],
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event in events(chunks):
            kind = event.get("type")
            usage = self.usage(event.get("usage"))
            index = event.get("index", 0)
            if kind == "start":
                result.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": parameters.get("response_id", "generated"),
                            "type": "message",
                            "role": "assistant",
                            "model": parameters.get("model", ""),
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0, **(usage or {})},
                        },
                    }
                )
            elif kind == "block_start":
                block = event["block"]
                result.append(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""}
                        if block["type"] == "thinking"
                        and parameters.get("source_protocol") == PROTOCOL
                        else {"type": "text", "text": ""},
                    }
                )
            elif kind == "delta":
                field = {"text": "text", "thinking": "thinking", "refusal": "text"}.get(
                    event.get("field")
                )
                if field is None:
                    continue
                if field == "thinking" and parameters.get("source_protocol") != PROTOCOL:
                    field = "text"
                value = {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": f"{field}_delta", field: deepcopy(event["value"])},
                }
                result.append(value)
            elif kind == "block_annotation":
                if event["field"] == "citations" and parameters.get("source_protocol") == PROTOCOL:
                    result.append(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "citations_delta",
                                "citation": select(event["value"], *CITATION_FIELDS),
                            },
                        }
                    )
            elif kind == "block_stop":
                signature = event["block"].get("signature")
                if signature and parameters.get("source_protocol") == PROTOCOL:
                    result.append(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "signature_delta", "signature": signature},
                        }
                    )
                result.append({"type": "content_block_stop", "index": index})
            elif kind == "tool":
                block = native_block(event["block"])
                if parameters.get("source_protocol") != PROTOCOL:
                    block = foreign_block(block)
                result.extend(
                    [
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {**block, "input": {}},
                        },
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json_arguments(block["input"]),
                            },
                        },
                        {"type": "content_block_stop", "index": index},
                    ]
                )
            elif kind == "block":
                block = event["block"]
                if parameters.get("source_protocol") != PROTOCOL:
                    block = foreign_block(block)
                    if block is None:
                        continue
                result.extend(
                    [
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": deepcopy(native_block(block)),
                        },
                        {"type": "content_block_stop", "index": index},
                    ]
                )
            elif kind == "finish":
                reason = REASONS.get(event["reason"], event["reason"])
                stop_sequence = None
                if parameters.get("source_protocol") == PROTOCOL:
                    reason = parameters.get("native_stop_reason", reason)
                    stop_sequence = parameters.get("native_stop_sequence")
                elif event["reason"] not in REASONS:
                    raise ResponseFailedError(
                        "Anthropic cannot represent the source finish reason",
                        details={"reason": event["reason"]},
                    )
                result.extend(
                    [
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": reason, "stop_sequence": stop_sequence},
                            "usage": usage or {"output_tokens": 0},
                        },
                        {"type": "message_stop"},
                    ]
                )
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
        if output_tokens is None:
            return None
        result = {"output_tokens": output_tokens}
        if input_tokens is not None:
            result["input_tokens"] = input_tokens - (cached or 0) - (cache_write or 0)
        if cached is not None:
            result["cache_read_input_tokens"] = cached
        if cache_write is not None:
            result["cache_creation_input_tokens"] = cache_write
        return result
