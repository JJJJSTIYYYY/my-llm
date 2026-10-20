"""gemini: intermediate to native output."""

from copy import deepcopy
from typing import Any

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.gemini.decoding.content import decode_part, foreign_block
from my_llm.codec.shared.conversion import (
    blocks,
    coalesce_messages,
    decode_functions,
    project_message,
    select,
    select_present,
    source_protocol,
)
from my_llm.codec.shared.options import PreparedOptions
from my_llm.codec.shared.stream import counts, events
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "gemini"


class GeminiDecoder(BaseDecoder):
    """Translate recognized intermediate values into protocol dictionaries."""

    __slots__ = ()
    protocol = PROTOCOL

    body_container = "config"
    headers_field = "$.config.http_options.headers"

    def _request_parameters(
        self, request: dict, profile: ResolvedProfile, options: PreparedOptions
    ) -> dict:
        config = request["config"]
        config["automatic_function_calling"] = {"disable": True}
        if options.effort and profile.get("supports_reasoning_effort", False):
            thinking = config.get("thinking_config")
            if isinstance(thinking, dict) and thinking.get("thinking_level") is not None:
                thinking.pop("thinking_budget", None)
        return request

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
        config = result["config"]
        system = [m for m in messages if m.get("role") in ("system", "developer")]
        history = [m for m in messages if m.get("role") not in ("system", "developer")]
        if system:
            config["system_instruction"] = {"parts": [part for m in system for part in m["parts"]]}
        if tools is not None:
            config["tools"] = tools
        result.update(model=profile.model, contents=history, config=config)
        return result

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        names = {}
        for message in coalesce_messages(messages):
            for block in message.get("content", []):
                if block.get("type") == "tool_use":
                    names[block["id"]] = block["name"]
            foreign = source_protocol(message) != PROTOCOL
            if foreign:
                message = project_message(message, foreign_block)
            parts = []
            for block in blocks(message.get("content")):
                if block.get("type") == "tool_result":
                    if not block.get("name"):
                        block = {
                            **select(
                                block,
                                "type",
                                "tool_use_id",
                                "content",
                                "thought_signature",
                                "part_metadata",
                            ),
                            "name": names.get(block["tool_use_id"]),
                        }
                parts.append(decode_part(block, foreign=foreign))
            result.append(
                {
                    "role": "model" if message["role"] == "assistant" else message["role"],
                    "parts": parts,
                }
            )
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return decode_functions(
            tools,
            lambda functions: [
                {
                    "function_declarations": [
                        {
                            **select_present(f, "name", "response_json_schema", "behavior"),
                            "description": f.get("description") or "",
                            "parameters_json_schema": f.get("parameters") or {"type": "object"},
                        }
                        for f in functions
                    ]
                }
            ],
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event in events(chunks):
            kind = event.get("type")
            usage = self.usage(event.get("usage"))
            parts = []
            reason = None
            if kind == "delta" and event.get("field") in ("text", "thinking", "refusal"):
                parts = [
                    {
                        "text": deepcopy(event["value"]),
                        **(
                            {"thought": True}
                            if event["field"] == "thinking"
                            and parameters.get("source_protocol") == PROTOCOL
                            else {}
                        ),
                    }
                ]
            elif kind in ("tool", "block"):
                block = event["block"]
                foreign = parameters.get("source_protocol") != PROTOCOL
                if foreign:
                    block = foreign_block(block)
                    if block is None:
                        continue
                part = decode_part(block, foreign=foreign)
                if "function_call" in part:
                    part["function_call"]["args"] = deepcopy(part["function_call"]["args"])
                elif "function_response" in part:
                    response = part["function_response"]
                    response["response"] = deepcopy(response["response"])
                if "part_metadata" in part:
                    part["part_metadata"] = deepcopy(part["part_metadata"])
                parts = [part]
            elif kind == "block_stop":
                metadata = select(event["block"], "thought_signature", "part_metadata")
                if metadata and parameters.get("source_protocol") == PROTOCOL:
                    if "part_metadata" in metadata:
                        metadata["part_metadata"] = deepcopy(metadata["part_metadata"])
                    parts = [
                        {
                            "text": "",
                            **metadata,
                            **(
                                {"thought": True}
                                if event["block"].get("type") == "thinking"
                                else {}
                            ),
                        }
                    ]
            elif kind == "finish":
                reason = {
                    "length": "MAX_TOKENS",
                    "content_filter": "SAFETY",
                    "stop": "STOP",
                    "tool_calls": "STOP",
                }.get(event["reason"], "OTHER")
            if parts or reason:
                value = {
                    "response_id": parameters.get("response_id", "generated"),
                    "model_version": parameters.get("model", ""),
                    "candidates": [
                        {
                            "index": 0,
                            "content": {"role": "model", "parts": parts},
                            **({"finish_reason": reason} if reason else {}),
                        }
                    ],
                }
                if kind == "finish" and usage is not None:
                    value["usage_metadata"] = usage
                result.append(value)
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
        return {
            key: child
            for key, child in {
                "prompt_token_count": input_tokens,
                "total_token_count": total,
                "candidates_token_count": output_tokens - (reasoning or 0)
                if output_tokens is not None
                else None,
                "thoughts_token_count": reasoning,
                "cached_content_token_count": cached,
            }.items()
            if child is not None
        } or None
