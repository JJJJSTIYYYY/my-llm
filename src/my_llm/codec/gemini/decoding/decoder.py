"""gemini: intermediate to native output."""

from copy import deepcopy
from typing import Any

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.gemini.decoding.content import decode_part, foreign_block
from my_llm.codec.shared.conversion import (
    OPAQUE,
    Translations,
    blocks,
    coalesce_groups,
    decode_functions,
    extras,
    original,
    plain,
    project_message,
    select_present,
    source_protocol,
)
from my_llm.codec.shared.options import PreparedOptions, apply_max_tokens
from my_llm.codec.shared.stream import counts, events
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "gemini"


class GeminiDecoder(BaseDecoder):
    """Translate intermediate values into protocol dictionaries, retaining replay data."""

    __slots__ = ()
    protocol = PROTOCOL

    def request_parameters(
        self, profile: ResolvedProfile, options: PreparedOptions
    ) -> dict[str, Any]:
        tokens, headers, effort = options.tokens, options.headers, options.effort
        config = deepcopy(options.body)
        # Admit native generation settings only, never native cloud tools or AFC overrides.
        config["automatic_function_calling"] = {"disable": True}
        if headers:
            config["http_options"] = {"headers": deepcopy(headers)}
        if effort:
            self._apply_reasoning(config, profile, effort)
        request = {"config": config}
        apply_max_tokens(request, profile, tokens)
        return request

    def assemble_request(
        self,
        messages: list,
        tools: list | None,
        profile: ResolvedProfile,
        *,
        options: PreparedOptions,
        **parameters: Any,
    ) -> dict:
        result = self.request_parameters(profile, options)
        config = result["config"]
        system = [m for m in messages if m.get("role") in ("system", "developer")]
        history = [m for m in messages if m.get("role") not in ("system", "developer")]
        if system:
            config["system_instruction"] = {"parts": [part for m in system for part in m["parts"]]}
        if tools is not None:
            config["tools"] = tools
        result.update(model=profile.model, contents=history, config=config)
        return result

    def _apply_reasoning(
        self, config: dict[str, Any], profile: ResolvedProfile, effort: str
    ) -> None:
        thinking = {"include_thoughts": True, **plain(config.get("thinking_config") or {})}
        if profile.get("supports_reasoning_effort", False):
            thinking["thinking_level"] = effort
            thinking.pop("thinking_budget", None)
        elif not any(key in thinking for key in ("thinking_level", "thinking_budget")):
            thinking["thinking_budget"] = -1
        config["thinking_config"] = thinking

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> Translations:
        result = Translations()
        names = {}
        for sources, message in coalesce_groups(messages):
            result.group(
                sources,
                structure={
                    "parts": [
                        {
                            "inline_data": {},
                            "file_data": {},
                            "function_call": {},
                            "function_response": {},
                        }
                    ]
                },
            )
            if message.get(OPAQUE):
                result.extend(original(message, PROTOCOL))
                continue
            native = original(message, PROTOCOL)
            for block in message.get("content", []):
                if block.get("type") == "tool_use":
                    names[block["id"]] = block["name"]
            if native is not None:
                result.extend(native)
                continue
            foreign = source_protocol(message) != PROTOCOL
            if foreign:
                message = project_message(message, foreign_block)
            else:
                # Without a native snapshot, these parts become the public
                # result. Capture their mutable payloads once at that boundary.
                message = {**message, "content": blocks(message.get("content"))}
            parts = []
            for block in message.get("content", []):
                if block.get("type") == "tool_result":
                    if not block.get("name"):
                        block = {**block, "name": names.get(block["tool_use_id"])}
                parts.append(decode_part(block, foreign=foreign))
            result.append(
                {
                    **extras(message, "role", "content", "tool_calls"),
                    "role": "model" if message["role"] == "assistant" else message["role"],
                    "parts": parts,
                }
            )
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> Translations:
        return decode_functions(
            tools,
            "gemini",
            lambda functions: [
                {
                    "function_declarations": [
                        {
                            **select_present(f, "name"),
                            "description": f.get("description") or "",
                            "parameters_json_schema": f.get("parameters") or {"type": "object"},
                        }
                        for f in functions
                    ]
                }
            ],
            structure={"function_declarations": [{}]},
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event in events(chunks, PROTOCOL):
            kind = event.get("type")
            usage = self.usage(event.get("usage"))
            if kind == "opaque":
                result.append(event["value"])
                continue
            parts = []
            reason = None
            if kind == "delta":
                parts = [
                    {
                        "text": event["value"],
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
                parts = [decode_part(block, foreign=foreign)]
            elif kind == "block_stop":
                signature = event["block"].get("thought_signature")
                if signature and parameters.get("source_protocol") == PROTOCOL:
                    parts = [
                        {
                            "text": "",
                            "thought_signature": signature,
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
        native = original(value, "gemini")
        if native is not None:
            return native
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
