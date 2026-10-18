"""gemini: intermediate to native output."""

from typing import Any

from google.genai import types
from openai import Omit

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.gemini.decoding.content import decode_part, foreign_block
from my_llm.codec.shared.chunks import events
from my_llm.codec.shared.conversion import (
    coalesce_messages,
    extras,
    original,
    source_protocol,
)
from my_llm.codec.shared.keys import OPAQUE
from my_llm.codec.shared.options import apply_max_tokens, native_options
from my_llm.codec.shared.rendering import project_message, select_present
from my_llm.codec.shared.tools import decode_functions
from my_llm.codec.shared.usage import counts
from my_llm.codec.shared.values import plain
from my_llm.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.profile.resolution import ResolvedProfile

PROTECTED = {
    "model",
    "stream",
    "messages",
    "contents",
    "input",
    "tools",
    "system_instruction",
    "previous_response_id",
    "config",
    "automatic_function_calling",
    "http_options",
}
REQUEST_FIELDS = set(types.GenerateContentConfig.model_fields) - PROTECTED
PROTOCOL = "gemini"


class GeminiDecoder(BaseDecoder):
    """Render intermediate values and assemble native request dictionaries."""

    __slots__ = ()
    usage_modes = ("none", "native")

    def messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        names = {}
        for message in coalesce_messages(messages):
            if message.get(OPAQUE):
                result.extend(original(message, PROTOCOL))
                continue
            native = original(message, PROTOCOL)
            foreign = source_protocol(message) != PROTOCOL
            if foreign:
                message = project_message(message, foreign_block)
            parts = []
            for block in message.get("content", []):
                if block.get("type") == "tool_use":
                    names[block["id"]] = block["name"]
                if block.get("type") == "tool_result":
                    if not block.get("name"):
                        block["name"] = names.get(block["tool_use_id"])
                parts.append(decode_part(block, foreign=foreign))
            if native is not None:
                result.extend(native)
            else:
                result.append(
                    {
                        **extras(message, "role", "content", "tool_calls"),
                        "role": "model" if message["role"] == "assistant" else message["role"],
                        "parts": parts,
                    }
                )
        return result

    def tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
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
        )

    def chunks(
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
            return native if set() <= native.keys() else None
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

    def profile_defaults(self) -> dict:
        return {
            "replay_reasoning_content_when_tool_calls": True,
            "supports_reasoning_effort": False,
            "max_tokens_path": "$.config.max_output_tokens",
            "content_path": "$.candidates[0].content.parts[0].text",
            "reasoning_content_path": "$.candidates[0].content.parts[0].text",
            "reasoning_enable_parameter": {},
            "reasoning_effort_parameter": {},
        }

    def request_parameters(self, profile: ResolvedProfile, **parameters: Any) -> dict[str, Any]:
        tokens, headers, config, effort = native_options(
            profile, request_fields=REQUEST_FIELDS, protected=PROTECTED, **parameters
        )
        if config.get("candidate_count") not in (None, 1):
            raise UnsupportedCapabilityError("only one generated candidate is supported")
        # Admit native generation settings only, never native cloud tools or AFC overrides.
        config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
        if headers:
            config["http_options"] = types.HttpOptions(headers=headers)
        if effort:
            self._apply_reasoning(config, profile, effort)
        request = {"config": config}
        apply_max_tokens(request, profile, tokens, protected=PROTECTED)
        return request

    def assemble_request(
        self, messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any
    ) -> dict:
        previous = parameters.pop("previous_response_id", None)
        if previous is not None and not isinstance(previous, Omit):
            raise ConfigurationError("previous_response_id requires OpenAI Responses API")
        result = self.request_parameters(profile, **parameters)
        config = result["config"]
        system = [m for m in messages if m.get("role") in ("system", "developer")]
        history = [m for m in messages if m.get("role") not in ("system", "developer")]
        if not history:
            raise ConfigurationError("a non-system message is required")
        if system:
            config["system_instruction"] = {"parts": [part for m in system for part in m["parts"]]}
        if tools is not None:
            config["tools"] = tools
        result.update(
            model=profile.model, contents=history, config=types.GenerateContentConfig(**config)
        )
        return result

    def _apply_reasoning(
        self, config: dict[str, Any], profile: ResolvedProfile, effort: str
    ) -> None:
        thinking = {"include_thoughts": True, **plain(config.get("thinking_config") or {})}
        if profile.get("supports_reasoning_effort", False):
            if effort not in ("minimal", "low", "medium", "high"):
                raise UnsupportedCapabilityError("map reasoning effort to a Gemini-supported level")
            thinking["thinking_level"] = effort
            thinking.pop("thinking_budget", None)
        elif not any(key in thinking for key in ("thinking_level", "thinking_budget")):
            thinking["thinking_budget"] = -1
        config["thinking_config"] = thinking
        if thinking.get("thinking_budget") == 0:
            raise ConfigurationError("reasoning conflicts with a zero Gemini thinking budget")
