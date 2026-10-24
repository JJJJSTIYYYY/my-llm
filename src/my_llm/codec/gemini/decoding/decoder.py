"""gemini: intermediate to native output."""

from typing import Any

from google.genai import types
from google.genai.models import AsyncModels

from my_llm.basic.data import capture
from my_llm.basic.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.gemini.decoding.content import decode_part
from my_llm.codec.shared.conversion import (
    coalesce_messages,
    decode_functions,
    select,
    select_present,
    source_protocol,
)
from my_llm.codec.shared.options import PreparedOptions, validate_request_options
from my_llm.codec.shared.schema import parameter_types
from my_llm.codec.shared.stream import counts, events
from my_llm.profile.resolution import ResolvedProfile, client_parameters, validate_max_tokens

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

PROTOCOL = "gemini"


class GeminiDecoder(BaseDecoder):
    """Translate recognized intermediate values into protocol dictionaries."""

    __slots__ = ()
    protocol = PROTOCOL

    tool_type = types.Tool
    message_type = types.Content
    message_path = ("contents",)
    tool_path = ("config", "tools")

    request_schema = {
        **parameter_types(AsyncModels.generate_content_stream),
        "contents": list[types.Content],
        "config": types.GenerateContentConfig,
    }

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(
            profile, options, protected=PROTECTED, containers=("config",)
        )

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        if not request["contents"]:
            raise ConfigurationError("a non-system message is required")
        config = request.get("config")
        if config is None:
            return request
        validate_max_tokens(config.get("max_output_tokens"))
        if config.get("candidate_count") not in (None, 1):
            raise UnsupportedCapabilityError("only one generated candidate is supported")
        # Tool declarations were validated separately in native_request. Validate
        # remaining public config fields without rebuilding every tool schema.
        thinking = types.GenerateContentConfig(
            **{key: value for key, value in config.items() if key != "tools"}
        ).thinking_config
        if thinking is not None and thinking.thinking_budget == 0 and thinking.include_thoughts:
            raise ConfigurationError("reasoning conflicts with a zero Gemini thinking budget")
        return request

    body_container = "config"
    headers_field = "$.config.http_options.headers"

    def client_parameters(self, profile: ResolvedProfile) -> dict:
        values = client_parameters(profile, headers_field="headers")
        options = {key: value for key, value in values.items() if key in ("base_url", "headers")}
        if "timeout" in values:
            options["timeout"] = max(1, round(values["timeout"] * 1000))
        if "max_retries" in values:
            options["retry_options"] = types.HttpRetryOptions(attempts=values["max_retries"] + 1)
        return dict(
            **({"api_key": values["api_key"]} if "api_key" in values else {}),
            http_options=types.HttpOptions(**options),
        )

    def _request_parameters(
        self, request: dict, profile: ResolvedProfile, options: PreparedOptions
    ) -> dict:
        config = request["config"]
        config["automatic_function_calling"] = {"disable": True}
        if options.effort and profile.parameter("reasoning_effort", "input") is not None:
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
        for index, message in enumerate(coalesce_messages(messages)):
            if message.get("name") is not None:
                report_loss(
                    message["name"],
                    "Gemini cannot represent message names",
                    path=f"{getattr(message, 'origin_path', f'$.messages[{index}]')}.name",
                )
            for block in message.get("content", []):
                if block.get("type") == "tool_use":
                    names[block["id"]] = block["name"]
            foreign = source_protocol(message) != PROTOCOL
            parts = []
            for block in message["content"]:
                if block.get("type") == "tool_result":
                    if not block.get("name"):
                        block = {
                            **select(
                                block,
                                "type",
                                "tool_use_id",
                                "content",
                                "signature",
                            ),
                            "name": names.get(block["tool_use_id"]),
                        }
                part = decode_part(block, foreign=foreign)
                if part is not None:
                    parts.append(part)
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
                            **select_present(f, "name"),
                            "description": f.get("description") or "",
                            "parameters_json_schema": f["parameters"]
                            if f.get("parameters") is not None
                            else {"type": "object"},
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
            if kind in ("block_start", "block_stop", "tool", "block"):
                owner = event["block"].get("protocol")
                native_owner = (
                    parameters.get("source_protocol") if owner is None else owner
                ) == PROTOCOL
            parts = []
            reason = None
            if (
                kind == "block_start"
                and event.get("block", {}).get("type") == "thinking"
                and not native_owner
            ):
                report_loss(
                    event["block"],
                    "Gemini cannot replay reasoning without its native thought signature",
                )
                continue
            if kind == "delta" and event.get("field") in ("text", "thinking"):
                if event["field"] == "thinking" and parameters.get("source_protocol") != PROTOCOL:
                    continue
                parts = [
                    {
                        "text": event["value"],
                        **({"thought": True} if event["field"] == "thinking" else {}),
                    }
                ]
            elif kind in ("tool", "block"):
                block = event["block"]
                foreign = parameters.get("source_protocol") != PROTOCOL
                part = decode_part(block, foreign=foreign)
                if part is None:
                    continue
                if "function_call" in part:
                    part["function_call"]["args"] = capture(part["function_call"]["args"])
                elif "function_response" in part:
                    response = part["function_response"]
                    response["response"] = capture(response["response"])
                parts = [part]
            elif kind == "block_stop":
                if event["block"].get("type") == "thinking" and not native_owner:
                    report_loss(
                        event["block"],
                        "Gemini cannot replay reasoning without its native thought signature",
                    )
                    continue
                if (
                    event["block"].get("type") == "thinking"
                    and not event["block"].get("signature")
                    and native_owner
                ):
                    report_loss(
                        event["block"], "Gemini reasoning requires a native thought signature"
                    )
                metadata = (
                    {"thought_signature": event["block"]["signature"]}
                    if "signature" in event["block"]
                    else {}
                )
                if metadata and native_owner:
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
