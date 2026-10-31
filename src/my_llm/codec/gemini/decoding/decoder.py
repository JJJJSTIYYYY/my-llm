"""gemini: intermediate to native output."""

from typing import Any

from google.genai import types
from google.genai.models import AsyncModels

from my_llm.basic.data import capture, dictionary_views
from my_llm.basic.exceptions import ConfigurationError
from my_llm.basic.semantic_loss import report_loss
from my_llm.cache.compiled_cache import parameter_types
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.gemini.decoding.content import decode_part
from my_llm.codec.shared.capabilities import content_capability, supports_content
from my_llm.codec.shared.conversion import (
    decode_functions,
    select,
    select_present,
    source_protocol,
    split_system_messages,
)
from my_llm.codec.shared.input import (
    feedback_reasoning,
    input_capability,
    input_content,
    message_feedback,
)
from my_llm.codec.shared.options import PreparedOptions, validate_request_options
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


def input_kind(part: dict) -> str:
    """Classify a native Gemini part for input-capability filtering."""
    if part.get("function_call") is not None:
        return "tool_use"
    if part.get("function_response") is not None:
        return "tool_result"
    if part.get("thought"):
        return "thinking"
    media = part.get("inline_data") or part.get("file_data")
    mime = media.get("mime_type") if isinstance(media, dict) else None
    return "image" if isinstance(mime, str) and mime.startswith("image/") else "text"


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

    def filter_messages(self, messages: list, profile: ResolvedProfile, **parameters: Any) -> list:
        """Filter owned Gemini messages to SDK fields and apply input capability policies."""
        if parameters.get("_translated") or (
            feedback_reasoning(profile, has_tool_calls=False)
            and supports_content("image", profile)
            and supports_content("tool_use", profile)
            and supports_content("message_name", profile)
        ):
            return super().filter_messages(messages, profile, **parameters)
        projected = []
        for index, message in enumerate(dictionary_views(messages)):
            path = f"$.contents[{index}]"
            if message.get("name") is not None:
                content_capability(message["name"], "message_name", profile, path=f"{path}.name")
            parts = message.get("parts")
            if isinstance(parts, list):
                feedback = feedback_reasoning(
                    profile,
                    has_tool_calls=any(
                        isinstance(b, dict) and b.get("function_call") is not None for b in parts
                    ),
                )
                retained = []
                for offset, block in enumerate(parts):
                    if not isinstance(block, dict):
                        retained.append(block)
                        continue
                    block_path = f"{path}.parts[{offset}]"
                    kind = input_kind(block)
                    if not input_capability(
                        block, kind, profile, feedback=feedback, path=block_path
                    ):
                        continue
                    response = block.get("function_response")
                    if (
                        kind == "tool_result"
                        and isinstance(response, dict)
                        and isinstance(children := response.get("parts"), list)
                    ):
                        block = {
                            **select(block, *types.Part.model_fields),
                            "function_response": {
                                **select(
                                    block["function_response"], *types.FunctionResponse.model_fields
                                ),
                                "parts": [
                                    child
                                    for child_index, child in enumerate(children)
                                    if not isinstance(child, dict)
                                    or input_capability(
                                        child,
                                        input_kind(child),
                                        profile,
                                        path=f"{block_path}.function_response.parts[{child_index}]",
                                    )
                                ],
                            },
                        }
                    retained.append(block)
                message = {**select(message, "role"), "parts": retained}
            projected.append(message)
        return super().filter_messages(projected, profile, **parameters)

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        """Validate shared controls and protect Gemini fields from extra-body replacement."""
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(
            profile, options, protected=PROTECTED, containers=("config",)
        )

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        """Validate non-system contents and config, recording unsupported candidate settings."""
        if not request["contents"]:
            raise ConfigurationError("a non-system message is required")
        config = request.get("config")
        if config is None:
            return request
        validate_max_tokens(config.get("max_output_tokens"))
        if config.get("candidate_count") not in (None, 1):
            report_loss(
                config["candidate_count"],
                "multiple generated candidates are not supported",
                path="$.config.candidate_count",
            )
            config["candidate_count"] = 1
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
        """Prepare Gemini HTTP options and map provider settings to SDK constructor fields."""
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
        """Build Gemini config and disable automatic SDK tool execution."""
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
        """Extract system instructions into config and assemble Gemini contents and tools."""
        result = request
        config = result["config"]
        system, history = split_system_messages(messages)
        if system:
            config["system_instruction"] = {"parts": [part for m in system for part in m["parts"]]}
        if tools is not None:
            config["tools"] = tools
        result.update(model=profile.model, contents=history, config=config)
        return result

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Render middleware history as Gemini Content messages and parts.

        Map assistant to model, apply content and reasoning-feedback policies, and
        retain compatible native thought signatures. Track tool-call names by ID
        so later function responses can include the required function name. Record
        foreign content that Gemini cannot replay as semantic losses.
        """
        result = []
        names = {}
        for index, message, feedback in message_feedback(messages, profile):
            if message.get("name") is not None:
                content_capability(
                    message["name"],
                    "message_name",
                    profile,
                    native_support=False,
                    path=f"{getattr(message, 'origin_path', f'$.messages[{index}]')}.name",
                )
            foreign = source_protocol(message) != PROTOCOL
            parts = []
            for block in message["content"]:
                if not input_content(block, profile, feedback=feedback):
                    continue
                if block["type"] == "tool_use":
                    names[block["id"]] = block["name"]
                elif block["type"] == "tool_result":
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
                part = decode_part(block, foreign=foreign, profile=profile)
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
        """Render canonical function declarations as Gemini tools and report unsupported fields."""
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
        """Render public middleware events as native Gemini stream events.

        Text/reasoning deltas are already assembled into ordered public events;
        tool events contain complete calls validated at successful source EOF.
        Use source metadata for native identity where available and report
        content that the target protocol cannot express as semantic losses.
        """
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
        """Render canonical token counters in the Gemini usage format."""
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
