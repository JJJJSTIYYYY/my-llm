"""ollama: intermediate to native output."""

from inspect import signature
from typing import Any

from ollama import AsyncClient
from openai import Omit

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.ollama.decoding.content import (
    chat_call,
    foreign_block,
    function_tool,
    thinking_enabled,
)
from my_llm.codec.shared.chunks import events
from my_llm.codec.shared.conversion import (
    coalesce_messages,
    extras,
    original,
    source_protocol,
    text,
)
from my_llm.codec.shared.keys import OPAQUE
from my_llm.codec.shared.options import apply_max_tokens, native_options
from my_llm.codec.shared.rendering import project_message, text_result
from my_llm.codec.shared.tools import decode_functions
from my_llm.codec.shared.usage import counts
from my_llm.exceptions import ConfigurationError, MessageValidationError
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "ollama"
PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}
REQUEST_FIELDS = set(signature(AsyncClient.chat).parameters) - PROTECTED - {"self"}


class OllamaDecoder(BaseDecoder):
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
            content = message.get("content", [])
            for block in content:
                if block.get("type") == "tool_use":
                    names[block["id"]] = block["name"]
            native = original(message, PROTOCOL)
            if native is not None:
                result.extend(native)
                continue
            if source_protocol(message) != PROTOCOL:
                message = project_message(message, foreign_block)
                content = message["content"]
            calls = []
            images = []
            reasoning = ""
            unknown = []
            results = []
            for block in content:
                kind = block.get("type")
                if kind == "tool_use":
                    call = chat_call(block)
                    calls.append({"function": call["function"]})
                elif kind == "tool_result":
                    results.append(
                        {
                            "role": "tool",
                            "tool_name": block.get("name") or names.get(block["tool_use_id"]),
                            "content": text_result(block.get("content")),
                        }
                    )
                elif kind == "thinking":
                    reasoning += block.get("thinking", "")
                elif kind == "image":
                    source = block.get("source", {})
                    if source.get("type") != "base64":
                        raise MessageValidationError("Ollama images require base64 data")
                    images.append(source.get("data"))
                elif kind != "text":
                    unknown.append(block)
            if len(content) != len(results) or not results:
                value = {
                    **extras(message, "content", "parts", "tool_calls"),
                    "content": text(content),
                }
                if calls:
                    value["tool_calls"] = calls
                if images:
                    value["images"] = images
                if reasoning:
                    value["thinking"] = reasoning
                if unknown:
                    value["content"] += text_result(unknown)
                result.append(value)
            result.extend(results)
        if profile:
            for value in result:
                if value.get("role") != "assistant":
                    continue
                if not profile.get("replay_reasoning_content_when_tool_calls", False):
                    value.pop("thinking", None)
                elif value.get("tool_calls") and thinking_enabled(profile, parameters):
                    # Only active tool-call turns need a missing reasoning field padded.
                    value.setdefault("thinking", "")
        return result

    def tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return decode_functions(
            tools,
            "ollama",
            lambda functions: [function_tool(function) for function in functions],
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
            message = {"role": "assistant", "content": ""}
            if kind == "delta":
                message["thinking" if event["field"] == "thinking" else "content"] = event["value"]
            elif kind == "tool":
                block = event["block"]
                if parameters.get("source_protocol") != PROTOCOL:
                    block = foreign_block(block)
                call = chat_call(block)
                message["tool_calls"] = [{"function": call["function"]}]
            elif kind != "finish":
                continue
            value = {
                "model": parameters.get("model", ""),
                "message": message,
                "done": kind == "finish",
            }
            if kind == "finish":
                value["done_reason"] = (
                    "stop" if event["reason"] == "tool_calls" else event["reason"]
                )
                value.update(usage or {})
            result.append(value)
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        native = original(value, "ollama")
        if native is not None:
            return native if set() <= native.keys() else None
        input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
        return {
            key: child
            for key, child in {
                "prompt_eval_count": input_tokens,
                "eval_count": output_tokens,
            }.items()
            if child is not None
        } or None

    def profile_defaults(self) -> dict:
        return {
            "replay_reasoning_content_when_tool_calls": True,
            "supports_reasoning_effort": False,
            "max_tokens_path": "$.options.num_predict",
            "content_path": "$.message.content",
            "reasoning_content_path": "$.message.thinking",
            "reasoning_enable_parameter": {},
            "reasoning_effort_parameter": {},
        }

    def request_parameters(self, profile: ResolvedProfile, **parameters: Any) -> dict[str, Any]:
        tokens, _, request, effort = native_options(
            profile, request_fields=REQUEST_FIELDS, protected=PROTECTED, **parameters
        )
        if effort:
            request["think"] = self._thinking_value(request, profile, effort)
        apply_max_tokens(request, profile, tokens, protected=PROTECTED)
        return request

    def assemble_request(
        self, messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any
    ) -> dict:
        previous = parameters.pop("previous_response_id", None)
        if previous is not None and not isinstance(previous, Omit):
            raise ConfigurationError("previous_response_id requires OpenAI Responses API")
        result = self.request_parameters(profile, **parameters)
        result.update(model=profile.model, stream=True, messages=messages)
        if tools is not None:
            result["tools"] = tools
        return result

    def _thinking_value(
        self, request: dict[str, Any], profile: ResolvedProfile, effort: str
    ) -> str | bool:
        if profile.get("supports_reasoning_effort", False):
            # Ollama's protocol permits model-defined strings. The profile owns
            # the model's supported mapping; there is no universal level enum.
            return effort
        thinking = request.get("think")
        if thinking is False:
            raise ConfigurationError("reasoning conflicts with think=False")
        return True if thinking is None else thinking
