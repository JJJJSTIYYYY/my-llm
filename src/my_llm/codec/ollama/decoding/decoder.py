"""ollama: intermediate to native output."""

from copy import deepcopy
from typing import Any

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.ollama.decoding.content import (
    chat_call,
    foreign_block,
    function_tool,
)
from my_llm.codec.shared.conversion import (
    blocks,
    coalesce_messages,
    decode_functions,
    project_message,
    source_protocol,
    text,
    text_result,
)
from my_llm.codec.shared.options import PreparedOptions, apply_max_tokens
from my_llm.codec.shared.stream import counts, events
from my_llm.exceptions import MessageValidationError
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile

PROTOCOL = "ollama"


def thinking_enabled(profile: ResolvedProfile, parameters: dict) -> bool:
    if parameters.get("reasoning") is True or parameters.get("reasoning_effort", "none") != "none":
        return True
    body = merge_dicts(profile.get("extra_body", {}), parameters.get("extra_body"))
    return body.get("think") not in (None, False)


class OllamaDecoder(BaseDecoder):
    """Translate recognized intermediate values into protocol dictionaries."""

    __slots__ = ()
    protocol = PROTOCOL

    def request_parameters(
        self, profile: ResolvedProfile, options: PreparedOptions
    ) -> dict[str, Any]:
        tokens, effort = options.tokens, options.effort
        request = deepcopy(options.body)
        if effort:
            request["think"] = self._thinking_value(request, profile, effort)
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
        return True if thinking is None else thinking

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        names = {}
        for message in coalesce_messages(messages):
            content = message.get("content", [])
            for block in content:
                if block.get("type") == "tool_use":
                    names[block["id"]] = block["name"]
            if source_protocol(message) != PROTOCOL:
                message = project_message(message, foreign_block)
                content = message["content"]
            else:
                # Public calls own independently mutable arguments.
                content = blocks(content)
            calls = []
            images = []
            reasoning = ""
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
            if len(content) != len(results) or not results:
                value = {
                    "role": message["role"],
                    "content": text(content),
                }
                if calls:
                    value["tool_calls"] = calls
                if images:
                    value["images"] = images
                if reasoning:
                    value["thinking"] = reasoning
                result.append(value)
            result.extend(results)
        if profile is not None and profile.get("replay_reasoning_content_when_tool_calls", False):
            if thinking_enabled(profile, parameters):
                for value in result:
                    if value.get("role") == "assistant" and value.get("tool_calls"):
                        value.setdefault("thinking", "")
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return decode_functions(
            tools,
            lambda functions: [function_tool(function) for function in functions],
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        for event in events(chunks):
            kind = event.get("type")
            usage = self.usage(event.get("usage"))
            message = {"role": "assistant", "content": ""}
            if kind == "delta":
                field = {"text": "content", "thinking": "thinking", "refusal": "content"}.get(
                    event.get("field")
                )
                if field is None:
                    continue
                message[field] = event["value"]
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
        input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
        return {
            key: child
            for key, child in {
                "prompt_eval_count": input_tokens,
                "eval_count": output_tokens,
            }.items()
            if child is not None
        } or None
