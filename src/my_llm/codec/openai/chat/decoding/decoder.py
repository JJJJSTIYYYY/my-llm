"""openai/chat: intermediate to native output."""

from typing import Any

from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from my_llm.basic.data import dictionary_views
from my_llm.basic.exceptions import (
    ConfigurationError,
    MessageValidationError,
    ResponseFailedError,
)
from my_llm.basic.semantic_loss import report_loss
from my_llm.cache.compiled_cache import _MISSING, _plan, parameter_types
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.openai.chat.decoding.content import (
    chat_call,
    foreign_block,
    render_content,
    tool_text,
)
from my_llm.codec.shared.capabilities import content_capability, supports_content
from my_llm.codec.shared.conversion import (
    decode_functions,
    select,
    select_present,
    source_protocol,
)
from my_llm.codec.shared.input import (
    feedback_reasoning,
    input_capability,
    input_content,
    input_reasoning,
    message_feedback,
)
from my_llm.codec.shared.options import (
    PreparedOptions,
    validate_request_options,
)
from my_llm.codec.shared.stream import counts, events
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile, validate_max_tokens

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}

PROTOCOL = "openai_chat"


def thinking_enabled(profile: ResolvedProfile, parameters: dict) -> bool:
    if parameters.get("reasoning") is True or parameters.get("reasoning_effort", "none") != "none":
        return True
    body = merge_dicts(profile.get("extra_body", {}), parameters.get("extra_body"))
    thinking = body.get("thinking", {})
    return (
        isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive")
    ) or body.get("enable_thinking") is True


class ChatCompletionsDecoder(BaseDecoder):
    """Translate recognized intermediate values into protocol dictionaries."""

    __slots__ = ()
    protocol = PROTOCOL

    usage_modes = ("none", "stream_options")

    tool_type = ChatCompletionToolParam
    message_type = ChatCompletionMessageParam
    message_path = ("messages",)
    tool_path = ("tools",)

    request_schema = parameter_types(AsyncCompletions.create)

    def filter_messages(self, messages: list, profile: ResolvedProfile, **parameters: Any) -> list:
        result = []
        decode = _plan(self.message_type).decode
        envelope = _plan(
            self.message_type,
            delegated_paths=(("content",), ("tool_calls",), ("function_call",)),
        ).decode
        for index, original in enumerate(dictionary_views(messages)):
            path = f"$.messages[{index}]"
            if original.get("role") in ("tool", "function") and not input_capability(
                original, "tool_result", profile, path=path
            ):
                continue
            projected = original
            if (
                not parameters.get("_translated")
                and isinstance(content := original.get("content"), list)
                and not supports_content("image", profile)
            ):
                retained = [
                    block
                    for offset, block in enumerate(content)
                    if not isinstance(block, dict)
                    or block.get("type") != "image_url"
                    or input_capability(block, "image", profile, path=f"{path}.content[{offset}]")
                ]
                if len(retained) != len(content):
                    projected = envelope(original, path)
                    if projected is _MISSING:
                        continue
                    projected["content"] = retained
            if not parameters.get("_translated") and not supports_content("tool_use", profile):
                for offset, call in enumerate(original.get("tool_calls") or []):
                    input_capability(call, "tool_use", profile, path=f"{path}.tool_calls[{offset}]")
                if original.get("function_call") is not None:
                    input_capability(
                        original["function_call"], "tool_use", profile, path=f"{path}.function_call"
                    )
                if "tool_calls" in original or "function_call" in original:
                    if projected is original:
                        projected = envelope(original, path)
                        if projected is _MISSING:
                            continue
                    projected.pop("tool_calls", None)
                    projected.pop("function_call", None)
            value = decode(projected, path)
            if value is _MISSING:
                continue
            if original.get("name") is not None and not content_capability(
                original["name"], "message_name", profile, path=f"{path}.name"
            ):
                value.pop("name", None)
            if value.get("role") == "assistant" and "reasoning_content" in original:
                reasoning = original["reasoning_content"]
                if not isinstance(reasoning, str):
                    raise MessageValidationError("reasoning_content must be a string")
                if input_reasoning(
                    reasoning,
                    profile,
                    feedback=feedback_reasoning(
                        profile, has_tool_calls=bool(value.get("tool_calls"))
                    ),
                    path=f"$.messages[{index}].reasoning_content",
                ):
                    value["reasoning_content"] = reasoning
                else:
                    value.pop("reasoning_content", None)
            result.append(value)
        return result

    def filter_tools(self, tools: list, profile: ResolvedProfile) -> list:
        result = super().filter_tools(tools, profile)
        for tool in result:
            function = tool.get("function", {})
            for field in ("description", "parameters"):
                if field in function and function[field] is None:
                    function.pop(field)
        return result

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(
            profile, options, protected=PROTECTED, containers=("extra_body",)
        )

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        for field in ("max_tokens", "max_completion_tokens"):
            if field in request:
                validate_max_tokens(request[field])
        effort = request.get("reasoning_effort")
        if effort is not None and not isinstance(effort, str):
            raise ConfigurationError("reasoning_effort must be a string")
        body = request.get("extra_body")
        if body is not None and body.get("n") not in (None, 1):
            report_loss(
                body["n"],
                "multiple generated candidates are not supported",
                path="$.extra_body.n",
            )
            body["n"] = 1
        if request.get("n") not in (None, 1):
            report_loss(request["n"], "multiple generated candidates are not supported", path="$.n")
            request["n"] = 1
        return request

    body_container = "extra_body"
    body_overlay = "extra_body"
    omit_empty_body = True

    def _request_parameters(
        self, request: dict, profile: ResolvedProfile, options: PreparedOptions
    ) -> dict:
        if profile.get("usage_mode", "none") == "stream_options":
            request["stream_options"] = {"include_usage": True}
            body = request.get("extra_body", {})
            if "stream_options" in body:
                # The SDK overlays extra_body after serializing root parameters.
                stream_options = body["stream_options"]
                body["stream_options"] = {
                    **(stream_options if isinstance(stream_options, dict) else {}),
                    "include_usage": True,
                }
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
        result.update(model=profile.model, stream=True, messages=messages)
        if tools is not None:
            result["tools"] = tools

        return result

    def _messages(
        self, messages: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        pad_reasoning = (
            profile is not None
            and feedback_reasoning(profile, has_tool_calls=True)
            and thinking_enabled(profile, parameters)
        )
        for index, message, feedback in message_feedback(messages, profile):
            name_supported = message.get("name") is None or content_capability(
                message["name"],
                "message_name",
                profile,
                path=f"{getattr(message, 'origin_path', f'$.messages[{index}]')}.name",
            )
            context_owner = source_protocol(message)
            content = []
            tool_results = []
            calls = []
            reasoning = []
            for block in message["content"]:
                if not input_content(block, profile, feedback=feedback):
                    continue
                owner = block.get("protocol")
                if (context_owner if owner is None else owner) != PROTOCOL:
                    block = foreign_block(block)
                if block is None:
                    continue
                kind = block["type"]
                if kind == "tool_result":
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": tool_text(block, profile),
                        }
                    )
                    continue
                content.append(block)
                if kind == "tool_use":
                    calls.append(chat_call(block))
                elif kind == "thinking":
                    reasoning.append(block["thinking"] or "\n".join(block.get("summary", ())))
            if content or not tool_results:
                value = {
                    **select(message, "role", "name"),
                    "content": render_content(content, role=message["role"]),
                }
                if not name_supported:
                    value.pop("name", None)
                if calls:
                    value["tool_calls"] = calls
                reasoning_text = "".join(reasoning)
                if reasoning_text or (pad_reasoning and calls and message["role"] == "assistant"):
                    value["reasoning_content"] = reasoning_text
                result.append(value)
            result.extend(tool_results)
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        return decode_functions(
            tools,
            lambda functions: [
                {
                    "type": "function",
                    "function": select_present(f, "name", "description", "parameters", "strict"),
                }
                for f in functions
            ],
        )

    def _chunks(
        self, chunks: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        result = []
        tool_chunk = None
        for event in events(chunks):
            kind = event.get("type")
            if kind in ("block_start", "block_stop", "tool", "block"):
                owner = event["block"].get("protocol")
                native_owner = (
                    parameters.get("source_protocol") if owner is None else owner
                ) == PROTOCOL
            delta = {}
            finish = None
            if kind == "start":
                delta = {"role": "assistant"}
            elif kind == "delta" and event["field"] in ("text", "thinking"):
                field = {
                    "text": "content",
                    "thinking": "reasoning_content",
                }[event["field"]]
                if (
                    event["field"] == "thinking"
                    and profile is not None
                    and profile.parameter("reasoning_content", "output") is None
                ):
                    report_loss(event, "Chat profile cannot express reasoning_content")
                    continue
                delta = {field: event["value"]}
            elif kind == "block_stop" and not native_owner:
                foreign_block(event["block"])
                continue
            elif kind == "block" and event["block"]["type"] == "thinking":
                foreign_block(event["block"])
                continue
            elif kind == "tool":
                block = event["block"]
                if not native_owner:
                    block = foreign_block(block)
                call = {**chat_call(block), "index": event.get("tool_index", 0)}
                if tool_chunk is not None:
                    tool_chunk["choices"][0]["delta"]["tool_calls"].append(call)
                    continue
                delta = {"tool_calls": [call]}
            elif kind == "finish":
                finish = event["reason"]
                if finish not in (
                    "stop",
                    "length",
                    "tool_calls",
                    "content_filter",
                    "function_call",
                ):
                    raise ResponseFailedError(
                        "Chat cannot represent the source finish reason", details={"reason": finish}
                    )
            if not delta and kind != "finish":
                continue
            value = {
                **self._envelope(**parameters),
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            result.append(value)
            if kind == "tool":
                tool_chunk = value
        return result

    def usage(self, value: dict | None) -> dict | None:
        if value is None:
            return None
        input_tokens, output_tokens, total, cached, cache_write, reasoning = counts(value)
        if input_tokens is None or output_tokens is None:
            return None
        result = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total,
        }
        if cached is not None:
            result["prompt_tokens_details"] = {"cached_tokens": cached}
        if reasoning is not None:
            result["completion_tokens_details"] = {"reasoning_tokens": reasoning}
        return result

    def _envelope(self, **parameters: Any) -> dict:
        return {
            "id": parameters.get("response_id", "generated"),
            "model": parameters.get("model", ""),
            "created": parameters.get("created_at", 0),
        }
