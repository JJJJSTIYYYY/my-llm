"""openai/chat: intermediate to native output."""

from typing import Any

from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from my_llm.basic.exceptions import (
    ConfigurationError,
    MessageValidationError,
    ResponseFailedError,
    UnsupportedCapabilityError,
)
from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.openai.chat.decoding.content import (
    chat_call,
    foreign_block,
    render_content,
    tool_text,
)
from my_llm.codec.shared.conversion import (
    coalesce_messages,
    decode_functions,
    select,
    select_present,
    source_protocol,
)
from my_llm.codec.shared.options import (
    PreparedOptions,
    validate_request_options,
)
from my_llm.codec.shared.schema import parameter_types
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
        result = super().filter_messages(messages, profile, **parameters)
        for index, (original, value) in enumerate(zip(messages, result)):
            if not profile.get("supports_message_name", True):
                if original.get("name") is not None:
                    report_loss(
                        original["name"],
                        "Chat profile cannot represent message names",
                        path=f"$.messages[{index}].name",
                    )
                value.pop("name", None)
            if (
                value.get("role") == "assistant"
                and profile.get("replay_reasoning_content_when_tool_calls", False)
                and "reasoning_content" in original
            ):
                if not isinstance(original["reasoning_content"], str):
                    raise MessageValidationError("reasoning_content must be a string")
                value["reasoning_content"] = original["reasoning_content"]
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
            raise UnsupportedCapabilityError("only one generated candidate is supported")
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
        name_supported = profile is None or profile.get("supports_message_name", True)
        for index, message in enumerate(coalesce_messages(messages)):
            if message.get("name") is not None and not name_supported:
                report_loss(
                    message["name"],
                    "Chat profile cannot represent message names",
                    path=f"{getattr(message, 'origin_path', f'$.messages[{index}]')}.name",
                )
            context_owner = source_protocol(message)
            content = []
            for block in message["content"]:
                owner = block.get("protocol")
                if (context_owner if owner is None else owner) != PROTOCOL:
                    block = foreign_block(block)
                if block is not None:
                    content.append(block)
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            other = [b for b in content if b.get("type") != "tool_result"]
            if other or not tool_results:
                value = {
                    **select(message, "role", "name"),
                    "content": render_content(other, role=message["role"]),
                }
                if not name_supported:
                    value.pop("name", None)
                calls = [chat_call(b) for b in other if b.get("type") == "tool_use"]
                if calls:
                    value["tool_calls"] = calls
                reasoning = "".join(
                    b.get("thinking", "") for b in other if b.get("type") == "thinking"
                )
                if reasoning:
                    if profile is not None and (
                        not profile.get("replay_reasoning_content_when_tool_calls", False)
                        or profile.parameter("reasoning_content", "output") is None
                    ):
                        for block in other:
                            if block.get("type") == "thinking" and block.get("thinking"):
                                report_loss(block, "Chat profile cannot replay reasoning_content")
                    else:
                        value["reasoning_content"] = reasoning
                result.append(value)
            result.extend(
                {
                    "role": "tool",
                    "tool_call_id": b["tool_use_id"],
                    "content": tool_text(b),
                }
                for b in tool_results
            )
        if profile is not None and profile.get("replay_reasoning_content_when_tool_calls", False):
            if thinking_enabled(profile, parameters):
                for value in result:
                    if value.get("role") == "assistant" and value.get("tool_calls"):
                        value.setdefault("reasoning_content", "")
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
            elif kind == "block" and event.get("block", {}).get("type") == "redacted_thinking":
                report_loss(
                    event["block"], "Chat cannot represent opaque reasoning from another protocol"
                )
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
            "created": parameters.get("created", 0),
        }
