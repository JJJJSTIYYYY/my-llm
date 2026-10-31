"""openai/chat: intermediate to native output."""

from collections.abc import Callable, Iterable
from typing import Any

from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

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
from my_llm.codec.shared.capabilities import content_capability
from my_llm.codec.shared.conversion import (
    decode_functions,
    select,
    select_present,
    source_protocol,
)
from my_llm.codec.shared.input import (
    InputPolicy,
    input_content,
    message_feedback,
    pad_reasoning,
)
from my_llm.codec.shared.options import (
    PreparedOptions,
    validate_request_options,
)
from my_llm.codec.shared.stream import counts, events
from my_llm.profile.resolution import ResolvedProfile, validate_max_tokens

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}

PROTOCOL = "openai_chat"


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

    def _filter_message(
        self,
        original: dict,
        policy: InputPolicy,
        *,
        path: str,
        decode: Callable,
        feedback: bool,
        **parameters: Any,
    ) -> dict | None:
        """Apply Chat capabilities and preserve its supported reasoning extension."""
        if original.get("role") in ("tool", "function") and not policy.allows(
            original, "tool_result", path=path
        ):
            return None
        projected = original
        if isinstance(content := original.get("content"), list) and not policy.supported["image"]:
            retained = [
                block
                for offset, block in enumerate(content)
                if not isinstance(block, dict)
                or block.get("type") != "image_url"
                or policy.allows(block, "image", path=f"{path}.content[{offset}]")
            ]
            if len(retained) != len(content):
                projected = _plan(
                    self.message_type,
                    delegated_paths=(("content",), ("tool_calls",), ("function_call",)),
                ).decode(original, path)
                if projected is _MISSING:
                    return None
                projected["content"] = retained
        if not policy.supported["tool_use"]:
            for offset, call in enumerate(original.get("tool_calls") or []):
                policy.allows(call, "tool_use", path=f"{path}.tool_calls[{offset}]")
            if original.get("function_call") is not None:
                policy.allows(original["function_call"], "tool_use", path=f"{path}.function_call")
            if "tool_calls" in original or "function_call" in original:
                if projected is original:
                    projected = _plan(
                        self.message_type,
                        delegated_paths=(("content",), ("tool_calls",), ("function_call",)),
                    ).decode(original, path)
                    if projected is _MISSING:
                        return None
                projected.pop("tool_calls", None)
                projected.pop("function_call", None)
        value = decode(projected, path)
        if value is _MISSING:
            return None
        if original.get("name") is not None and not policy.allows(
            original["name"], "message_name", path=f"{path}.name"
        ):
            value.pop("name", None)
        if (
            value.get("role") == "assistant"
            and (reasoning := original.get("reasoning_content")) is not None
        ):
            if not isinstance(reasoning, str):
                raise MessageValidationError("reasoning_content must be a string")
            if policy.allows(
                reasoning,
                "thinking",
                feedback=policy.reasoning_feedback(
                    has_tool_calls=bool(value.get("tool_calls") or value.get("function_call"))
                ),
                path=f"{path}.reasoning_content",
            ):
                value["reasoning_content"] = reasoning
            else:
                value.pop("reasoning_content", None)
        pad_reasoning(
            value,
            "reasoning_content",
            feedback=policy.reasoning_feedback(has_tool_calls=True),
            has_tool_calls=bool(value.get("tool_calls") or value.get("function_call")),
        )
        return value

    def filter_tools(self, tools: list, profile: ResolvedProfile) -> list:
        """Filter native function tools and remove absent optional function fields."""
        result = super().filter_tools(tools, profile)
        for tool in result:
            function = tool.get("function", {})
            for field in ("description", "parameters"):
                if field in function and function[field] is None:
                    function.pop(field)
        return result

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        """Validate shared controls and protect Chat Completions fields from extra-body replacement."""
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(
            profile, options, protected=PROTECTED, containers=("extra_body",)
        )

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        """Validate token/reasoning controls and record unsupported multiple-candidate requests."""
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
        """Add requested Chat streaming-usage options to prepared parameters."""
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
        """Insert model, native messages and optional tools into a streaming Chat request."""
        result = request
        result.update(model=profile.model, stream=True, messages=messages)
        if tools is not None:
            result["tools"] = tools

        return result

    def _messages(
        self, messages: Iterable[dict], profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Render middleware history as Chat messages with separate tool results.

        Apply capability and reasoning-feedback policies, collect assistant calls
        under tool_calls and emit tool_result blocks as role=tool messages. Convert
        foreign reasoning/media through Chat content helpers and record supported
        values that Chat cannot preserve as semantic losses.
        """
        result = []
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
                if reasoning_text:
                    value["reasoning_content"] = reasoning_text
                pad_reasoning(
                    value,
                    "reasoning_content",
                    feedback=profile is not None and feedback,
                    has_tool_calls=bool(calls),
                )
                result.append(value)
            result.extend(tool_results)
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Render canonical function declarations as Chat Completions tools and report unsupported fields."""
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
        """Render public middleware events as native Chat Completions stream events.

        Text/reasoning deltas are already assembled into ordered public events;
        tool events contain complete calls validated at successful source EOF.
        Use source metadata for native identity where available and report
        content that the target protocol cannot express as semantic losses.
        """
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
        """Render canonical token counters in the Chat Completions usage format."""
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
        """Build one Chat completion chunk with response identity and optional usage."""
        return {
            "id": parameters.get("response_id", "generated"),
            "model": parameters.get("model", ""),
            "created": parameters.get("created_at", 0),
        }
