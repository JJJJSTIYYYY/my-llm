"""ollama: intermediate to native output."""

from typing import Any

from ollama import Image, Message, Options, Tool
from pydantic import ValidationError

from my_llm.basic.data import capture, read
from my_llm.basic.exceptions import ConfigurationError, UnsupportedCapabilityError
from my_llm.basic.semantic_loss import report_loss
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.ollama.decoding.content import (
    chat_call,
    foreign_block,
    function_tool,
    tool_text,
)
from my_llm.codec.ollama.schema import REQUEST_SCHEMA, validate_ollama_values
from my_llm.codec.shared.conversion import (
    coalesce_messages,
    decode_functions,
    source_protocol,
    text,
)
from my_llm.codec.shared.input import feedback_reasoning, input_content, message_feedback
from my_llm.codec.shared.options import PreparedOptions, validate_request_options
from my_llm.codec.shared.stream import counts, events
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile, client_parameters

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}


def validate_schema(parameters: dict) -> dict:
    """Do not let the official SDK silently discard JSON Schema constraints."""
    try:
        parsed = Tool(function={"name": "schema_check", "parameters": parameters}).model_dump(
            exclude_none=True
        )["function"]["parameters"]
    except (ValidationError, KeyError):
        report_loss(parameters, "installed Ollama SDK cannot represent the tool parameter schema")
        return {"type": "object"}

    def preserves(expected, actual):
        if isinstance(expected, dict):
            return (
                isinstance(actual, dict)
                and not ("type" not in expected and "type" in actual)
                and all(
                    key in actual and preserves(value, actual[key])
                    for key, value in expected.items()
                )
            )
        if isinstance(expected, list):
            return (
                isinstance(actual, list)
                and len(expected) == len(actual)
                and all(preserves(left, right) for left, right in zip(expected, actual))
            )
        return expected == actual

    if not preserves(parameters, parsed):
        report_loss(
            parameters, "installed Ollama SDK would discard tool parameter schema constraints"
        )
        return parsed
    return parameters


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

    tool_type = Tool
    message_type = Message
    message_path = ("messages",)
    tool_path = ("tools",)

    request_schema = REQUEST_SCHEMA

    def validate_values(self, values: list, schema: Any, *, filtered: bool = False) -> None:
        validate_ollama_values(values, schema, filtered=filtered)

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(profile, options, protected=PROTECTED, containers=())

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        thinking = request.get("think")
        if thinking is not None and not isinstance(thinking, (str, bool)):
            raise ConfigurationError("Ollama think must be a boolean or model-defined string")
        if (
            (binding := profile.parameter("reasoning_enable", "input")) is not None
            and binding.path == "$.think"
            and (
                parameters.get("reasoning") is True
                or parameters.get("reasoning_effort", "none") != "none"
            )
            and thinking is False
        ):
            raise ConfigurationError("reasoning conflicts with think=False")
        if request.get("options") is not None:
            try:
                Options(**request["options"])
            except (ValidationError, TypeError, ValueError) as exc:
                raise ConfigurationError("invalid Ollama options") from exc
        return request

    def filter_tools(self, tools: list, profile: ResolvedProfile) -> list:
        for tool in tools:
            if "parameters" in tool.get("function", {}):
                tool["function"]["parameters"] = validate_schema(tool["function"]["parameters"])
        return super().filter_tools(tools, profile)

    def filter_messages(self, messages: list, profile: ResolvedProfile, **parameters: Any) -> list:
        result = super().filter_messages(messages, profile, **parameters)
        for index, (original, value) in enumerate(zip(messages, result)):
            images = read(original, "images")
            if images:
                value["images"] = [
                    image.model_dump(mode="json") if isinstance(image, Image) else image
                    for image in images
                ]
            if not profile.get("supports_message_name", True):
                value.pop("name", None)
            if not feedback_reasoning(profile, has_tool_calls=bool(value.get("tool_calls"))):
                if reasoning := read(original, "thinking"):
                    report_loss(
                        reasoning,
                        "profile excludes reasoning feedback from this message",
                        path=f"$.messages[{index}].thinking",
                    )
                value.pop("thinking", None)
        return result

    def validate_profile(self, profile: ResolvedProfile) -> None:
        super().validate_profile(profile)
        api_key = profile.parameter("api_key")
        if api_key is not None and api_key.path == "$.api_key":
            raise ConfigurationError("Ollama api_key must use $.extra_headers.<name>")
        retries = profile.get("max_retries")
        if isinstance(retries, int) and not isinstance(retries, bool) and retries > 0:
            raise UnsupportedCapabilityError("Ollama SDK does not support a positive max_retries")

    headers_field = None

    def client_parameters(self, profile: ResolvedProfile) -> dict:
        values = client_parameters(profile, headers_field="headers")
        result = {key: value for key, value in values.items() if key == "timeout"}
        result["host"] = values.get("base_url")
        result["headers"] = merge_dicts(values.get("headers"), profile.get("extra_headers", {}))
        return result

    def request_client(self, client: dict, options: PreparedOptions) -> dict:
        return {**client, "headers": merge_dicts(client.get("headers"), options.headers)}

    def client_key(self, client: dict) -> tuple:
        return tuple(sorted(client["headers"].items()))

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
        names = {}
        for index, message, feedback in message_feedback(coalesce_messages(messages), profile):
            if message.get("name") is not None:
                report_loss(
                    message["name"],
                    "Ollama cannot represent message names",
                    path=f"{getattr(message, 'origin_path', f'$.messages[{index}]')}.name",
                )
            content = message["content"]
            for block in content:
                if block.get("type") == "tool_use":
                    names[block["id"]] = block["name"]
            context_owner = source_protocol(message)
            content = []
            for block in message["content"]:
                if not input_content(block, profile, feedback=feedback):
                    continue
                owner = block.get("protocol")
                if (context_owner if owner is None else owner) != PROTOCOL:
                    block = foreign_block(block)
                if block is not None:
                    content.append(block)
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
                            "content": tool_text(block, profile),
                        }
                    )
                elif kind == "thinking":
                    reasoning += block["thinking"] or "\n".join(block.get("summary", ()))
                elif kind == "image":
                    source = block.get("source", {})
                    if source.get("type") != "base64":
                        report_loss(block, "Ollama images require base64 data")
                        continue
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
        if profile is not None and feedback_reasoning(profile, has_tool_calls=True):
            if thinking_enabled(profile, parameters):
                for value in result:
                    if value.get("role") == "assistant" and value.get("tool_calls"):
                        value.setdefault("thinking", "")
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        def wrap(functions):
            result = []
            for function in functions:
                tool = function_tool(function)
                tool["function"]["parameters"] = validate_schema(tool["function"]["parameters"])
                result.append(tool)
            return result

        return decode_functions(tools, wrap)

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
            message = {"role": "assistant", "content": ""}
            if kind == "delta" and event["field"] in ("text", "thinking"):
                field = {"text": "content", "thinking": "thinking"}[event["field"]]
                message[field] = event["value"]
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
                call = chat_call(block)
                call["function"]["arguments"] = capture(call["function"]["arguments"])
                if tool_chunk is not None:
                    tool_chunk["message"]["tool_calls"].append({"function": call["function"]})
                    continue
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
            result.append(value)
            if kind == "tool":
                tool_chunk = value
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
