"""ollama: intermediate to native output."""

from copy import deepcopy
from typing import Any

from ollama import Image, Message, Options, Tool
from pydantic import ValidationError

from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.ollama.decoding.content import (
    chat_call,
    foreign_block,
    function_tool,
)
from my_llm.codec.ollama.utils.schema import REQUEST_SCHEMA, validate_ollama_values
from my_llm.codec.shared.conversion import (
    blocks,
    coalesce_messages,
    decode_functions,
    project_message,
    read,
    source_protocol,
    text,
    text_result,
)
from my_llm.codec.shared.options import PreparedOptions, validate_request_options
from my_llm.codec.shared.stream import counts, events
from my_llm.exceptions import ConfigurationError, MessageValidationError, UnsupportedCapabilityError
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile, client_parameters

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}


def validate_schema(parameters: dict) -> dict:
    """Do not let the official SDK silently discard JSON Schema constraints."""
    try:
        parsed = Tool(function={"name": "schema_check", "parameters": parameters}).model_dump(
            exclude_none=True
        )["function"]["parameters"]
    except (ValidationError, KeyError) as exc:
        raise UnsupportedCapabilityError(
            "installed Ollama SDK cannot represent the tool parameter schema"
        ) from exc

    def preserves(expected, actual):
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                key in actual and preserves(value, actual[key]) for key, value in expected.items()
            )
        if isinstance(expected, list):
            return (
                isinstance(actual, list)
                and len(expected) == len(actual)
                and all(preserves(left, right) for left, right in zip(expected, actual))
            )
        return expected == actual

    if not preserves(parameters, parsed):
        raise UnsupportedCapabilityError(
            "installed Ollama SDK would discard tool parameter schema constraints"
        )
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
                validate_schema(tool["function"]["parameters"])
        return super().filter_tools(tools, profile)

    def filter_messages(self, messages: list, profile: ResolvedProfile, **parameters: Any) -> list:
        result = super().filter_messages(messages, profile, **parameters)
        for original, value in zip(messages, result):
            images = read(original, "images")
            if images:
                value["images"] = [
                    image.model_dump(mode="json") if isinstance(image, Image) else image
                    for image in images
                ]
            if not profile.get("supports_message_name", True):
                value.pop("name", None)
            if not profile.get("replay_reasoning_content_when_tool_calls", False):
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
                message[field] = deepcopy(event["value"])
            elif kind == "tool":
                block = event["block"]
                if parameters.get("source_protocol") != PROTOCOL:
                    block = foreign_block(block)
                call = chat_call(block)
                call["function"]["arguments"] = deepcopy(call["function"]["arguments"])
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
