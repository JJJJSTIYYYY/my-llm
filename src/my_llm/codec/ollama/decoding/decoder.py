"""ollama: intermediate to native output."""

from collections.abc import Callable, Iterable
from typing import Any

from ollama import Image, Message, Options, Tool
from pydantic import ValidationError

from my_llm.basic.data import capture, dictionary_views
from my_llm.basic.exceptions import ConfigurationError
from my_llm.basic.semantic_loss import report_loss
from my_llm.cache.compiled_cache import _MISSING, _plan
from my_llm.codec.base.decoder import BaseDecoder
from my_llm.codec.ollama.decoding.content import (
    chat_call,
    foreign_block,
    function_tool,
    tool_text,
)
from my_llm.codec.ollama.schema import REQUEST_SCHEMA, validate_ollama_values
from my_llm.codec.shared.capabilities import content_capability
from my_llm.codec.shared.conversion import (
    decode_functions,
    source_protocol,
)
from my_llm.codec.shared.input import (
    InputPolicy,
    input_content,
    message_feedback,
    pad_reasoning,
)
from my_llm.codec.shared.options import PreparedOptions, validate_request_options
from my_llm.codec.shared.stream import counts, events
from my_llm.profile.configuration import merge_dicts
from my_llm.profile.resolution import ResolvedProfile, client_parameters

PROTECTED = {"model", "stream", "messages", "input", "tools", "previous_response_id"}


def validate_schema(parameters: dict, *, path: str | None = None) -> dict:
    """Do not let the official SDK silently discard JSON Schema constraints."""
    try:
        parsed = Tool(function={"name": "schema_check", "parameters": parameters}).model_dump(
            exclude_none=True
        )["function"]["parameters"]
    except (ValidationError, KeyError):
        report_loss(
            parameters, "installed Ollama SDK cannot represent the tool parameter schema", path=path
        )
        return {"type": "object"}

    def preserves(expected, actual):
        """Whether SDK-normalized schema data preserves each supplied constraint."""
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
            parameters,
            "installed Ollama SDK would discard tool parameter schema constraints",
            path=path,
        )
        return parsed
    return parameters


PROTOCOL = "ollama"


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
        """Validate Ollama values with SDK-compatible image normalization."""
        validate_ollama_values(values, schema, filtered=filtered)

    def validate_parameters(self, profile: ResolvedProfile, **parameters: Any) -> PreparedOptions:
        """Validate shared controls and protect Ollama fields from extra-body replacement."""
        options = super().validate_parameters(profile, **parameters)
        return validate_request_options(profile, options, protected=PROTECTED, containers=())

    def validate_request(self, request: dict, profile: ResolvedProfile, **parameters: Any) -> dict:
        """Validate thinking/options and record retry settings unsupported by the Ollama SDK."""
        retries = profile.get("max_retries")
        if isinstance(retries, int) and not isinstance(retries, bool) and retries > 0:
            report_loss(
                retries,
                "Ollama SDK does not support a positive max_retries",
                path="$.max_retries",
            )
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
        """Filter native tools while reporting schema constraints the installed SDK would discard."""
        decode = _plan(self.tool_type, delegated_paths=(("function", "parameters"),)).decode
        result = []
        for index, original in enumerate(dictionary_views(tools)):
            tool = decode(original, f"$.tools[{index}]")
            if tool is _MISSING:
                continue
            function = tool.get("function", {})
            if "parameters" in function:
                function["parameters"] = capture(
                    validate_schema(
                        function["parameters"], path=f"$.tools[{index}].function.parameters"
                    )
                )
            result.append(tool)
        return result

    message_delegated_paths = (("images",),)

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
        """Normalize Ollama images and filter tool/reasoning capabilities per message."""
        if original.get("role") == "tool" and not policy.allows(original, "tool_result", path=path):
            return None
        value = decode(original, path)
        if value is _MISSING:
            return None
        images = value.get("images")
        if images:
            retained = []
            for offset, image in enumerate(images):
                if policy.allows(image, "image", path=f"{path}.images[{offset}]"):
                    native = getattr(image, "origin_value", image)
                    retained.append(
                        native.model_dump(mode="json") if isinstance(native, Image) else native
                    )
            if retained:
                value["images"] = retained
            else:
                value.pop("images", None)
        elif "images" in value:
            value["images"] = capture(images)
        if not policy.supported["tool_use"]:
            for offset, call in enumerate(original.get("tool_calls") or []):
                policy.allows(call, "tool_use", path=f"{path}.tool_calls[{offset}]")
            value.pop("tool_calls", None)
        if original.get("name") is not None and not policy.allows(
            original["name"], "message_name", path=f"{path}.name"
        ):
            value.pop("name", None)
        if (reasoning := original.get("thinking")) is not None and (
            not policy.allows(
                reasoning,
                "thinking",
                feedback=policy.reasoning_feedback(has_tool_calls=bool(value.get("tool_calls"))),
                path=f"{path}.thinking",
            )
        ):
            value.pop("thinking", None)
        pad_reasoning(
            value,
            "thinking",
            feedback=policy.reasoning_feedback(has_tool_calls=True),
            has_tool_calls=bool(value.get("tool_calls")),
        )
        return value

    def validate_profile(self, profile: ResolvedProfile) -> None:
        """Validate shared parameter roots and require header-based API-key bindings."""
        super().validate_profile(profile)
        api_key = profile.parameter("api_key")
        if api_key is not None and api_key.path == "$.api_key":
            raise ConfigurationError("Ollama api_key must use $.extra_headers.<name>")

    headers_field = None

    def client_parameters(self, profile: ResolvedProfile) -> dict:
        """Map provider settings to Ollama host, timeout and merged headers."""
        values = client_parameters(profile, headers_field="headers")
        result = {key: value for key, value in values.items() if key == "timeout"}
        result["host"] = values.get("base_url")
        result["headers"] = merge_dicts(values.get("headers"), profile.get("extra_headers", {}))
        return result

    def request_client(self, client: dict, options: PreparedOptions) -> dict:
        """Merge per-call headers into independent Ollama client settings."""
        return {**client, "headers": merge_dicts(client.get("headers"), options.headers)}

    def client_key(self, client: dict) -> tuple:
        """Key reusable Ollama clients by their effective header values."""
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
        """Insert model, native messages and optional tools into a streaming Ollama request."""
        result = request
        result.update(model=profile.model, stream=True, messages=messages)
        if tools is not None:
            result["tools"] = tools
        return result

    def _messages(
        self, messages: Iterable[dict], profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Render middleware history as Ollama messages, images and tool results.

        Apply capability and reasoning-feedback policies, retain representable
        content and record foreign media/signature losses. Track call names for
        role=tool messages; native tool calls carry argument objects rather than
        JSON strings. Reasoning padding follows the active model/request policy.
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
            context_owner = source_protocol(message)
            calls = []
            images = []
            reasoning = []
            text_parts = []
            results = []
            has_other = False
            for block in message["content"]:
                if not input_content(block, profile, feedback=feedback):
                    continue
                owner = block.get("protocol")
                if (context_owner if owner is None else owner) != PROTOCOL:
                    block = foreign_block(block)
                if block is None:
                    continue
                kind = block.get("type")
                has_other = has_other or kind != "tool_result"
                if kind == "tool_use":
                    names[block["id"]] = block["name"]
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
                    reasoning.append(block["thinking"] or "\n".join(block.get("summary", ())))
                elif kind == "text":
                    text_parts.append(block.get("text", ""))
                elif kind == "image":
                    source = block.get("source", {})
                    if source.get("type") != "base64":
                        report_loss(block, "Ollama images require base64 data")
                        continue
                    images.append(source.get("data"))
            if has_other or not results:
                value = {
                    "role": message["role"],
                    "content": "".join(text_parts),
                }
                if calls:
                    value["tool_calls"] = calls
                if images:
                    value["images"] = images
                reasoning_text = "".join(reasoning)
                if reasoning_text:
                    value["thinking"] = reasoning_text
                pad_reasoning(
                    value,
                    "thinking",
                    feedback=profile is not None and feedback,
                    has_tool_calls=bool(calls),
                )
                result.append(value)
            result.extend(results)
        return result

    def _tools(
        self, tools: list, profile: ResolvedProfile | None = None, **parameters: Any
    ) -> list[dict]:
        """Render canonical function declarations as Ollama tools and report unsupported fields."""

        def wrap(functions):
            """Validate and render each canonical function as a native Ollama tool."""
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
        """Render public middleware events as native Ollama stream events.

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
        """Render canonical token counters in the Ollama usage format."""
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
