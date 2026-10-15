"""Shared validation for native protocols; wire formats stay in each codec."""

import json
from copy import deepcopy
from typing import Any
from uuid import uuid4

from openai import Omit, omit

from my_llm.codec.utils.common import _expand
from my_llm.codec.utils.input import normalize_messages
from my_llm.codec.utils.tools import _function_definition
from my_llm.codec.utils.validation import validate_messages, validate_tool_call
from my_llm.profile.utils.common import EFFORTS, merge_dicts
from my_llm.profile.utils.resolution import ResolvedProfile, validate_max_tokens
from my_llm.types.exceptions import (
    ConfigurationError,
    IncompleteStreamError,
    MessageValidationError,
    UnsupportedCapabilityError,
)


def native_messages(messages: Any) -> list[dict[str, Any]]:
    entries = normalize_messages(messages)
    if any(
        item.get("type") == "item_reference"
        for entry in entries
        for item in entry.response_items or []
    ):
        raise MessageValidationError("native protocols cannot resolve Responses item_reference")
    history = [entry.message for entry in entries]
    for message in history:
        output = message.get("response_output")
        if output is not None and (
            message.get("role") != "assistant"
            or not isinstance(output, list)
            or any(not isinstance(item, dict) for item in output)
        ):
            raise MessageValidationError("response_output must be assistant output items")
    # response_output is opaque protocol metadata, not necessarily Responses items.
    validate_messages([{k: v for k, v in m.items() if k != "response_output"} for m in history])
    if any(m.get("audio") or m.get("function_call") or m.get("refusal") for m in history):
        raise MessageValidationError(
            "native protocols cannot replay Chat audio/refusal/legacy calls"
        )
    return history


def text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.get("type") not in ("text", "input_text", "output_text") or not isinstance(
            block.get("text"), str
        ):
            raise MessageValidationError("this protocol message requires text content")
        parts.append(block["text"])
    return "".join(parts)


def content_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


def media_source(block: dict[str, Any]) -> tuple[str, str | None, str | None]:
    """Read a unified image/file as URI or base64; never fetch remote content."""
    kind = block.get("type")
    if kind == "image_url":
        image = block.get("image_url")
        source = image.get("url") if isinstance(image, dict) else None
    elif kind == "input_image":
        source = block.get("image_url")
    else:
        file = block.get("file", {}) if kind == "file" else block
        source = (file.get("file_data") or file.get("file_url")) if isinstance(file, dict) else None
    if not isinstance(source, str) or not source:
        raise MessageValidationError(
            "native media needs a URI or base64 data URI; file_id cannot be translated"
        )
    if source.startswith("data:"):
        header, separator, data = source.partition(",")
        if not separator or not header.endswith(";base64") or not data or not header[5:-7]:
            raise MessageValidationError("media data URI must contain base64 data and a MIME type")
        return data, header[5:-7], "base64"
    if not source.startswith(("https://", "http://", "gs://")):
        raise MessageValidationError("native media requires a supported URI or base64 data URI")
    return source, None, None


def function_definitions(tools: Any, profile: ResolvedProfile) -> list[dict[str, Any]] | Omit:
    if tools is None or isinstance(tools, Omit):
        return omit
    if not isinstance(tools, list):
        raise MessageValidationError("tools must be a list of local function tools")
    if tools and not profile.get("supports_function_calling", False):
        raise UnsupportedCapabilityError("model does not support function calling")
    return deepcopy(
        [
            {
                key: value
                for key, value in _function_definition(tool)[1].items()
                if key in ("name", "description", "parameters")
            }
            for tool in tools
        ]
    )


def native_parameters(
    profile: ResolvedProfile,
    *,
    request_fields: tuple[str, ...],
    extra_headers: Any = None,
    extra_body: Any = None,
    max_tokens: Any = None,
    reasoning: bool | None = None,
    reasoning_effort: str = "none",
    previous_response_id: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], str | None]:
    if previous_response_id is not None and not isinstance(previous_response_id, Omit):
        raise ConfigurationError("previous_response_id requires OpenAI Responses API")
    if reasoning_effort not in EFFORTS:
        raise ConfigurationError(f"unsupported reasoning_effort: {reasoning_effort!r}")
    if reasoning is not None and not isinstance(reasoning, bool):
        raise ConfigurationError("reasoning must be a bool")
    if reasoning is False and reasoning_effort != "none":
        raise ConfigurationError("reasoning=False conflicts with a non-none effort")
    effort = "medium" if reasoning is True and reasoning_effort == "none" else reasoning_effort
    if effort != "none":
        if not profile.get("supports_reasoning", False):
            raise UnsupportedCapabilityError("model does not support reasoning")
        if profile.get("supports_reasoning_effort", False):
            effort = profile.get("reasoning_effort_mapping", {}).get(effort)
            if not isinstance(effort, str) or not effort:
                raise UnsupportedCapabilityError(
                    f"model does not support effort {reasoning_effort!r}"
                )
    tokens = profile.get("max_tokens") if max_tokens is None else max_tokens
    if isinstance(tokens, Omit):
        tokens = None
    validate_max_tokens(tokens)
    headers = merge_dicts(profile.values["extra_headers"], extra_headers)
    body = merge_dicts(profile.values["extra_body"], extra_body)
    protected = {
        "model",
        "stream",
        "messages",
        "contents",
        "input",
        "tools",
        "system",
        "system_instruction",
        "previous_response_id",
        "config",
    }
    if set(body) & protected:
        raise ConfigurationError("extra_body cannot override protocol parameters")
    if effort != "none":
        bindings = [profile.get("reasoning_enable_parameter", {})]
        if profile.get("supports_reasoning_effort", False):
            bindings.append(profile.get("reasoning_effort_parameter", {}))
        for binding in bindings:
            expanded = _expand(binding, effort)
            binding_body = expanded.pop("extra_body", None)
            if set(expanded) & protected or (
                isinstance(binding_body, dict) and set(binding_body) & protected
            ):
                raise ConfigurationError("reasoning bindings cannot override protocol parameters")
            body = merge_dicts(body, binding_body, expanded)
    # Isolate mutable configuration once, after merging; codecs borrow it read-only.
    return (
        tokens,
        deepcopy(headers),
        deepcopy({key: body[key] for key in request_fields if key in body}),
        None if effort == "none" else effort,
    )


def complete_call(name: str, arguments: Any, call_id: str | None = None) -> dict[str, Any]:
    return validate_tool_call(
        {
            "id": call_id or f"call_{uuid4().hex}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        },
        streamed=True,
    )


def final_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for call in calls:
        validate_tool_call(call, streamed=True)
    if len({c["id"] for c in calls}) != len(calls):
        raise IncompleteStreamError("duplicate tool call IDs")
    return calls


def validate_replay_calls(message: dict[str, Any], native_calls: list[dict[str, Any]]) -> None:
    """Reject edits that would associate a tool result with different native history."""

    def signatures(calls: list[dict[str, Any]]) -> list[tuple[Any, Any, Any]]:
        return [
            (c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls
        ]

    if signatures(message.get("tool_calls", [])) != signatures(native_calls):
        raise MessageValidationError("unified tool_calls must match native response_output calls")
