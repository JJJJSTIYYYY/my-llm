"""SDK type validation that also rejects fields silently ignored by SDKs."""

from collections.abc import Iterable
from typing import Annotated, get_args, get_origin

from anthropic.types import MessageParam, RawMessageStreamEvent
from anthropic.types import ToolParam as AnthropicToolParam
from google.genai import types
from ollama import ChatResponse, Image, Message, Tool
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.responses import Response, ResponseInputItemParam, ResponseStreamEvent, ToolParam
from pydantic import BaseModel, TypeAdapter

from tests.helpers.schema import output_event_for_schema, response_fields, validate_event, validator

MESSAGE_TYPES = {
    "openai_chat": ChatCompletionMessageParam,
    "openai_responses": ResponseInputItemParam,
    "anthropic": MessageParam,
    "gemini": types.Content,
    "ollama": Message,
}


TOOL_TYPES = {
    "openai_chat": ChatCompletionToolParam,
    "openai_responses": ToolParam,
    "anthropic": AnthropicToolParam,
    "gemini": types.Tool,
    "ollama": Tool,
}


def variants(value):
    if get_origin(value) is Annotated:
        yield from variants(get_args(value)[0])
    elif isinstance(value, type) and issubclass(value, BaseModel):
        yield value
    else:
        for child in get_args(value):
            yield from variants(child)


def event_models(value):
    return {
        kind: model
        for model in variants(value)
        for kind in get_args(model.model_fields["type"].annotation)
    }


EVENT_TYPES = {
    "openai_chat": ChatCompletionChunk,
    "openai_responses": event_models(ResponseStreamEvent),
    "anthropic": event_models(RawMessageStreamEvent),
    "gemini": types.GenerateContentResponse,
    "ollama": ChatResponse,
}


def assert_no_ignored_fields(native, validated):
    if isinstance(validated, BaseModel) and isinstance(native, dict):
        fields = type(validated).model_fields
        aliases = {field.alias or key: key for key, field in fields.items()}
        # The pinned OpenAPI has mandatory Response fields absent from the
        # installed SDK. Validate those with the full official schema instead
        # of declaring SDK omissions to be proof that the fields are illegal.
        extra_fields = set()
        if isinstance(validated, Response):
            validator("openai_responses", "response").validate(native)
            extra_fields = response_fields()
        assert set(native) <= (fields.keys() | aliases.keys() | extra_fields)
        for key, value in native.items():
            assert_no_ignored_fields(value, getattr(validated, aliases.get(key, key)))
    elif isinstance(native, dict) and isinstance(validated, dict):
        assert native.keys() <= validated.keys()
        for key, value in native.items():
            assert_no_ignored_fields(value, validated[key])
    elif isinstance(native, list) and isinstance(validated, Iterable):
        # TypedDict SDK schemas often use Iterable[Block]. Pydantic validates
        # those lazily; consume the iterator before trusting nested fields.
        validated = list(validated)
        assert len(native) == len(validated)
        for value, parsed in zip(native, validated):
            assert_no_ignored_fields(value, parsed)


def assert_input_native(value, schema):
    """Validate source fields, including association IDs absent from the Ollama SDK.

    These IDs are declared by the pinned REST schema and explicitly read by the
    input encoder. Output assertions remain strict about SDK-ignored fields.
    """
    if schema is Message:
        value = {**value}
        validator("ollama", "request").validate({"model": "test", "messages": [value]})
        if (identity := value.pop("tool_call_id", None)) is not None:
            assert isinstance(identity, str) and identity
        if value.get("tool_calls"):
            calls = []
            for call in value["tool_calls"]:
                call = {**call}
                if (identity := call.pop("id", None)) is not None:
                    assert isinstance(identity, str) and identity
                calls.append(call)
            value["tool_calls"] = calls
    assert_native(value, schema)


def assert_native(value, schema):
    for protocol, event_schema in EVENT_TYPES.items():
        if schema is event_schema or (
            isinstance(event_schema, dict) and schema in event_schema.values()
        ):
            validate_event(protocol, value)
            break
    # The Ollama client normalizes REST base64 strings into Image objects before
    # validating Message; the wire format remains a base64 string array.
    sdk_value = value
    if schema is Message and value.get("images"):
        sdk_value = {**value, "images": [Image(value=image) for image in value["images"]]}
    # Keep the adapter alive while consuming any lazy SDK Iterable validators.
    adapter = TypeAdapter(schema)
    validated = adapter.validate_python(sdk_value)
    assert_no_ignored_fields(value, validated)
    # Some SDK models mark these as Optional for convenience, while the REST
    # contract requires them. Do not let model permissiveness weaken the oracle.
    if schema is types.Content:
        assert value.get("role", "user") in ("user", "model")
        for part in value["parts"]:
            for field in ("function_call", "function_response"):
                if field in part:
                    assert isinstance(part[field].get("name"), str) and part[field]["name"]
            if "function_response" in part:
                assert isinstance(part["function_response"].get("response"), dict)
    if schema is types.Tool:
        for function in value.get("function_declarations", []):
            assert isinstance(function.get("name"), str) and function["name"]
            assert isinstance(function.get("description"), str)
    if schema is Message and value.get("images"):
        assert validated.model_dump(mode="json", exclude_none=True)["images"] == value["images"]


def assert_output_native(value, schema):
    """Validate output shape while asserting that public events omit usage."""
    for protocol, event_schema in EVENT_TYPES.items():
        if schema is event_schema or (
            isinstance(event_schema, dict) and schema in event_schema.values()
        ):
            return assert_native(output_event_for_schema(protocol, value), schema)
    return assert_native(value, schema)
