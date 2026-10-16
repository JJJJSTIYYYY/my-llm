"""SDK request assembly after protocol conversion."""

from typing import Any

from google.genai import types
from openai import Omit

from my_llm.codec.anthropic.decoding.parameters import request_parameters as anthropic_parameters
from my_llm.codec.gemini.decoding.parameters import request_parameters as gemini_parameters
from my_llm.codec.ollama.decoding.parameters import request_parameters as ollama_parameters
from my_llm.codec.openai._parameters import request_parameters as openai_parameters
from my_llm.exceptions import ConfigurationError
from my_llm.profile import ResolvedProfile


def build_request(
    messages: list, tools: list | None, profile: ResolvedProfile, **parameters: Any
) -> dict:
    protocol = profile.protocol
    previous = parameters.pop("previous_response_id", None)
    has_previous = previous is not None and not isinstance(previous, Omit)
    if has_previous and (
        protocol != "openai_responses" or not isinstance(previous, str) or not previous
    ):
        raise ConfigurationError("previous_response_id requires Responses and a non-empty ID")
    if protocol in ("openai_chat", "openai_responses"):
        result = openai_parameters(profile, **parameters)
        result.update(model=profile.model, stream=True)
        result["input" if protocol == "openai_responses" else "messages"] = messages
        # An explicit [] must reach the SDK; None means omit the tools argument.
        if tools is not None:
            result["tools"] = tools
        if has_previous:
            result["previous_response_id"] = previous
        return result
    # Anthropic and Gemini use top-level system instructions, not history roles.
    if protocol == "anthropic":
        result = anthropic_parameters(profile, **parameters)
        system = [m.get("content") for m in messages if m.get("role") in ("system", "developer")]
        history = [m for m in messages if m.get("role") not in ("system", "developer")]
        if not history:
            raise ConfigurationError("Anthropic requires a non-system message")
        result.update(model=profile.model, stream=True, messages=history)
        if system:
            result["system"] = [
                block
                for content in system
                for block in (
                    [{"type": "text", "text": content}] if isinstance(content, str) else content
                )
            ]
        if tools is not None:
            result["tools"] = tools
        return result
    if protocol == "gemini":
        config = gemini_parameters(profile, **parameters)
        system = [m for m in messages if m.get("role") in ("system", "developer")]
        history = [m for m in messages if m.get("role") not in ("system", "developer")]
        if not history:
            raise ConfigurationError("Gemini requires a non-system message")
        if system:
            config["system_instruction"] = {"parts": [part for m in system for part in m["parts"]]}
        if tools is not None:
            config["tools"] = tools
        # SDK models validate supported native fields at the transport boundary.
        return {
            "model": profile.model,
            "contents": history,
            "config": types.GenerateContentConfig(**config),
        }
    result = ollama_parameters(profile, **parameters)
    result.update(model=profile.model, stream=True, messages=messages)
    if tools is not None:
        result["tools"] = tools
    return result
