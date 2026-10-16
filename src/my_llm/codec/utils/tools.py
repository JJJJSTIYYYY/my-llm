"""Shared function declarations; unknown tool dictionaries remain opaque."""

from my_llm.codec.utils.conversion import (
    dictionaries,
    extras,
    original,
    remember,
    residual,
    source_protocol,
)
from my_llm.codec.utils.keys import FUNCTIONS


def encode_functions(tools, protocol, unwrap):
    result = []
    for tool in dictionaries(tools):
        functions = unwrap(tool)
        if functions is None and (
            tool.get("type") == "function"
            or "input_schema" in tool
            or "function_declarations" in tool
        ):
            from my_llm.exceptions import MessageValidationError

            raise MessageValidationError(f"tool does not use the bound {protocol} protocol")
        declaration = dict.fromkeys(("name", "description", "parameters", "strict"), True)
        consumed = {}
        if functions is not None:
            if protocol in ("openai_chat", "ollama"):
                consumed = {"type": True, "function": declaration}
            elif protocol == "gemini":
                consumed = {
                    "function_declarations": [
                        {
                            "name": True,
                            "description": True,
                            "parameters": True,
                            "parameters_json_schema": True,
                        }
                    ]
                }
            elif protocol == "anthropic":
                consumed = {"name": True, "description": True, "input_schema": True}
            else:
                consumed = {"type": True, **declaration}
        result.append(
            remember(
                {**residual(tool, consumed), FUNCTIONS: functions},
                tool,
                protocol,
            )
        )
    return result


def decode_functions(tools, protocol, wrap):
    result = []
    for tool in dictionaries(tools):
        native = original(tool, protocol)
        if native is not None:
            result.append(native)
        elif tool.get(FUNCTIONS) is None:
            result.append(extras(tool, FUNCTIONS))
        else:
            extensions = extras(
                tool,
                FUNCTIONS,
                "type",
                "function",
                "function_declarations",
                "input_schema",
                "name",
                "description",
                "parameters",
                "strict",
            )
            # Consuming known leaves can remove the original wrapper entirely;
            # provenance, rather than wrapper presence, identifies its format.
            if source_protocol(tool) not in ("openai_chat", "ollama", "gemini"):
                extensions = {}
            result.extend({**extensions, **value} for value in wrap(tool[FUNCTIONS]))
    return result
