"""Shared function declarations; unknown tool dictionaries remain opaque."""

from my_llm.codec.shared.conversion import (
    dictionaries,
    original,
    remember,
    residual,
)
from my_llm.codec.shared.keys import FUNCTIONS
from my_llm.exceptions import UnsupportedCapabilityError


def encode_functions(tools, protocol, unwrap, *, consumed):
    result = []
    for tool in dictionaries(tools):
        functions = unwrap(tool)
        fields = consumed(tool) if callable(consumed) else consumed
        result.append(
            remember(
                {**residual(tool, fields if functions is not None else {}), FUNCTIONS: functions},
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
            raise UnsupportedCapabilityError(
                "opaque native tools cannot be replayed to another protocol"
            )
        else:
            result.extend(wrap(tool[FUNCTIONS]))
    return result
