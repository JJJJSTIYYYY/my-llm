"""Protocol-independent model capabilities and their semantic loss reasons."""

from my_llm.basic.semantic_loss import report_loss
from my_llm.profile.resolution import ResolvedProfile
from my_llm.profile.types import Omit, Parameter

_CONTENT_CAPABILITIES = {
    "image": ("supports_vision", False, "image input"),
    "tool_use": ("supports_function_calling", False, "function calling"),
    "tool_result": ("supports_function_calling", False, "function calling"),
    "tool_definition": ("supports_function_calling", False, "function calling"),
    "message_name": ("supports_message_name", True, "message names"),
}


def supports_content(kind: str, profile: ResolvedProfile | None) -> bool:
    """Read model support without recording a loss or applying history policy."""
    if profile is None:
        return True
    if kind == "thinking":
        declaration = profile.parameters.get("reasoning_enable")
        return not isinstance(declaration, Parameter) or declaration.supports
    capability = _CONTENT_CAPABILITIES.get(kind)
    if capability is None:
        return True
    value = profile.values.get(capability[0], capability[1])
    return not isinstance(value, Omit) and bool(value)


def content_capability(
    value: object,
    kind: str,
    profile: ResolvedProfile | None,
    *,
    native_support: bool = True,
    path: str | None = None,
) -> bool:
    """Report an unsupported model/protocol capability at the value's source."""
    supported = supports_content(kind, profile)
    if supported and native_support:
        return True
    label = "reasoning content" if kind == "thinking" else _CONTENT_CAPABILITIES[kind][2]
    owner = "model" if not supported else "target protocol"
    report_loss(value, f"{owner} does not support {label}", path=path)
    return False
