"""Custom encoder validation applies to direct checks and prepared requests."""

import pytest

from my_llm import MessageValidationError
from my_llm.codec.openai.chat.encoding.encoder import ChatCompletionsEncoder


class GuardedEncoder(ChatCompletionsEncoder):
    def _validation_messages(self, messages):
        if any(message.get("reject_projection") for message in messages):
            raise MessageValidationError("custom message projection")
        return super()._validation_messages(messages)

    def _validation_tools(self, tools):
        if any(tool.get("reject_projection") for tool in tools):
            raise MessageValidationError("custom tool projection")
        return super()._validation_tools(tools)

    def validate_values(self, values, schema, *, filtered=False):
        if schema is self.message_type and any(
            message.get("content") == "reject-values" for message in values
        ):
            raise MessageValidationError("custom value validation")
        super().validate_values(values, schema, filtered=filtered)


@pytest.mark.parametrize("boundary", ["validate_input", "request"])
@pytest.mark.parametrize("guard", ["message projection", "tool projection", "value validation"])
def test_custom_validation_hooks_reject_inputs_at_both_boundaries(profile_factory, boundary, guard):
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    if guard == "message projection":
        messages[0]["reject_projection"] = True
    elif guard == "tool projection":
        tools[0]["reject_projection"] = True
    else:
        messages[0]["content"] = "reject-values"

    encoder = GuardedEncoder()
    with pytest.raises(MessageValidationError, match="bound input protocol") as caught:
        if boundary == "validate_input":
            encoder.validate_input(messages, tools)
        else:
            encoder.request({"messages": messages, "tools": tools}, profile_factory())
    assert str(caught.value.__cause__) == f"custom {guard}"
