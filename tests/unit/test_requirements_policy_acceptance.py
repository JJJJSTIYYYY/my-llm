"""Explicitly supplied loss hooks are selected by presence, not truth value."""

import pytest

from my_llm import MyCodec


@pytest.mark.parametrize("direction", ["input", "output"])
def test_falsey_callable_hook_receives_the_complete_delivered_draft(direction):
    seen = []

    class Hook:
        def __bool__(self):
            return False

        def __call__(self, draft, losses, context):
            seen.append((draft, losses, context))
            messages = draft["messages"] if direction == "input" else [draft]
            messages[0]["content"].append({"type": "text", "text": "repaired"})

    hook = Hook()
    codec = MyCodec("hook", input_semantic_loss_hook=hook, output_semantic_loss_hook=hook)
    values = [{"role": "assistant", "content": "answer", "reasoning_content": "reason"}]
    if direction == "input":
        output = codec.convert(values, input_format="openai_chat", target_format="anthropic")
        messages = output["messages"]
    else:
        encoded = codec.encode(values, input_format="openai_chat")
        output = codec.result(
            encoded.messages, input_format="openai_chat", target_format="anthropic"
        )
        messages = [output]
    assert len(seen) == 1
    draft, losses, context = seen[0]
    assert draft is output
    assert losses.origin_source == context.origin_source == direction
    assert messages[0]["content"] == [
        {"type": "text", "text": "answer"},
        {"type": "text", "text": "repaired"},
    ]
