"""Native adapters own supported payloads once and discard unknown output locally."""

from copy import deepcopy

import pytest

from my_llm import MyCodec


@pytest.mark.parametrize("protocol", ["anthropic"])
def test_stream_tool_arguments_are_owned_once_at_the_middleware_boundary(protocol):
    copies = []

    class TrackedArguments(dict):
        def __deepcopy__(self, memo):
            copies.append(self)
            return deepcopy(dict(self), memo)

    arguments = TrackedArguments(items=[{"value": 1}])
    event = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "call_1",
            "name": "lookup",
            "input": arguments,
        },
    }

    middleware = MyCodec().encode(chunks=[event], input_format=protocol)
    tool = middleware.stream[0].events[-1]["block"]
    assert len(copies) == 1
    assert tool["input"] == arguments
    arguments["items"][0]["value"] = 2
    assert tool["input"] == {"items": [{"value": 1}]}
    original = event["content_block"]
    assert tool.origin_value is original


@pytest.mark.parametrize("protocol", ["anthropic"])
def test_native_stream_projection_discards_generated_media_before_middleware(protocol):
    from my_llm.codec.anthropic.encoding.encoder import AnthropicEncoder

    encoder = AnthropicEncoder()
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="},
            },
        }
    ]
    projected = encoder._chunks(events)
    assert all("block" not in operation for batch in projected for operation in batch["events"])
