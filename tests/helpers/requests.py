"""Native request assembly for tests through the public codec API."""

from types import SimpleNamespace

from my_llm import MyCodec


def encode_request(profile, messages=None, tools=None, *, tools_format="openai_chat", **parameters):
    codec = MyCodec()
    intermediate = codec.encode(
        messages or [{"role": "user", "content": "x"}], input_format="openai_chat"
    )
    if tools is not None and not isinstance(tools, __import__("openai").Omit):
        intermediate.update(codec.encode(tools=tools, input_format=tools_format))
    return codec.request(intermediate, profile, **parameters)


def codec(profile):
    return SimpleNamespace(
        encode=lambda messages, tools=None, **parameters: encode_request(
            profile, messages, tools, **parameters
        )
    )
