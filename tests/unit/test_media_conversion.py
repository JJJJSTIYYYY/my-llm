"""Image sources are supported; file references and image hints are discarded."""

from copy import deepcopy

import pytest

from my_llm import MyCodec


@pytest.mark.parametrize("detail", ["low", "high", "auto"])
@pytest.mark.parametrize(
    "uri", ["https://example.test/image.png", "data:image/png;base64,aW1hZ2U="]
)
def test_responses_image_hint_is_discarded_while_image_source_is_preserved(uri, detail):
    incoming = [
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": uri, "detail": detail}],
        }
    ]
    result = MyCodec().convert(
        incoming, input_format="openai_responses", target_format="openai_chat"
    )["messages"]
    assert result == [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": uri}}],
        }
    ]


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
@pytest.mark.parametrize(
    "uri,source",
    [
        (
            "https://example.test/image.png",
            {"type": "url", "url": "https://example.test/image.png"},
        ),
        (
            "data:image/png;base64,aW1hZ2U=",
            {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="},
        ),
    ],
)
def test_openai_images_encode_equivalent_remote_and_inline_sources(protocol, uri, source):
    block = (
        {"type": "image_url", "image_url": {"url": uri, "detail": "high"}}
        if protocol == "openai_chat"
        else {"type": "input_image", "image_url": uri, "detail": "high"}
    )
    incoming = [{"role": "user", "content": [block]}]
    before = deepcopy(incoming)
    result = MyCodec().encode(incoming, input_format=protocol)["messages"]
    assert result[0]["content"] == [{"type": "image", "source": source}]
    assert incoming == before


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
def test_openai_files_are_discarded_even_when_the_native_protocol_supports_them(protocol):
    uri = "data:application/pdf;base64,cGRm"
    block = (
        {"type": "file", "file": {"file_data": uri}}
        if protocol == "openai_chat"
        else {"type": "input_file", "file_data": uri}
    )
    result = MyCodec().encode([{"role": "user", "content": [block]}], input_format=protocol)[
        "messages"
    ]
    assert result[0]["content"] == []


@pytest.mark.parametrize(
    "protocol,block",
    [
        ("openai_chat", {"type": "file", "file": {"file_id": "file_123"}}),
        ("openai_responses", {"type": "input_file", "file_id": "file_123"}),
        ("openai_responses", {"type": "input_image", "file_id": "file_123", "detail": "high"}),
    ],
)
def test_file_id_media_is_discarded_because_images_support_urls_or_base64(protocol, block):
    incoming = [{"role": "user", "content": [{**block, "vendor": {"nested": [1, None]}}]}]
    original = deepcopy(incoming)
    codec = MyCodec()
    intermediate = codec.encode(incoming, input_format=protocol)
    assert intermediate["messages"][0]["content"] == []
    translated = codec.decode(**intermediate, target_format=protocol)["messages"]
    assert "file_123" not in str(translated)
    assert "vendor" not in str(translated)
    assert incoming == original
