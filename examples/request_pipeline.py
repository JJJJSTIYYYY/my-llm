"""Inspect the complete request pipeline offline, without constructing an SDK client.

Run with: python examples/request_pipeline.py
"""

from my_llm import ModelProfile, MyCodec, ProviderProfile, omit, resolve_profiles


def main():
    codec = MyCodec()
    profile = resolve_profiles(
        ProviderProfile(protocol="anthropic"),
        ModelProfile(model="example-model"),
    )
    codec.validate_profile(profile)
    # Prepare connection kwargs once; the same read-only settings serve later calls.
    client = codec.client_parameters(profile)
    inputs = dict(
        messages=[{"role": "user", "content": "Hello"}],
        tools=omit,
        stream=False,
        max_tokens=512,
        reasoning=None,
        reasoning_effort="none",
        extra_headers=None,
        extra_body={"temperature": 0.2},
        previous_response_id=None,
    )
    prepared = codec.prepare(inputs, profile, input_format="openai_chat", client=client)
    assert set(prepared) == {"client", "request", "local"}
    assert prepared["request"]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    ]
    assert prepared["request"]["max_tokens"] == 512
    assert prepared["local"] == {"stream": False, "client_key": None}
    assert prepared["request"]["stream"] is True
    print(prepared)
    return prepared


if __name__ == "__main__":
    main()
