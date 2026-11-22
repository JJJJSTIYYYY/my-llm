"""Translate supported messages through the public middleware without network I/O.

Run with: python examples/translation.py
"""

from my_llm import MyCodec


def main():
    codec = MyCodec()
    messages = [{"role": "user", "content": "Hello"}]
    middleware = codec.encode(messages=messages, input_format="openai_chat")
    converted = codec.decode(middleware.messages, target_format="anthropic")
    assert converted == codec.convert(
        messages=messages, input_format="openai_chat", target_format="anthropic"
    )
    print(converted)
    return converted


if __name__ == "__main__":
    main()
