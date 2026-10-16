"""Offline demonstration of standard messages/tools across five protocols.

Run with uv run python examples/local_demo.py. No credentials or network needed.
"""

from my_llm import MyCodec


def main():
    codec = MyCodec()
    messages = [{"role": "user", "content": "Hello"}]
    for protocol in codec.codecs:
        converted = codec.convert(messages, input_format="openai_chat", target_format=protocol)
        restored = codec.convert(**converted, input_format=protocol, target_format="openai_chat")
        assert restored["messages"][0]["role"] == "user"
        assert "Hello" in str(restored)
        print(protocol, converted["messages"])
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                },
            },
        }
    ]
    print(
        "Gemini tools:",
        codec.convert(tools=tools, input_format="openai_chat", target_format="gemini")["tools"],
    )


if __name__ == "__main__":
    main()
