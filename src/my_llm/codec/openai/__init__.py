"""Independent OpenAI Chat Completions and Responses codecs."""

from my_llm.codec.openai.chat import ChatCompletionsCodec
from my_llm.codec.openai.responses import ResponsesCodec

__all__ = ["ChatCompletionsCodec", "ResponsesCodec"]
