"""Public interfaces for the Ollama codec."""

from my_llm.codec.ollama.codec import OllamaCodec
from my_llm.codec.ollama.decoding.stream import OllamaDecoder

__all__ = ["OllamaCodec", "OllamaDecoder"]
