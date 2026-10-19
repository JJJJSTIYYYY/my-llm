"""Real official SDKs over immediate, incremental in-memory HTTP responses."""

import json
from contextlib import ExitStack
from unittest.mock import patch

import httpx
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types
from ollama import AsyncClient
from openai import AsyncOpenAI


def wire_frames(protocol: str, events: list[dict], chunk_bytes: int) -> tuple[bytes, ...]:
    frames = []
    for event in events:
        if protocol == "gemini":
            event = types.GenerateContentResponse.model_validate(event).model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        encoded = json.dumps(event, ensure_ascii=False).encode()
        if protocol == "ollama":
            frames.append(encoded + b"\n")
        else:
            prefix = (
                f"event: {event['type']}\n".encode()
                if protocol in ("anthropic", "openai_responses")
                else b""
            )
            frames.append(prefix + b"data: " + encoded + b"\n\n")
    if protocol == "openai_chat":
        frames.append(b"data: [DONE]\n\n")
    # Splitting UTF-8 and SSE/NDJSON boundaries exercises the real SDK decoders.
    return tuple(
        frame[offset : offset + chunk_bytes]
        for frame in frames
        for offset in range(0, len(frame), chunk_bytes)
    )


class Body(httpx.AsyncByteStream):
    def __init__(self, frames: tuple[bytes, ...]):
        self.frames = frames
        self.closed = False
        self.read = 0

    async def __aiter__(self):
        for frame in self.frames:
            self.read += 1
            yield frame

    async def aclose(self):
        self.closed = True


class LocalSDK:
    """Patch SDK construction only; MyLLM/codec/stream execution stays untouched."""

    def __init__(self, protocol: str, frames: tuple[bytes, ...]):
        self.protocol = protocol
        self.frames = frames
        self.bodies: list[Body] = []
        self.requests: list[httpx.Request] = []
        self.http_clients: list[httpx.AsyncClient] = []
        self.created_clients = 0
        self._patches = ExitStack()

    def respond(self, request: httpx.Request):
        self.requests.append(request)
        body = Body(self.frames)
        self.bodies.append(body)
        return httpx.Response(
            200,
            stream=body,
            headers={
                "content-type": "application/x-ndjson"
                if self.protocol == "ollama"
                else "text/event-stream"
            },
        )

    def __enter__(self):
        protocol = self.protocol
        sdk_class = {
            "openai_chat": AsyncOpenAI,
            "openai_responses": AsyncOpenAI,
            "anthropic": AsyncAnthropic,
            "gemini": genai.Client,
            "ollama": AsyncClient,
        }[protocol]

        def factory(**kwargs):
            self.created_clients += 1
            transport = httpx.MockTransport(self.respond)
            if protocol == "gemini":
                options = kwargs.pop("http_options")
                options = options.model_copy(
                    update={
                        "async_client_args": {
                            **options.async_client_args,
                            "transport": transport,
                            "trust_env": False,
                        }
                    }
                )
                return sdk_class(**kwargs, http_options=options)
            if protocol == "ollama":
                return sdk_class(**kwargs, transport=transport, trust_env=False)
            http = httpx.AsyncClient(transport=transport, trust_env=False)
            self.http_clients.append(http)
            return sdk_class(**kwargs, http_client=http)

        target = {
            "openai_chat": "my_llm.codec.openai.chat.transport.AsyncOpenAI",
            "openai_responses": "my_llm.codec.openai.responses.transport.AsyncOpenAI",
            "anthropic": "my_llm.codec.anthropic.transport.AsyncAnthropic",
            "gemini": "my_llm.codec.gemini.transport.genai.Client",
            "ollama": "my_llm.codec.ollama.transport.AsyncOllama",
        }[protocol]
        self._patches.enter_context(patch(target, factory))
        return self

    def __exit__(self, *exc):
        self._patches.__exit__(*exc)

    def release_samples(self):
        """Drop observations outside timing, keeping the benchmark memory bounded."""
        self.requests.clear()
        self.bodies.clear()
