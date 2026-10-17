"""Gemini SDK HTTP options and deterministic asynchronous resource cleanup."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from google import genai
from google.genai import types

from my_llm.profile.resolution import ResolvedProfile, client_parameters


@asynccontextmanager
async def open_stream(
    request: dict, profile: ResolvedProfile, **parameters: Any
) -> AsyncIterator[Any]:
    values = client_parameters(profile)
    options = {key: value for key, value in values.items() if key == "base_url"}
    async with AsyncExitStack() as clients, AsyncExitStack() as responses:

        async def track_response(response: Any) -> None:
            # Closing GenAI's outer generator can leave inner HTTPX bodies
            # suspended. Every response, including retries, is request-owned.
            responses.push_async_callback(response.aclose)

        options["async_client_args"] = {"event_hooks": {"response": [track_response]}}
        if "timeout" in values:
            options["timeout"] = max(1, round(values["timeout"] * 1000))
        if "max_retries" in values:
            options["retry_options"] = types.HttpRetryOptions(attempts=values["max_retries"] + 1)
        client = genai.Client(
            **({"api_key": values["api_key"]} if "api_key" in values else {}),
            http_options=types.HttpOptions(**options),
        )
        clients.callback(client.close)
        clients.push_async_callback(client.aio.aclose)
        upstream = await client.aio.models.generate_content_stream(**request)
        async with AsyncExitStack() as stream:
            stream.push_async_callback(upstream.aclose)
            yield upstream
