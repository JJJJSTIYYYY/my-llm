"""Concurrent MyLLM requests through real SDKs and synchronized local streams."""

import asyncio
import json
import re
from copy import deepcopy

import httpx
import pytest
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types
from ollama import AsyncClient
from openai import AsyncOpenAI

from my_llm import ModelProfile, MyCodec, MyLLM, ProviderProfile
from my_llm.stream._state import StreamState
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import PROTOCOLS, history, stream_events, tools
from tests.helpers.transport import sse

GenAIClient = genai.Client


def replace_strings(value, replacements):
    if isinstance(value, dict):
        return {key: replace_strings(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [replace_strings(child, replacements) for child in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def response_frames(protocol, index):
    events = replace_strings(
        stream_events(protocol, tool=True), {"Hello": f"reply-{index}", "上海": f"city-{index}"}
    )
    # Give every lane distinct counters as well as text and tool arguments.
    prompt, completion = index + 1, index + 2
    if protocol == "openai_chat":
        events[-1]["usage"].update(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        )
    elif protocol == "openai_responses":
        events[-1]["response"]["usage"].update(
            input_tokens=prompt, output_tokens=completion, total_tokens=prompt + completion
        )
    elif protocol == "anthropic":
        events[0]["message"]["usage"].update(input_tokens=prompt)
        events[-2]["usage"].update(output_tokens=completion)
    elif protocol == "gemini":
        events[-1]["usage_metadata"].update(
            prompt_token_count=prompt,
            candidates_token_count=completion,
            total_token_count=prompt + completion,
        )
        events = [
            types.GenerateContentResponse.model_validate(event).model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
            for event in events
        ]
    else:
        events[-1].update(prompt_eval_count=prompt, eval_count=completion)
    if protocol == "ollama":
        return [(json.dumps(event) + "\n").encode() for event in events]
    frames = [
        sse([event], anthropic=protocol == "anthropic", done=False).encode() for event in events
    ]
    if protocol == "openai_chat":
        frames.append(b"data: [DONE]\n\n")
    return frames


class ControlledBody(httpx.AsyncByteStream):
    def __init__(self, harness, protocol, index):
        self.harness = harness
        self.protocol = protocol
        self.index = index
        self.closed = False
        self.resume = asyncio.Event()

    async def __aiter__(self):
        frames = response_frames(self.protocol, self.index)
        self.harness.active.add(self.index)
        self.harness.peak = max(self.harness.peak, len(self.harness.active))
        text_seen = False
        for offset, frame in enumerate(frames):
            # All lanes must enter parsing before any one can finish. Successful
            # runs also rendezvous on every frame, forcing interleaved fragments.
            if offset == 0 or self.harness.every_frame:
                await self.harness.barrier.wait()
            if text_seen and self.harness.paused:
                await self.resume.wait()
                if self.index == self.harness.failure_index:
                    raise RuntimeError("injected stream failure")
            yield frame
            text_seen |= f"reply-{self.index}".encode() in frame

    async def aclose(self):
        self.closed = True
        self.harness.active.discard(self.index)


class ConcurrentTransport:
    def __init__(self, monkeypatch, count, *, paused=False, every_frame=True):
        self.barrier = asyncio.Barrier(count)
        self.paused = paused
        self.every_frame = every_frame
        self.failure_index = None
        self.requests = {}
        self.bodies = {}
        self.clients = []
        self.active = set()
        self.peak = 0

        def handler(protocol, request):
            # Both OpenAI APIs share a constructor; route by the real endpoint.
            if protocol == "openai_chat" and request.url.path.endswith("/responses"):
                protocol = "openai_responses"
            index_header = request.headers.get("X-Request-Index")
            index = (
                int(index_header)
                if index_header is not None
                else int(re.search(r"request-(\d+)", request.content.decode())[1])
            )
            assert index not in self.requests
            self.requests[index] = json.loads(request.content)
            body = ControlledBody(self, protocol, index)
            self.bodies[index] = body
            content_type = "application/x-ndjson" if protocol == "ollama" else "text/event-stream"
            return httpx.Response(200, stream=body, headers={"Content-Type": content_type})

        def factory(protocol, **kwargs):
            transport = httpx.MockTransport(lambda request: handler(protocol, request))
            if protocol == "gemini":
                options = kwargs.pop("http_options")
                client = GenAIClient(
                    **kwargs,
                    http_options=options.model_copy(
                        update={
                            "async_client_args": {
                                **(options.async_client_args or {}),
                                "transport": transport,
                            }
                        }
                    ),
                )
                http = client._api_client._async_httpx_client
            elif protocol == "ollama":
                client = AsyncClient(**kwargs, transport=transport)
                http = client._client
            else:
                http = httpx.AsyncClient(transport=transport)
                sdk = AsyncAnthropic if protocol == "anthropic" else AsyncOpenAI
                client = sdk(**kwargs, http_client=http)
            self.clients.append(http)
            return client

        monkeypatch.setattr(
            "my_llm.codec.openai.chat.transport.AsyncOpenAI",
            lambda **kw: factory("openai_chat", **kw),
        )
        monkeypatch.setattr(
            "my_llm.codec.openai.responses.transport.AsyncOpenAI",
            lambda **kw: factory("openai_chat", **kw),
        )
        monkeypatch.setattr(
            "my_llm.codec.anthropic.transport.AsyncAnthropic",
            lambda **kw: factory("anthropic", **kw),
        )
        monkeypatch.setattr(
            "my_llm.codec.ollama.transport.AsyncOllama",
            lambda **kw: factory("ollama", **kw),
        )
        monkeypatch.setattr(
            "my_llm.codec.gemini.transport.genai.Client",
            lambda **kw: factory("gemini", **kw),
        )

    def assert_responses_closed(self, count):
        assert len(self.bodies) == count
        assert all(body.closed for body in self.bodies.values()), [
            index for index, body in self.bodies.items() if not body.closed
        ]
        assert not self.active

    def assert_clients_open(self, count):
        assert len(self.clients) == count
        assert all(not client.is_closed for client in self.clients)


def make_llm(protocol, stdio):
    return MyLLM(
        provider=ProviderProfile(
            protocol=protocol,
            api_key=parameter(
                "api_key",
                "Bearer local-concurrency-test"
                if protocol == "ollama"
                else "local-concurrency-test",
                protocol=protocol,
            ),
            base_url=config("https://local.test"),
            max_retries=config(0),
            usage_mode=config("stream_options" if protocol == "openai_chat" else "native"),
        ),
        model=ModelProfile(
            model="demo",
            max_tokens=parameter("max_tokens", 2048, protocol=protocol),
            supports_function_calling=config(True),
        ),
        # Exercise unique format binding while other requests are in flight.
        stdio_protocol=None if stdio == "gemini" else stdio,
    )


def assert_reply(result, stdio, index):
    native = result if stdio == "openai_responses" else [result]
    chat = MyCodec().convert(native, input_format=stdio, target_format="openai_chat")["messages"]
    assert chat[0]["content"] == f"reply-{index}"
    calls = [call for message in chat for call in message.get("tool_calls", [])]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": f"city-{index}"}


def assert_usage(events, stdio, index):
    if stdio == "openai_chat":
        usage = events[-1]["usage"]
        assert usage["prompt_tokens"] == index + 1
        assert usage["completion_tokens"] == index + 2
    elif stdio == "openai_responses":
        assert [event["sequence_number"] for event in events] == list(range(len(events)))
        usage = events[-1]["response"]["usage"]
        assert usage["input_tokens"] == index + 1
        assert usage["output_tokens"] == index + 2
    elif stdio == "anthropic":
        assert events[-2]["usage"]["input_tokens"] == index + 1
        assert events[-2]["usage"]["output_tokens"] == index + 2
    elif stdio == "gemini":
        assert events[-1]["usage_metadata"]["prompt_token_count"] == index + 1
        assert events[-1]["usage_metadata"]["candidates_token_count"] == index + 2
    else:
        assert events[-1]["prompt_eval_count"] == index + 1
        assert events[-1]["eval_count"] == index + 2


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("stdio", PROTOCOLS)
async def test_one_instance_concurrent_stream_and_collected_requests(monkeypatch, protocol, stdio):
    count = 8
    transport = ConcurrentTransport(monkeypatch, count)
    llm = make_llm(protocol, stdio)
    shared_tools = tools(stdio)
    before_tools = deepcopy(shared_tools)
    histories = [
        replace_strings(history(stdio), {"Weather?": f"request-{i}"}) for i in range(count)
    ]
    before_histories = deepcopy(histories)

    async def run(index):
        streaming = index % 2 == 0
        result = await llm.generate(
            messages=histories[index],
            tools=shared_tools,
            stream=streaming,
            max_tokens=101 + index,
            extra_headers={"X-Request-Index": str(index)},
        )
        if streaming:
            async with result:
                events = [event async for event in result]
            assert_usage(events, stdio, index)
            # Validate the emitted content as well as the final snapshot: a
            # correct result must not hide transient cross-request stream data.
            codec = MyCodec()
            replay = StreamState("demo")
            replay.feed(codec.encode(chunks=events, input_format=stdio)["chunks"])
            replay.finish()
            chat = codec.decode(replay.messages(), target_format="openai_chat")["messages"]
            assert_reply(chat[0], "openai_chat", index)
            result = result.result
        assert_reply(result, stdio, index)
        return result

    # A deadline turns a serialization regression into a bounded test failure.
    async with asyncio.timeout(15), asyncio.TaskGroup() as group:
        tasks = [group.create_task(run(index)) for index in range(count)]
    assert all(task.done() for task in tasks)
    assert transport.peak == count
    transport.assert_responses_closed(count)
    transport.assert_clients_open(count if protocol == "ollama" else 1)
    assert llm.stdio_protocol == stdio
    assert shared_tools == before_tools and histories == before_histories
    for index, body in transport.requests.items():
        assert f"request-{index}" in str(body)
        maximum = {
            "openai_chat": body.get("max_completion_tokens"),
            "openai_responses": body.get("max_output_tokens"),
            "anthropic": body.get("max_tokens"),
            "gemini": body.get("generationConfig", {}).get("maxOutputTokens"),
            "ollama": body.get("options", {}).get("num_predict"),
        }[protocol]
        assert maximum == 101 + index
    await llm.aclose()
    assert all(client.is_closed for client in transport.clients)


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_failure_cancellation_and_early_exit_do_not_affect_other_requests(
    monkeypatch, protocol
):
    count = 5
    transport = ConcurrentTransport(monkeypatch, count, paused=True, every_frame=False)
    transport.failure_index = 2
    llm = make_llm(protocol, "openai_chat")
    ready = [asyncio.Event() for _ in range(count)]
    streams = {}

    async def run(index):
        stream = await llm.generate(
            messages=[{"role": "user", "content": f"request-{index}"}],
            stream=True,
        )
        streams[index] = stream
        try:
            async with stream:
                async for event in stream:
                    if any(choice["delta"].get("content") for choice in event.get("choices", [])):
                        ready[index].set()
                        if index == 0:
                            return None  # Early exit closes only this request.
        except RuntimeError as exc:
            assert index == 2 and str(exc) == "injected stream failure"
            return exc
        assert_reply(stream.result, "openai_chat", index)
        return stream.result

    async with asyncio.timeout(15), asyncio.TaskGroup() as group:
        tasks = [group.create_task(run(index)) for index in range(count)]
        await asyncio.gather(*(event.wait() for event in ready))
        await tasks[0]
        tasks[1].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[1]
        transport.bodies[2].resume.set()
        assert isinstance(await tasks[2], RuntimeError)
        assert all(not tasks[index].done() for index in (3, 4))
        assert all(not transport.bodies[index].closed for index in (3, 4))
        assert all(transport.bodies[index].closed for index in (0, 1, 2))
        transport.assert_clients_open(1)
        assert all(streams[index].result is None for index in (0, 1, 2))
        for index in (3, 4):
            transport.bodies[index].resume.set()
    assert transport.peak == count
    transport.assert_responses_closed(count)
    # Failed/closed requests must not poison later calls on the same instance.
    transport.paused = False
    transport.barrier = asyncio.Barrier(1)
    async with asyncio.timeout(15):
        result = await llm.generate(
            messages=[{"role": "user", "content": f"request-{count}"}],
        )
    assert_reply(result, "openai_chat", count)
    transport.assert_responses_closed(count + 1)
    transport.assert_clients_open(1)
    await llm.aclose()
    assert all(client.is_closed for client in transport.clients)


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_same_client_is_reused_across_concurrent_first_requests_and_later_calls(
    monkeypatch, protocol
):
    count = 6
    transport = ConcurrentTransport(monkeypatch, count)
    async with make_llm(protocol, "openai_chat") as llm:

        async def run(index):
            result = await llm.generate(
                messages=[{"role": "user", "content": f"request-{index}"}],
                stream=index % 2 == 0,
            )
            if index % 2 == 0:
                async with result:
                    async for _ in result:
                        pass
                result = result.result
            assert_reply(result, "openai_chat", index)

        async with asyncio.timeout(15), asyncio.TaskGroup() as group:
            for index in range(count):
                group.create_task(run(index))
        transport.assert_clients_open(1)
        transport.assert_responses_closed(count)
        assert transport.peak == count
        transport.barrier = asyncio.Barrier(1)
        await run(count)
        transport.assert_clients_open(1)
        transport.assert_responses_closed(count + 1)
    assert transport.clients[0].is_closed


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_request_can_be_closed_from_another_task_without_closing_shared_client(
    monkeypatch, protocol
):
    transport = ConcurrentTransport(monkeypatch, 2, paused=True, every_frame=False)
    async with make_llm(protocol, "openai_chat") as llm:
        streams = [
            await llm.generate(
                messages=[{"role": "user", "content": f"request-{index}"}], stream=True
            )
            for index in range(2)
        ]

        async def read_text(stream):
            async for event in stream:
                if any(choice["delta"].get("content") for choice in event.get("choices", [])):
                    return

        async with asyncio.timeout(15), asyncio.TaskGroup() as group:
            for stream in streams:
                group.create_task(read_text(stream))
        # The tasks that read the SDK streams have exited. Their owner now
        # closes one response from a different task and continues the other.
        await streams[0].aclose()
        assert transport.bodies[0].closed and not transport.bodies[1].closed
        transport.assert_clients_open(1)
        transport.bodies[1].resume.set()
        async with streams[1]:
            async for _ in streams[1]:
                pass
        assert_reply(streams[1].result, "openai_chat", 1)
        transport.assert_responses_closed(2)
    assert transport.clients[0].is_closed
