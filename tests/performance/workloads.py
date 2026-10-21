"""Scalable native fixtures; no production codec is used to construct the inputs."""

import base64
import json
from copy import deepcopy
from dataclasses import dataclass

from my_llm import Attribute, Config, ModelProfile, ProviderProfile
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import chat_event, response_completed

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
SIGNATURE = base64.b64encode(b"offline-benchmark-signature").decode()


def pdf_data() -> str:
    """A small self-contained PDF with correct object offsets, built outside timing."""
    content = b"BT /F1 12 Tf 20 50 Td (Offline benchmark source) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    document = b"%PDF-1.4\n"
    offsets = []
    for index, value in enumerate(objects, 1):
        offsets.append(len(document))
        document += f"{index} 0 obj\n".encode() + value + b"\nendobj\n"
    xref = len(document)
    document += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    document += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets)
    document += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return "data:application/pdf;base64," + base64.b64encode(document).decode()


PDF = pdf_data()


@dataclass(frozen=True, kw_only=True)
class Size:
    rounds: int = 12
    tools: int = 12
    parallel_calls: int = 4
    text_chars: int = 2048
    output_chars: int = 8192
    fragment_chars: int = 96
    argument_items: int = 8
    schema_depth: int = 3

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.parallel_calls > self.tools:
            raise ValueError("parallel_calls cannot exceed tools")


@dataclass
class Workload:
    protocol: str
    stdio: str
    native: bool
    size: Size
    messages: list
    tools: list
    events: list
    answer: str
    reasoning: str
    calls: list


def text(length: int) -> str:
    value = '性能 benchmark / 上海 / 😀 / "quoted" / \\ / \n'
    return (value * (length // len(value) + 1))[:length]


def fragments(value: str, length: int):
    return [value[index : index + length] for index in range(0, len(value), length)]


def details(depth: int, *, schema: bool):
    if schema:
        properties = {
            "label": {"type": "string"},
            "values": {"type": "array", "items": {"type": "number"}},
        }
        if depth > 1:
            properties["child"] = details(depth - 1, schema=True)
        return {"type": "object", "properties": properties, "required": list(properties)}
    result = {"label": text(48), "values": [1, 2.5, -3, 0]}
    if depth > 1:
        result["child"] = details(depth - 1, schema=False)
    return result


def parameters(size: Size, *, extended: bool):
    properties = {
        "query": {"type": "string", "description": text(256)},
        "items": {
            "type": "array",
            "items": details(size.schema_depth, schema=True),
            "description": "Nested records with numeric arrays and Unicode labels",
        },
        "mode": {"type": "string", "enum": ["fast", "complete"]},
    }
    result = {"type": "object", "properties": properties, "required": list(properties)}
    if extended:
        result["additionalProperties"] = False
    return result


def arguments(size: Size, index: int):
    return {
        "query": f"query-{index}: {text(128)}",
        "items": [details(size.schema_depth, schema=False) for _ in range(size.argument_items)],
        "mode": "complete",
    }


def native_tools(protocol: str, size: Size, native: bool):
    functions = [
        {
            "name": f"analyze_{index}",
            "description": text(512),
            "parameters": parameters(size, extended=native and protocol != "ollama"),
        }
        for index in range(size.tools)
    ]
    if protocol == "openai_responses":
        return [{"type": "function", **function, "strict": False} for function in functions]
    if protocol == "anthropic":
        return [
            {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
                **({"cache_control": {"type": "ephemeral"}} if native and index == 0 else {}),
            }
            for index, function in enumerate(functions)
        ]
    if protocol == "gemini":
        return [
            {
                "function_declarations": [
                    {
                        "name": function["name"],
                        "description": function["description"],
                        "parameters_json_schema": function["parameters"],
                    }
                    for function in functions
                ]
            }
        ]
    return [{"type": "function", "function": function} for function in functions]


def user_message(protocol: str, value: str, *, media: bool, native: bool):
    if protocol == "openai_chat":
        content = [{"type": "text", "text": value}]
        if media:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{PNG}", "detail": "high"},
                }
            )
            if native:
                content.append(
                    {"type": "file", "file": {"file_data": PDF, "filename": "benchmark.pdf"}}
                )
        return {
            "role": "user",
            "content": content,
            **({"name": "benchmark_user"} if native else {}),
        }
    if protocol == "openai_responses":
        content = [{"type": "input_text", "text": value}]
        if media:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{PNG}",
                    "detail": "high",
                }
            )
            if native:
                content.append(
                    {"type": "input_file", "file_data": PDF, "filename": "benchmark.pdf"}
                )
        return {"type": "message", "role": "user", "content": content}
    if protocol == "anthropic":
        content = [{"type": "text", "text": value}]
        if media:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": PNG},
                }
            )
            if native:
                content.append(
                    {
                        "type": "document",
                        "source": {"type": "text", "media_type": "text/plain", "data": value},
                        "title": "Benchmark source",
                        "citations": {"enabled": True},
                    }
                )
        return {"role": "user", "content": content}
    if protocol == "gemini":
        parts = [{"text": value}]
        if media:
            parts.append({"inline_data": {"mime_type": "image/png", "data": PNG}})
            if native:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": "text/plain",
                            "data": base64.b64encode(value.encode()).decode(),
                        }
                    }
                )
        return {"role": "user", "parts": parts}
    return {"role": "user", "content": value, **({"images": [PNG]} if media else {})}


def assistant_turn(protocol: str, index: int, calls: list, native: bool):
    value = f"Analyzing round {index}"
    output = {"status": "ok", "records": [call["input"] for call in calls]}
    rendered = json.dumps(output, ensure_ascii=False)
    if protocol == "openai_chat":
        return [
            {
                "role": "assistant",
                "content": value,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["input"], ensure_ascii=False),
                        },
                    }
                    for call in calls
                ],
            },
            *[{"role": "tool", "tool_call_id": call["id"], "content": rendered} for call in calls],
        ]
    if protocol == "openai_responses":
        return [
            *(
                [
                    {
                        "type": "reasoning",
                        "id": f"rs_history_{index}",
                        "summary": [{"type": "summary_text", "text": value}],
                        "encrypted_content": SIGNATURE,
                    }
                ]
                if native
                else []
            ),
            {
                "type": "message",
                "id": f"msg_history_{index}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": value, "annotations": [], "logprobs": []}
                ],
            },
            *[
                {
                    "type": "function_call",
                    "id": f"fc_{call['id']}",
                    "call_id": call["id"],
                    "name": call["name"],
                    "arguments": json.dumps(call["input"], ensure_ascii=False),
                    "status": "completed",
                }
                for call in calls
            ],
            *[
                {"type": "function_call_output", "call_id": call["id"], "output": rendered}
                for call in calls
            ],
        ]
    if protocol == "anthropic":
        return [
            {
                "role": "assistant",
                "content": [
                    *(
                        [
                            {"type": "thinking", "thinking": value, "signature": SIGNATURE},
                            {"type": "redacted_thinking", "data": SIGNATURE},
                        ]
                        if native
                        else []
                    ),
                    {"type": "text", "text": value},
                    *[{"type": "tool_use", **call} for call in calls],
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": [{"type": "text", "text": rendered}],
                        "is_error": False,
                    }
                    for call in calls
                ],
            },
        ]
    if protocol == "gemini":
        return [
            {
                "role": "model",
                "parts": [
                    *(
                        [{"text": value, "thought": True, "thought_signature": SIGNATURE}]
                        if native
                        else []
                    ),
                    {"text": value},
                    *[
                        {
                            "function_call": {
                                "id": call["id"],
                                "name": call["name"],
                                "args": call["input"],
                            },
                            "thought_signature": SIGNATURE,
                        }
                        for call in calls
                    ],
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": call["id"],
                            "name": call["name"],
                            "response": output,
                        }
                    }
                    for call in calls
                ],
            },
        ]
    return [
        {
            "role": "assistant",
            "content": value,
            **({"thinking": value} if native else {}),
            "tool_calls": [
                {"function": {"name": call["name"], "arguments": call["input"]}} for call in calls
            ],
        },
        *[{"role": "tool", "tool_name": call["name"], "content": rendered} for call in calls],
    ]


def response_events(protocol: str, size: Size, native: bool, answer: str, reason: str, calls: list):
    pieces = fragments(answer, size.fragment_chars)
    thoughts = fragments(reason, size.fragment_chars)
    encoded = [
        fragments(json.dumps(call["input"], ensure_ascii=False), size.fragment_chars)
        for call in calls
    ]
    if protocol == "openai_chat":
        result = [chat_event(content=piece) for piece in pieces]
        for part in range(max(map(len, encoded))):
            result.append(
                chat_event(
                    calls=[
                        {
                            "index": index,
                            **({"id": call["id"], "type": "function"} if part == 0 else {}),
                            "function": {
                                **({"name": call["name"]} if part == 0 else {}),
                                "arguments": encoded[index][part],
                            },
                        }
                        for index, call in enumerate(calls)
                        if part < len(encoded[index])
                    ]
                )
            )
        result.extend(
            [
                chat_event(finish="tool_calls"),
                {
                    **chat_event(
                        usage={
                            "prompt_tokens": 4096,
                            "completion_tokens": 2048,
                            "total_tokens": 6144,
                            "prompt_tokens_details": {"cached_tokens": 512},
                            "completion_tokens_details": {"reasoning_tokens": 128},
                        }
                    ),
                    "choices": [],
                },
            ]
        )
        return result
    if protocol == "openai_responses":
        result = [
            {
                "type": "response.created",
                "response": {**response_completed()["response"], "status": "in_progress"},
            }
        ]
        output = []
        if reason:
            item = {
                "type": "reasoning",
                "id": "rs_output",
                "summary": [{"type": "summary_text", "text": reason}],
                **({"encrypted_content": SIGNATURE} if native else {}),
            }
            address = {"item_id": item["id"], "output_index": 0, "summary_index": 0}
            result.extend(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {**item, "summary": []},
                    },
                    {
                        "type": "response.reasoning_summary_part.added",
                        **address,
                        "part": {"type": "summary_text", "text": ""},
                    },
                ]
            )
            result.extend(
                {"type": "response.reasoning_summary_text.delta", **address, "delta": piece}
                for piece in thoughts
            )
            result.extend(
                [
                    {"type": "response.reasoning_summary_text.done", **address, "text": reason},
                    {
                        "type": "response.reasoning_summary_part.done",
                        **address,
                        "part": item["summary"][0],
                    },
                    {"type": "response.output_item.done", "output_index": 0, "item": item},
                ]
            )
            output.append(item)
        index = len(output)
        part = {"type": "output_text", "text": answer, "annotations": [], "logprobs": []}
        item = {
            "type": "message",
            "id": "msg_output",
            "role": "assistant",
            "status": "completed",
            "content": [part],
        }
        address = {"item_id": item["id"], "output_index": index, "content_index": 0}
        result.extend(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "status": "in_progress", "content": []},
                },
                {"type": "response.content_part.added", **address, "part": {**part, "text": ""}},
            ]
        )
        result.extend(
            {"type": "response.output_text.delta", **address, "delta": piece, "logprobs": []}
            for piece in pieces
        )
        result.extend(
            [
                {"type": "response.output_text.done", **address, "text": answer, "logprobs": []},
                {"type": "response.content_part.done", **address, "part": part},
                {"type": "response.output_item.done", "output_index": index, "item": item},
            ]
        )
        output.append(item)
        offset = len(output)
        for index, call in enumerate(calls):
            item = {
                "type": "function_call",
                "id": f"fc_output_{index}",
                "call_id": call["id"],
                "name": call["name"],
                "arguments": json.dumps(call["input"], ensure_ascii=False),
                "status": "completed",
            }
            result.append(
                {
                    "type": "response.output_item.added",
                    "output_index": offset + index,
                    "item": {**item, "arguments": "", "status": "in_progress"},
                }
            )
            output.append(item)
        for part_index in range(max(map(len, encoded))):
            for index, parts in enumerate(encoded):
                if part_index < len(parts):
                    result.append(
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": f"fc_output_{index}",
                            "output_index": offset + index,
                            "delta": parts[part_index],
                        }
                    )
        for index, item in enumerate(output[offset:]):
            result.extend(
                [
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": item["id"],
                        "output_index": offset + index,
                        "name": item["name"],
                        "arguments": item["arguments"],
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": offset + index,
                        "item": item,
                    },
                ]
            )
        result.append(
            response_completed(
                output,
                {
                    "input_tokens": 4096,
                    "output_tokens": 2048,
                    "total_tokens": 6144,
                    "input_tokens_details": {"cached_tokens": 512, "cache_write_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 128},
                },
            )
        )
        return [{**event, "sequence_number": index} for index, event in enumerate(result)]
    if protocol == "anthropic":
        result = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_benchmark",
                    "type": "message",
                    "role": "assistant",
                    "model": "benchmark",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 4096,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 512,
                        "cache_creation_input_tokens": 128,
                    },
                },
            }
        ]
        index = 0
        if reason:
            result.append(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                }
            )
            result.extend(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": piece},
                }
                for piece in thoughts
            )
            result.extend(
                [
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "signature_delta", "signature": SIGNATURE},
                    },
                    {"type": "content_block_stop", "index": index},
                ]
            )
            index += 1
        result.append(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            }
        )
        result.extend(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": piece},
            }
            for piece in pieces
        )
        if native:
            result.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "citations_delta",
                        "citation": {
                            "type": "char_location",
                            "cited_text": "source",
                            "document_index": 0,
                            "document_title": "Benchmark source",
                            "start_char_index": 0,
                            "end_char_index": 6,
                        },
                    },
                }
            )
        result.append({"type": "content_block_stop", "index": index})
        for offset, call in enumerate(calls, index + 1):
            result.append(
                {
                    "type": "content_block_start",
                    "index": offset,
                    "content_block": {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": {},
                    },
                }
            )
            result.extend(
                {
                    "type": "content_block_delta",
                    "index": offset,
                    "delta": {"type": "input_json_delta", "partial_json": piece},
                }
                for piece in encoded[offset - index - 1]
            )
            result.append({"type": "content_block_stop", "index": offset})
        result.extend(
            [
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 2048},
                },
                {"type": "message_stop"},
            ]
        )
        return result
    if protocol == "gemini":

        def event(parts):
            return {
                "response_id": "gem_benchmark",
                "model_version": "benchmark",
                "candidates": [{"index": 0, "content": {"role": "model", "parts": parts}}],
            }

        result = [
            event([{"text": piece, "thought": True, "thought_signature": SIGNATURE}])
            for piece in thoughts
        ]
        result.extend(event([{"text": piece}]) for piece in pieces)
        result.append(
            event(
                [
                    {
                        "function_call": {
                            "id": call["id"],
                            "name": call["name"],
                            "args": call["input"],
                        },
                        "thought_signature": SIGNATURE,
                    }
                    for call in calls
                ]
            )
        )
        result.append(
            {
                "response_id": "gem_benchmark",
                "candidates": [{"index": 0, "finish_reason": "STOP"}],
                "usage_metadata": {
                    "prompt_token_count": 4096,
                    "candidates_token_count": 2048,
                    "total_token_count": 6272,
                    "thoughts_token_count": 128,
                    "cached_content_token_count": 512,
                },
            }
        )
        return result

    def event(message):
        return {
            "model": "benchmark",
            "created_at": "2026-10-19T00:00:00Z",
            "message": {"role": "assistant", "content": "", **message},
            "done": False,
        }

    result = [event({"thinking": piece}) for piece in thoughts]
    result.extend(event({"content": piece}) for piece in pieces)
    result.append(
        event(
            {
                "tool_calls": [
                    {"function": {"name": call["name"], "arguments": call["input"]}}
                    for call in calls
                ]
            }
        )
    )
    result.append(
        {
            **event({}),
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 4096,
            "eval_count": 2048,
            "total_duration": 1,
            "load_duration": 0,
            "prompt_eval_duration": 0,
            "eval_duration": 1,
        }
    )
    return result


def build(protocol: str, stdio: str, size: Size, *, native: bool) -> Workload:
    if native and protocol != stdio:
        raise ValueError("native workloads require matching upstream and I/O protocols")
    messages = []
    if stdio in ("openai_chat", "openai_responses", "ollama"):
        messages.append({"role": "system", "content": text(size.text_chars)})
    if native and stdio in ("openai_chat", "openai_responses"):
        messages.append(
            {
                "role": "developer",
                "content": "Preserve source citations and use the declared tool schemas.",
            }
        )
    for index in range(size.rounds):
        messages.append(
            user_message(stdio, text(size.text_chars), media=index % 4 == 0, native=native)
        )
        calls = [
            {
                "id": f"call_{index}_{number}",
                "name": f"analyze_{number}",
                "input": arguments(size, number),
            }
            for number in range(size.parallel_calls)
        ]
        messages.extend(assistant_turn(stdio, index, calls, native))
    messages.append(
        user_message(
            stdio, "Finish the analysis and call each requested tool.", media=True, native=native
        )
    )
    answer = text(size.output_chars)
    reason = text(size.output_chars // 4) if protocol != "openai_chat" else ""
    calls = [
        {"id": f"call_output_{index}", "name": f"analyze_{index}", "input": arguments(size, index)}
        for index in range(size.parallel_calls)
    ]
    return Workload(
        protocol,
        stdio,
        native,
        size,
        messages,
        native_tools(stdio, size, native),
        response_events(protocol, size, native, answer, reason, calls),
        answer,
        reason,
        calls,
    )


def profiles(protocol: str):
    return (
        ProviderProfile(
            protocol=Config(default_value=protocol),
            api_key=parameter(
                "api_key",
                "Bearer offline-benchmark" if protocol == "ollama" else "offline-benchmark",
                protocol=protocol,
            ),
            base_url=config("https://benchmark.invalid"),
            max_retries=config(0),
            usage_mode=config("stream_options" if protocol == "openai_chat" else "native"),
            extra_headers=config({"x-benchmark": "provider"}),
        ),
        ModelProfile(
            model=Attribute(default_value="benchmark"),
            max_tokens=parameter("max_tokens", 32768, protocol=protocol),
            supports_function_calling=config(True),
            reasoning_enable=parameter("reasoning_enable", supports=True, protocol=protocol),
            reasoning_effort=parameter("reasoning_effort", supports=True, protocol=protocol),
            extra_headers=config({"x-benchmark-model": "model"}),
        ),
    )


def controls(protocol: str):
    body = {
        "openai_chat": {"temperature": 0.2, "top_p": 0.9, "parallel_tool_calls": True},
        "openai_responses": {
            "metadata": {"purpose": "offline-benchmark"},
            "parallel_tool_calls": True,
            "store": False,
        },
        "anthropic": {"temperature": 1.0, "metadata": {"user_id": "offline-benchmark"}},
        "gemini": {"temperature": 0.2, "top_p": 0.9, "candidate_count": 1},
        "ollama": {"options": {"temperature": 0.2, "top_p": 0.9, "seed": 42}, "keep_alive": "5m"},
    }[protocol]
    return {
        "reasoning": True,
        "reasoning_effort": "high",
        "max_tokens": 16384,
        "extra_headers": {"x-benchmark": "request"},
        "extra_body": deepcopy(body),
    }
