"""Independent histories and upstream responses for conversation prefix tests."""

import json

from my_llm import (
    AssistantMessage,
    FunctionTool,
    ImageContent,
    ImageSource,
    InstructionMessage,
    ReasoningContent,
    TextContent,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tests.helpers.protocols import chat_event, history, stream_events, tools
from tests.helpers.tool_images import history as image_history
from tests.performance.workloads import PNG, Size, response_events, user_message

INPUT_PROTOCOLS = ("middleware", "openai_chat", "openai_responses", "anthropic")
MODES = (
    (False, False, False, False),
    (True, True, True, True),
    (False, True, False, True),
    (True, False, True, False),
)
MODE_IDS = ("collected", "streamed", "collected-first", "streamed-first")
SCENARIOS = (
    "text",
    "system-media",
    "history",
    "single-tool",
    "parallel-tools",
    "tool-only",
    "empty-results",
    "repair-history",
    "reasoning-not-replayed",
    "tool-images",
)


def user(protocol, value, *, media=False):
    if protocol != "middleware":
        return user_message(protocol, value, media=media)
    content = [TextContent(value)]
    if media:
        content.append(ImageContent(ImageSource("base64", data=PNG, media_type="image/png")))
    return UserMessage("user", content)


def declarations(protocol):
    if protocol != "middleware":
        return tools(protocol)
    return [
        FunctionTool(
            "weather",
            description="Weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            strict=False,
        )
    ]


def initial_history(protocol, scenario):
    if scenario == "tool-images":
        return (
            [*history(protocol), user(protocol, "User supplies the image", media=True)]
            if protocol == "openai_chat"
            else image_history(protocol)
        )
    if scenario in ("history", "repair-history"):
        messages = (
            [
                user(protocol, "Weather?"),
                AssistantMessage(
                    "assistant",
                    [TextContent("Checking"), ToolCall("call_1", "weather", {"city": "上海"})],
                ),
                ToolMessage("tool", [TextContent("Sunny")], call_id="call_1"),
            ]
            if protocol == "middleware"
            else history(protocol)
        )
        if scenario == "repair-history":
            # A missing result is repaired before the first request and must stay
            # identical when the full caller history is translated again.
            messages.pop()
        return [*messages, user(protocol, "Continue from the historical tool round")]
    messages = [user(protocol, "first-input / 上海 / 😀", media=scenario == "system-media")]
    if scenario == "system-media" and protocol != "anthropic":
        system = (
            InstructionMessage("system", [TextContent("Keep these instructions")])
            if protocol == "middleware"
            else {"role": "system", "content": "Keep these instructions"}
        )
        messages.insert(0, system)
    if scenario == "reasoning-not-replayed":
        if protocol == "middleware":
            previous = [
                AssistantMessage(
                    "assistant", [ReasoningContent("old-plan"), TextContent("old-answer")]
                )
            ]
        elif protocol == "openai_chat":
            previous = [
                {"role": "assistant", "content": "old-answer", "reasoning_content": "old-plan"}
            ]
        elif protocol == "openai_responses":
            previous = [
                {
                    "type": "reasoning",
                    "id": "rs_old",
                    "summary": [{"type": "summary_text", "text": "old-plan"}],
                },
                {"role": "assistant", "content": "old-answer"},
            ]
        else:
            previous = [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "old-plan", "signature": "b2ZmbGluZQ=="},
                        {"type": "text", "text": "old-answer"},
                    ],
                }
            ]
        messages = [user(protocol, "old question"), *previous, *messages]
    return messages


def calls_for(scenario, round_index):
    if (
        scenario not in ("single-tool", "parallel-tools", "tool-only", "empty-results")
        or round_index >= 2
    ):
        return []
    return [
        {"id": f"call_r{round_index}_{index}", "name": "weather", "input": {"city": city}}
        for index, city in enumerate(
            ("上海", "北京") if scenario == "parallel-tools" else ("上海",)
        )
    ]


def compensation_history(protocol, scenario):
    """Put every compensation input in the first request's caller history."""
    messages = []
    if scenario in ("reasoning", "both"):
        messages = initial_history(protocol, "reasoning-not-replayed")
        if protocol == "middleware":
            assistant = messages[1]
            assistant.protocol = "anthropic"
            assistant.content[0].feedback_item = {
                "type": "thinking",
                "thinking": "old-plan",
                "signature": "b2ZmbGluZQ==",
            }
    if scenario in ("images", "both"):
        messages.extend(image_history(protocol))
    return messages


def upstream_events(protocol, scenario, round_index):
    calls = calls_for(scenario, round_index)
    answer = f"answer-{round_index} / 上海 / 😀"
    if calls:
        events = response_events(protocol, Size(fragment_chars=3), False, answer, "", calls)
        if scenario == "tool-only":
            if protocol == "openai_chat":
                events = [
                    event
                    for event in events
                    if not event["choices"] or "content" not in event["choices"][0]["delta"]
                ]
            elif protocol == "openai_responses":
                events = [event for event in events if event.get("output_index") != 0]
                events[-1]["response"]["output"].pop(0)
                for event in events:
                    if "output_index" in event:
                        event["output_index"] -= 1
            else:
                events = [event for event in events if event.get("index") != 0]
    elif protocol == "openai_chat":
        events = [chat_event(content=piece) for piece in (answer[:9], answer[9:])]
        events.append(chat_event(finish="stop"))
    else:
        events = stream_events(protocol)
    # Give each round different native message/item IDs. Reusing IDs would hide
    # deduplication and ordering bugs in the full-history path.
    return json.loads(
        json.dumps(events, ensure_ascii=False)
        .replace('"Hello"', json.dumps(answer, ensure_ascii=False))
        .replace('"msg_1"', f'"msg_round_{round_index}"')
        .replace('"msg_output"', f'"msg_round_{round_index}"')
        .replace('"chat_1"', f'"chat_round_{round_index}"')
        .replace('"fc_output_', f'"fc_round_{round_index}_')
    )


def reasoned_events(protocol, round_index):
    """Native reasoning with synthetic feedback, followed by tools or plain text."""
    events = upstream_events(protocol, "single-tool", round_index)
    reason = f"plan-{round_index}"
    if protocol == "openai_chat":
        return [chat_event(reasoning=reason[:3]), chat_event(reasoning=reason[3:]), *events]
    if protocol == "openai_responses":
        item = {
            "type": "reasoning",
            "id": f"rs_round_{round_index}",
            "summary": [{"type": "summary_text", "text": reason}],
            "encrypted_content": "b2ZmbGluZQ==",
        }
        for event in events:
            if "output_index" in event:
                event["output_index"] += 1
        events[-1]["response"]["output"].insert(0, item)
        events.insert(1, {"type": "response.output_item.done", "output_index": 0, "item": item})
        return [{**event, "sequence_number": index} for index, event in enumerate(events)]
    for event in events:
        if "index" in event:
            event["index"] += 1
    return [
        events[0],
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": reason},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "b2ZmbGluZQ=="},
        },
        {"type": "content_block_stop", "index": 0},
        *events[1:],
    ]


def tool_results(protocol, calls, *, empty=False):
    # Reverse parallel results to exercise matching by ID rather than position.
    if protocol == "middleware":
        return [
            ToolMessage(
                "tool", [] if empty else [TextContent(f"result-{call['id']}")], call_id=call["id"]
            )
            for call in reversed(calls)
        ]
    if protocol == "openai_chat":
        return [
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": "" if empty else f"result-{call['id']}",
            }
            for call in reversed(calls)
        ]
    if protocol == "openai_responses":
        return [
            {
                "type": "function_call_output",
                "call_id": call["id"],
                "output": [] if empty else f"result-{call['id']}",
            }
            for call in reversed(calls)
        ]
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": [] if empty else f"result-{call['id']}",
                }
                for call in reversed(calls)
            ],
        }
    ]
