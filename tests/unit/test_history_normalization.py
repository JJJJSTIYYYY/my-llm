"""Repair tool history at the shared boundary and preserve caller ownership."""

from copy import deepcopy

import pytest

from my_llm import (
    AssistantMessage,
    MessageValidationError,
    MyCodec,
    TextContent,
    ToolCall,
    ToolResult,
    UserMessage,
    omit,
)
from my_llm.codec.shared.conversion import validate_history
from my_llm.middleware.history import HistoryNormalizer
from my_llm.middleware.protocol import wire_values
from my_llm.middleware.validation import validate_messages
from tests.helpers.profiles import config, parameter
from tests.helpers.protocols import PROTOCOLS, history, tools
from tests.helpers.schema import validate_request


def normalize(messages, **options):
    normalizer = HistoryNormalizer(**options)
    validate_messages(wire_values(messages), history=normalizer)
    return list(normalizer.finish())


def test_orphan_results_are_discarded_and_valid_duplicates_merge_without_mutating_input():
    messages = [
        UserMessage("user", [ToolResult("c", [TextContent("discarded")])]),
        AssistantMessage("assistant", [ToolCall("c", "weather", {"city": "Shanghai"})]),
        UserMessage("user", [ToolResult("c", [TextContent("first")])]),
        UserMessage("user", [ToolResult("c", [TextContent("first"), TextContent("second")])]),
        UserMessage("user", [TextContent("tail")]),
    ]
    before = deepcopy(messages)
    result = normalize(messages)
    validate_history(result)
    assert [m["role"] for m in result] == ["assistant", "user", "user"]
    assert result[0]["content"][0]["name"] == "weather"
    assert result[0]["content"][0]["input"] == {"city": "Shanghai"}
    assert [p["text"] for p in result[1]["content"][0]["content"]] == [
        "first",
        "first",
        "second",
    ]
    assert result[1]["content"][0]["name"] == "weather"
    assert result[-1]["content"][0]["text"] == "tail"
    assert messages == before


def test_missing_results_follow_their_call_message_and_duplicate_calls_keep_surrounding_text():
    result = normalize(
        [
            AssistantMessage("assistant", [ToolCall("a", "f"), ToolCall("b", "g")]),
            AssistantMessage(
                "assistant", [TextContent("explanation"), ToolCall("a", "f", {"x": 1})]
            ),
            UserMessage("user", [TextContent("tail")]),
        ]
    )
    validate_history(result)
    assert result[0]["content"][0]["input"] == {"x": 1}
    assert result[1]["role"] == "user"
    assert [result[i]["content"][0]["tool_use_id"] for i in (1, 2)] == ["a", "b"]
    assert all(result[i]["content"][0]["content"] == [] for i in (1, 2))
    assert result[3]["content"] == [{"type": "text", "text": "explanation"}]


def test_idless_results_discard_the_whole_message_and_leave_only_empty_padding():
    messages = [
        AssistantMessage("assistant", [ToolCall("a", "f"), ToolCall("b", "g")]),
        UserMessage(
            "user",
            [
                TextContent("discarded before"),
                ToolResult(None, [TextContent("discarded partial result")]),
                ToolResult("orphan", [TextContent("discarded orphan")]),
                TextContent("discarded after"),
                ToolResult("b", [TextContent("discarded later result")]),
            ],
        ),
        UserMessage(
            "user",
            [
                ToolResult(None, [TextContent("real first")]),
                ToolResult(None, [TextContent("real second")]),
            ],
        ),
    ]
    before = deepcopy(messages)
    result = normalize(messages)
    validate_history(result)
    assert len(result) == 3
    assert "discarded" not in str(result)
    assert [result[i]["content"][0]["tool_use_id"] for i in (1, 2)] == ["a", "b"]
    assert all(result[i]["content"][0]["content"] == [] for i in (1, 2))
    assert messages == before


def test_discarded_message_cannot_modify_a_previously_merged_tool_result():
    result = normalize(
        [
            AssistantMessage("assistant", [ToolCall("c", "f")]),
            UserMessage("user", [ToolResult("c", [TextContent("original")])]),
            UserMessage(
                "user",
                [
                    ToolResult("c", [TextContent("discarded merge")]),
                    TextContent("discarded text"),
                    ToolResult("orphan", []),
                ],
            ),
        ]
    )
    validate_history(result)
    assert len(result) == 2
    assert [p["text"] for p in result[1]["content"][0]["content"]] == ["original"]


def test_missing_call_ids_are_distinct_but_never_assigned_to_idless_results():
    result = normalize(
        [
            AssistantMessage("assistant", [ToolCall(None, "f"), ToolCall(None, "f")]),
            UserMessage("user", [ToolResult(None, [TextContent("first")], name="f")]),
            UserMessage("user", [ToolResult(None, [TextContent("second")], name="f")]),
        ]
    )
    validate_history(result)
    ids = [b["id"] for b in result[0]["content"]]
    assert len(set(ids)) == 2
    assert [result[i]["content"][0]["tool_use_id"] for i in (1, 2)] == ids
    assert all(result[i]["content"][0]["content"] == [] for i in (1, 2))


def test_external_result_references_stay_external_and_merge_repeated_content():
    result = normalize(
        [
            UserMessage("user", [ToolResult("server", [TextContent("first")])]),
            UserMessage("user", [ToolResult("server", [TextContent("first")])]),
        ],
        allow_external_calls=True,
    )
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert [part["text"] for part in result[0]["content"][0]["content"]] == ["first", "first"]
    validate_history(result, allow_external_calls=True)


def test_local_call_reusing_an_external_completed_id_is_discarded():
    result = normalize(
        [
            UserMessage("user", [ToolResult("server", [TextContent("first")])]),
            AssistantMessage("assistant", [ToolCall("server", "f", {"x": 1})]),
        ],
        allow_external_calls=True,
    )
    validate_history(result, allow_external_calls=True)
    assert [m["role"] for m in result] == ["user"]


def test_repairs_share_the_structural_message_and_content_scan():
    class Once(list):
        def __iter__(self):
            assert not getattr(self, "visited", False)
            self.visited = True
            return super().__iter__()

    messages = Once(
        [
            {
                "role": "user",
                "content": Once([{"type": "tool_result", "tool_use_id": "orphan", "content": []}]),
            }
        ]
    )
    normalizer = HistoryNormalizer()
    validate_messages(messages, history=normalizer)
    assert list(normalizer.finish(has_cached_history=True)) == []


@pytest.mark.parametrize("rounds", [1, 128])
def test_tool_pairing_reads_each_message_once_without_scanning_prior_calls(rounds):
    class SingleVisit(list):
        def __init__(self, values):
            super().__init__(values)
            self.reads = set()

        def __iter__(self):
            return (self[index] for index in range(len(self)))

        def __getitem__(self, index):
            assert isinstance(index, int), "pairing must not rescan a history slice"
            assert index not in self.reads, "pairing must not revisit a consumed message/block"
            self.reads.add(index)
            return super().__getitem__(index)

    class Associations(dict):
        lookups = 0

        def get(self, *args):
            self.lookups += 1
            return super().get(*args)

        def __contains__(self, key):
            self.lookups += 1
            return super().__contains__(key)

        def setdefault(self, *args):
            self.lookups += 1
            return super().setdefault(*args)

        def __iter__(self):
            raise AssertionError("pairing must resolve IDs without scanning prior calls")

        keys = items = values = __iter__

    messages = []
    contents = []
    expected = []
    for round_index in range(rounds):
        call_ids = [f"call_{round_index}_a", f"call_{round_index}_b"]
        calls = SingleVisit(
            [{"type": "tool_use", "id": key, "name": "f", "input": {}} for key in call_ids]
        )
        messages.append({"role": "assistant", "content": calls})
        contents.append(calls)
        # Results may arrive in either order; pairing must use IDs rather than positions.
        for key in call_ids if round_index % 2 == 0 else reversed(call_ids):
            result = SingleVisit(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": key,
                        "content": [{"type": "text", "text": key}],
                    }
                ]
            )
            messages.append({"role": "user", "content": result})
            contents.append(result)
            expected.append(key)
    messages = SingleVisit(messages)
    normalizer = HistoryNormalizer()
    normalizer.tools = Associations()
    validate_messages(messages, history=normalizer)
    repaired = list(normalizer.finish())
    validate_history(repaired)
    assert len(messages.reads) == len(messages)
    assert all(len(content.reads) == len(content) for content in contents)
    assert normalizer.tools.lookups <= 12 * rounds
    results = [
        block
        for message in repaired
        for block in message["content"]
        if block["type"] == "tool_result"
    ]
    assert [block["tool_use_id"] for block in results] == expected
    assert all(
        block["content"] == [{"type": "text", "text": block["tool_use_id"]}] for block in results
    )


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
@pytest.mark.parametrize("repair", ["missing_call", "missing_result", "duplicates"])
def test_all_native_protocol_pairs_repair_tools_before_sdk_validation(
    profile_factory, source, target, repair
):
    messages = history(source)
    if source == "ollama":
        messages[1]["tool_calls"][0]["id"] = "call_1"
        messages[2]["tool_call_id"] = "call_1"
    if repair == "missing_call":
        # Responses separates the assistant message from its call item.
        messages = [messages[0], messages[-1]]
    elif repair == "missing_result":
        messages = messages[:-1]
    else:
        messages.extend(deepcopy(messages[1:]))
    original = deepcopy(messages)
    codec = MyCodec()
    profile = profile_factory(
        target,
        supports_function_calling=config(True),
        max_tokens=parameter("max_tokens", 2048, protocol=target),
    )
    request = codec.prepare(
        {"messages": messages, "tools": tools(source), "stream": False},
        profile,
        input_format=source,
        client={},
    )["request"]
    validate_request(target, request)
    native = (
        request["input"]
        if target == "openai_responses"
        else (request["contents"] if target == "gemini" else request["messages"])
    )
    restored = normalize(codec.encode(messages=native, input_format=target)["messages"])
    validate_history(restored)
    blocks = [b for m in restored for b in m["content"]]
    calls = [b for b in blocks if b["type"] == "tool_use"]
    results = [b for b in blocks if b["type"] == "tool_result"]
    assert len(calls) == len(results) == (0 if repair == "missing_call" else 1)
    if calls:
        assert calls[0]["name"] == "weather"
    assert messages == original


@pytest.mark.parametrize("target", PROTOCOLS)
def test_all_orphan_results_fail_without_inventing_a_user_prompt(profile_factory, target):
    codec = MyCodec()
    with pytest.raises(MessageValidationError, match="no usable history"):
        codec.prepare(
            {
                "messages": [UserMessage("user", [ToolResult("c", [TextContent("result")])])],
                "tools": omit,
                "stream": False,
                "max_tokens": 2048,
            },
            profile_factory(target, supports_function_calling=config(True)),
            input_format="middleware",
            client={},
        )["request"]


@pytest.mark.parametrize("target", PROTOCOLS)
def test_orphan_message_is_removed_with_all_its_text_in_every_target(profile_factory, target):
    messages = [
        UserMessage("user", [TextContent("keep first")]),
        UserMessage(
            "user",
            [
                TextContent("discarded before"),
                ToolResult("orphan", [TextContent("discarded result")]),
                TextContent("discarded after"),
            ],
        ),
        UserMessage("user", [TextContent("keep last")]),
    ]
    before = deepcopy(messages)
    request = MyCodec().prepare(
        {"messages": messages, "tools": omit, "stream": False, "max_tokens": 2048},
        profile_factory(target, supports_function_calling=config(True)),
        input_format="middleware",
        client={},
    )["request"]
    validate_request(target, request)
    assert "discarded" not in str(request)
    assert "orphan" not in str(request)
    assert "keep first" in str(request) and "keep last" in str(request)
    assert messages == before


@pytest.mark.parametrize("source", PROTOCOLS)
@pytest.mark.parametrize("target", PROTOCOLS)
def test_native_idless_results_are_discarded_before_missing_results_are_padded(
    profile_factory, source, target
):
    messages = history(source)
    if source == "openai_chat":
        messages[-1].pop("tool_call_id")
        messages[-1]["content"] = "discarded result"
    elif source == "openai_responses":
        messages[-1].pop("call_id")
        messages[-1]["output"] = "discarded result"
    elif source == "anthropic":
        messages[-1]["content"][0].pop("tool_use_id")
        messages[-1]["content"][0]["content"] = "discarded result"
        messages[-1]["content"].append({"type": "text", "text": "discarded sibling"})
    elif source == "gemini":
        messages[-1]["parts"][0]["function_response"].pop("id")
        messages[-1]["parts"][0]["function_response"]["response"] = {"result": "discarded result"}
        messages[-1]["parts"].append({"text": "discarded sibling"})
    else:
        messages[-1].pop("tool_call_id")
        messages[-1]["content"] = "discarded result"
    before = deepcopy(messages)
    profile = profile_factory(
        target,
        supports_function_calling=config(True),
        max_tokens=parameter("max_tokens", 2048, protocol=target),
    )
    request = MyCodec().prepare(
        {"messages": messages, "tools": omit, "stream": False},
        profile,
        input_format=source,
        client={},
    )["request"]
    validate_request(target, request)
    assert "discarded" not in str(request)
    assert "Sunny" not in str(request)
    assert messages == before


def test_finish_does_not_scan_and_translation_consumes_repaired_slots_once(profile_factory):
    class Counted(list):
        visits = 0

        def __iter__(self):
            self.visits += 1
            return super().__iter__()

    first = Counted([{"type": "text", "text": "first"}])
    second = Counted([{"type": "text", "text": "first"}, {"type": "text", "text": "second"}])
    messages = Counted(
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "c", "name": "f", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c", "content": first}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c", "content": second}],
            },
        ]
    )
    normalizer = HistoryNormalizer()
    normalizer.messages = Counted()
    validate_messages(messages, history=normalizer)
    assert messages.visits == 1
    # Existing tag detection plus the existing nested structural-validation visit.
    assert first.visits == second.visits == 2
    assert normalizer.messages.visits == 0
    repaired = normalizer.finish()
    assert normalizer.messages.visits == 0
    translated = (
        MyCodec()
        .decoder("openai_chat")
        ._messages(repaired, profile_factory(supports_function_calling=config(True)))
    )
    assert normalizer.messages.visits == 1
    assert translated[-1]["content"] == "firstfirstsecond"


def test_result_parts_append_in_order_without_equality_checks():
    class Part(dict):
        def __eq__(self, other):
            raise AssertionError("merging result parts must not compare their content")

    first = [Part(type="text", text=str(index // 2)) for index in range(512)]
    second = [Part(type="text", text="0"), Part(type="text", text="tail")]
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "c", "name": "f", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c", "content": first}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c", "content": second}],
        },
    ]
    result = normalize(messages)
    validate_history(result)
    merged = result[1]["content"][0]["content"]
    assert len(result) == 2
    assert len(merged) == len(first) + len(second)
    assert all(
        actual is expected for actual, expected in zip(merged, [*first, *second], strict=True)
    )
    assert messages[1]["content"][0]["content"] is first
    assert messages[2]["content"][0]["content"] is second


def test_idless_result_is_discarded_even_with_a_unique_matching_tool_name():
    result = normalize(
        [
            AssistantMessage("assistant", [ToolCall("weather", "weather")]),
            UserMessage("user", [ToolResult(None, [TextContent("discarded")], name="weather")]),
        ]
    )
    validate_history(result)
    assert result[1]["content"][0]["tool_use_id"] == "weather"
    assert result[1]["content"][0]["content"] == []
    assert "discarded" not in str(result)
