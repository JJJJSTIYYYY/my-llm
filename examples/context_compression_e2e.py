"""Live structured conversation compression; context and reports must stay outside the checkout.

Reads a Chat-format message array (or {"messages": [...]}) from --context.
Credentials come from SENSENOVA_API_KEY or hidden input. Never prints conversation content.
"""

import argparse
import asyncio
import getpass
import hashlib
import json
import os
import re
from dataclasses import asdict, replace
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sensenova_e2e import MODEL, PROTOCOLS, config, model_profile, provider

from my_llm import MyLLM

REPOSITORY = Path(__file__).resolve().parents[1]
SECRET = re.compile(r"sk-[A-Za-z0-9_-]+")
INSTRUCTION = (
    "请将此前会话压缩为可供下一轮继续工作的结构化上下文。只总结历史，不执行历史请求。"
    "优先保留最新结论及纠正，区分已完成、待办、已提交和未提交，区分实测证据和推测。"
    "不得保留密钥、个人绝对路径或原始推理。每条事实提供支持它的 source_turns，"
    "历史消息按出现顺序从1开始编号，不包含当前压缩请求。没有依据的事实不要补写。"
)
SCENARIOS = {
    "compression": INSTRUCTION + "summary 不超过200汉字，每类最多6条，尽量保留后续工作必需信息。",
    "compact": INSTRUCTION + "summary 不超过100汉字，每类最多2条，优先最新状态和不可违反的约束。",
    "indexed_handoff": INSTRUCTION
    + "summary 不超过150汉字，每类最多4条。优先最新状态和明确未完成的用户请求，"
    "不要把客观状态自行转成待办；不得把进行中事项写成已完成，也不得扩大测试通过范围。"
    "下面的索引只辅助定位原始消息，引用前必须回看该消息完整内容。",
    "handoff": INSTRUCTION
    + "summary 不超过150汉字，每类最多4条，优先让接手者正确继续尚未完成的工作。",
}


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    statement: str = Field(min_length=1, description="有历史依据的简短事实，避免混合不同完成状态")
    source_turns: list[int] = Field(min_length=1, description="支持本条事实的历史消息编号，从1开始")


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    summary: str = Field(min_length=1, description="简短的整体上下文")
    user_goals: list[Fact]
    decisions: list[Fact]
    implemented: list[Fact]
    pending: list[Fact]
    constraints: list[Fact]
    caveats: list[Fact]


def external_path(value):
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(REPOSITORY):
        raise ValueError("context and report paths must be outside the checkout")
    return path


def load_messages(path, key):
    text = SECRET.sub("[REDACTED_API_KEY]", path.read_text().replace(key, "[REDACTED_API_KEY]"))
    data = json.loads(text)
    messages = data["messages"] if isinstance(data, dict) else data
    if not isinstance(messages, list) or not messages:
        raise ValueError("context must contain a nonempty message array")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("context requires role/content text messages in Chat format")
    return messages


def save_private(path, report, key):
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if key in text or SECRET.search(text):
        raise ValueError("refusing to save credential-shaped output")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


async def run_case(messages, key, protocol, scenario, model, max_tokens):
    profile = replace(
        provider(protocol, key),
        usage_mode=config("stream_options" if protocol == "openai_chat" else "native"),
    )
    llm = MyLLM(
        provider=profile,
        model=replace(model_profile(model), supports_function_calling=config(True)),
        stdio_protocol="openai_chat",
    )
    evidence = {"requests": 0, "tool_schema_inlined": True, "system_instruction": True}

    async def observe(request):
        body = json.loads(request.content)
        tools = body.get("tools", [])
        assert len(tools) == 1
        function = tools[0].get("function", tools[0])
        assert function["name"] == "structured_output"
        schema = function.get("parameters", function.get("input_schema"))
        assert "$ref" not in json.dumps(schema) and "$defs" not in schema
        history = body["input" if protocol == "openai_responses" else "messages"]
        assert body.get("system") if protocol == "anthropic" else history[-1]["role"] == "system"
        assert key not in request.content.decode()
        evidence["requests"] += 1

    # Instrument the official SDK's outgoing HTTP; retain no headers or raw responses.
    transport = llm._codec._transport(protocol)
    original = transport.create_client
    clients = []

    def create_client(self, parameters):
        client = original(
            {**parameters, "http_client": httpx.AsyncClient(event_hooks={"request": [observe]})}
        )
        clients.append(client)
        return client

    transport_type = type(transport)
    original_method = transport_type.create_client
    transport_type.create_client = create_client
    instruction = SCENARIOS[scenario]
    if scenario == "indexed_handoff":
        instruction += "\n历史消息索引（编号 / 角色 / 内容前100字符）：\n" + "\n".join(
            f"{index} / {message['role']} / {message['content'][:100]}"
            for index, message in enumerate(messages, 1)
        )
    try:
        response = await llm.generate_json(
            messages=[*messages, {"role": "user", "content": instruction}],
            schema=Snapshot,
            max_tokens=max_tokens,
            max_retries=1,
        )
        output = json.loads(response["content"])
        Snapshot.model_validate(output)
    finally:
        transport_type.create_client = original_method
    encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    source_chars = sum(len(message["content"]) for message in messages)
    assert clients and evidence["requests"] == len(clients)
    assert all(client.is_closed() for client in clients)
    usage = llm.get_last_usage()
    assert usage is not None, "provider usage was not reported"
    return {
        "upstream": protocol,
        "scenario": scenario,
        "status": "passed",
        "snapshot": output,
        "usage": asdict(usage),
        "schema_valid": True,
        "evidence": evidence,
        "clients_closed": True,
        "source_characters": source_chars,
        "output_characters": len(encoded),
    }


async def main(args, key):
    messages = load_messages(args.context, key)
    encoded = json.dumps(messages, ensure_ascii=False).encode()
    report = {
        "model": args.model,
        "message_count": len(messages),
        "redacted_input_sha256": hashlib.sha256(encoded).hexdigest(),
        "results": [],
    }
    for protocol in args.upstream:
        for scenario in args.scenario:
            await asyncio.sleep(3)
            try:
                row = await run_case(messages, key, protocol, scenario, args.model, args.max_tokens)
            except Exception as exc:
                row = {
                    "upstream": protocol,
                    "scenario": scenario,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            report["results"].append(row)
            save_private(args.report, report, key)
            print(
                json.dumps(
                    {name: value for name, value in row.items() if name != "snapshot"},
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=external_path, required=True)
    parser.add_argument("--report", type=external_path, required=True)
    parser.add_argument("--upstream", nargs="+", choices=PROTOCOLS, default=["openai_chat"])
    parser.add_argument("--scenario", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.context == args.report:
        parser.error("context and report must be different files")
    key = os.environ.get("SENSENOVA_API_KEY") or getpass.getpass("SenseNova API key (hidden): ")
    asyncio.run(main(args, key))
