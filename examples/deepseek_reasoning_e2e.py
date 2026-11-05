"""Live DeepSeek input/repair matrix; opt-in, billable API requests.

Run: uv run python examples/deepseek_reasoning_e2e.py --report /tmp/deepseek-results.json
Use --validate for offline preparation. Keys are read without echo and never saved.
The matrix covers Chat, Responses, Anthropic and typed middleware inputs, with
collected and streaming completions. Raw controls use independent, unseen IDs.

Official references:
https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/
https://api-docs.deepseek.com/zh-cn/guides/responses_api/
https://api-docs.deepseek.com/zh-cn/guides/anthropic_api/
https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
"""

import argparse
import asyncio
import getpass
import logging
import resource
from pathlib import Path

from e2e_inputs import INPUT_CASES, LIVE_CASES, PROTOCOLS, STDIO_PROTOCOLS
from sensenova_e2e import BASE_CASES, config, run

from my_llm import Attribute, InputParameter, ModelProfile, OutputParameter, ProviderProfile


def provider(protocol, key, *, reasoning=False):
    """Declare DeepSeek's documented connection and generation options in Profile."""
    enabled = "enabled" if reasoning else "disabled"
    effort = "low" if reasoning else "none"
    body = {
        "openai_chat": {"thinking": {"type": enabled}, "reasoning_effort": "low"},
        "openai_responses": {"reasoning": {"effort": effort}},
        "anthropic": {
            "thinking": {"type": "enabled", "budget_tokens": 1024}
            if reasoning
            else {"type": "disabled"},
            "output_config": {"effort": "low"},
        },
    }[protocol]
    overrides = {}
    if protocol == "openai_chat":
        overrides["reasoning_effort"] = InputParameter(path="$.reasoning_effort")
        overrides["reasoning_content"] = OutputParameter(
            path="$.choices[0].delta.reasoning_content"
        )
    elif protocol == "openai_responses":
        overrides["reasoning_content"] = OutputParameter(
            path="$.delta", event_type="response.reasoning_text.delta"
        )
    else:
        overrides["reasoning_effort"] = InputParameter(path="$.output_config.effort")
    return ProviderProfile(
        protocol=config(protocol),
        base_url=config(
            "https://api.deepseek.com" + ("/anthropic" if protocol == "anthropic" else "/v1")
        ),
        api_key=InputParameter(path="$.api_key", default_value=key),
        timeout=config(90.0),
        max_retries=config(0),
        max_tokens=InputParameter(
            path="$.max_output_tokens" if protocol == "openai_responses" else "$.max_tokens",
            default_value=2048 if reasoning else 1024,
        ),
        extra_body=config(body),
        **overrides,
    )


def model_profile(model_id, case=None):
    return ModelProfile(
        model=Attribute(default_value=model_id),
        supports_function_calling=config(True),
        supports_vision=config(case == "tool_result_images"),
        should_feedback_reasoning_item=config("when_tool_calls"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--upstream", nargs="+", choices=PROTOCOLS)
    parser.add_argument("--stdio", nargs="+", choices=STDIO_PROTOCOLS)
    parser.add_argument("--case", nargs="+", choices=(*BASE_CASES, *LIVE_CASES))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--request-interval", type=float, default=0.0, metavar="SECONDS")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--validate", action="store_true", help="prepare cases without network or a key"
    )
    parser.add_argument("--semantic-loss-protocol", choices=("raise", "ignore"), default="ignore")
    args = parser.parse_args()
    args.case = args.case or (INPUT_CASES if args.validate else LIVE_CASES)
    if args.request_interval < 0:
        parser.error("--request-interval must be nonnegative")
    if args.retry_failed and (args.report is None or not args.report.is_file()):
        parser.error("--retry-failed requires an existing --report file")
    if args.validate and any(case not in INPUT_CASES for case in args.case):
        parser.error("--validate supports input cases only")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    logging.disable(logging.CRITICAL)
    key = "offline-validation-only" if args.validate else getpass.getpass("DeepSeek API key: ")
    if not key.strip():
        parser.error("an API key is required")
    raise SystemExit(
        asyncio.run(
            run(
                args,
                key,
                provider_factory=provider,
                model_factory=model_profile,
                endpoint="https://api.deepseek.com",
            )
        )
    )


if __name__ == "__main__":
    main()
