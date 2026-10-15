<div align="center">

# MyLLM

[![PyPI](https://img.shields.io/pypi/v/my-llm-py.svg)](https://pypi.org/project/my-llm-py/)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python)
[![License](https://img.shields.io/badge/license-Apache%202.0%20%2B%20MIT-blue.svg)](https://github.com/JJJJSTIYYYY/my-llm/blob/master/LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/tUCQthwK5)
[![QQ群](https://custom-icon-badges.demolab.com/badge/QQ群-639459172-00BFFF?style=flat&logo=tencent-qq)](https://qun.qq.com/universal-share/share?ac=1&authKey=ommoQrT2zhzHU%2FUxv8pfGCJbNifW%2BJyUAFBkNdzkHTPUxdxCnlgxm5aNgGslTmdE&busi_data=eyJncm91cENvZGUiOiI2Mzk0NTkxNzIiLCJ0b2tlbiI6Im9ZZkdNUWZnSVV1Y2REeUhKNnlTbWEwc05Bb093djRzUXdXNE55dklBVnlBQk9XbGNpS0ZXSDlzK3orSW1sQ3YiLCJ1aW4iOiIzMTI5NDI0NTcyIn0%3D&data=OGTchcr80RAQg8Z8_GZTdvBb7kZDeM9B3hHcNqLaAX2ZK_KYq260C4CubblEBT1bK5fP6zgtnCk2D8fIoph1ZQ&svctype=4&tempid=h5_group_info)

**通用的 LLM Provider 协议适配器**

</div>

通过配置统一调用 OpenAI（Responses / Chat Completions）、Anthropic Messages、
Gemini 和 Ollama 原生协议的异步 Python 库，四种协议均使用官方 SDK。
兼容服务使用相同的 OpenAI SDK，通过 `ProviderProfile` 和 `ModelProfile` 描述差异，
不内置厂商或模型清单。需要 Python 3.12+。

## 使用

```python
import asyncio
import os

from my_llm import ModelProfile, ProviderProfile, generate


async def main():
    provider = ProviderProfile(
        api_type="responses",
        api_key=os.environ["OPENAI_API_KEY"],
        usage_mode="native",
    )
    model = ModelProfile(model=os.environ["LLM_MODEL"])
    message = await generate(
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": "你好"}],
        reasoning_effort="none",
        stream=False,
    )
    print(message["content"])


asyncio.run(main())
```

重复调用时可复用 `MyLLM`，构造时显式传入模型配置：

```python
from my_llm import ModelProfile, MyLLM, ProviderProfile

llm = MyLLM(
    provider=ProviderProfile(api_type="responses"),
    model=ModelProfile(model="your-model-id", max_tokens=2048),
)
# 在异步函数中调用：
# message = await llm.generate(messages=[{"role": "user", "content": "你好"}])
```

`stream=True` 返回可关闭的异步流，调用时同样需要先 `await`：

```python
async def print_answer(llm):
    stream = await llm.generate(
        messages=[{"role": "user", "content": "你好"}], stream=True,
    )
    async with stream:
        async for chunk in stream:
            print(chunk.content, end="", flush=True)
            if chunk.tool_calls:
                print(chunk.tool_calls)  # 完整工具调用，由调用者执行本地函数
```

推理开启方式、强度映射、输出路径、消息名称与用量模式均可配置。Omit 可禁用
配置或删除某个字典键。流异常和缺少完成状态会明确报错。
Tools 可传 Responses 或 Chat Completions 格式，自动识别并按目标 API 转换。
Messages 同样支持两种格式：自动转换内容块、工具调用及工具结果，并允许混合历史。
Responses 结果保留原生 output items，支持工具调用后的历史回放。

调用参数、流式消费、工具回合及资源管理见
[API 使用指南](https://github.com/JJJJSTIYYYY/my-llm/blob/main/docs/reference/api.md)；
Profile 字段、默认值、继承规则及兼容服务适配见
[配置指南](https://github.com/JJJJSTIYYYY/my-llm/blob/main/docs/reference/configuration.md)。
通过 `ProviderProfile(protocol="anthropic" / "gemini" / "ollama")` 选择原生协议。
Anthropic 必须设置 `ModelProfile.max_tokens` 或调用时的 `max_tokens`。
原生协议的参数映射、推理配置和 SDK Client 类型见

## 本地开发与验证

```sh
uv sync --group dev --locked
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run python examples/local_demo.py
uv build
package_version=$(uv run python -c 'from my_llm import __version__; print(__version__)')
uv run twine check --strict "dist/my_llm-${package_version}-py3-none-any.whl" "dist/my_llm-${package_version}.tar.gz"
```