# my-llm

通过配置统一调用 OpenAI Responses 和 Chat Completions 协议的异步 Python 库。
兼容服务使用相同的 OpenAI SDK，通过 `ProviderProfile` 和 `ModelProfile` 描述差异，
不内置厂商或模型清单。需要 Python 3.12+。

## 使用

```python
import asyncio
import os

from my_llm import ModelProfile, ProviderProfile, create


async def main():
    provider = ProviderProfile(
        api_type="responses",
        api_key=os.environ["OPENAI_API_KEY"],
        usage_mode="native",
    )
    model = ModelProfile(model=os.environ["LLM_MODEL"])
    message = await create(
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": "你好"}],
        reasoning_effort="none",
        stream=False,
    )
    print(message["content"])


asyncio.run(main())
```

`stream=True` 返回可关闭的异步流；底层 SDK 在两种模式下都执行流式请求：

```python
stream = await create(provider=provider, model=model, messages=messages, stream=True)
async with stream:
    async for chunk in stream:
        print(chunk.content, end="", flush=True)
        if chunk.tool_calls:
            print(chunk.tool_calls)  # 仅返回完整工具调用，由调用者执行本地函数
```

推理开启方式、强度映射、输出路径、消息名称与用量模式均可配置。Omit 可禁用
配置或删除某个字典键。流异常和缺少完成状态会明确报错。
Tools 可传 Responses 或 Chat Completions 格式，Codec 自动识别并按目标 API 转换。
Responses 结果保留原生 output items，支持工具调用后的历史回放。

完整参数、配置默认值及资源管理见
[API 文档](https://github.com/JJJJSTIYYYY/my-llm/blob/main/docs/reference/api.md)。
Anthropic、Gemini、Ollama 原生协议为预留扩展点，当前仅实现 OpenAI 两套协议。

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

测试使用真实 OpenAI SDK 和本地模拟 HTTP/SSE 传输，无需密钥或访问供应商服务。
`examples/local_demo.py` 离线演示两套 API、流式拼接和本地工具调用回合。
本地安装项目可运行 `uv sync`，或在构建后安装 `dist/` 中的 wheel。

构建说明见
[构建指南](https://github.com/JJJJSTIYYYY/my-llm/blob/main/docs/guides/building.md)。
发布名暂用 `my-llm`；本地构建不会发布到 PyPI。

版本号统一在 `src/my_llm/_version.py` 的 `__version__` 中维护，打包元数据与
`my_llm.__version__` 均使用该值。升级版本时只修改此文件，再运行 `uv lock` 更新锁文件。
