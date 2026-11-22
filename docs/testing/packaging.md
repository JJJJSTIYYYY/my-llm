# 发行包验证

本页复现发行包的构建、内容与安装验证。构建配置的职责和发布流程见
[构建开发指南](../guides/building.md)。以下命令从仓库根目录执行。

## 构建与内容检查

```sh
uv sync --locked --group dev
uv run --no-sync pytest -q tests/packaging
uv build
package_version=$(uv run --no-sync python -c 'from my_llm import __version__; print(__version__)')
uv run --no-sync twine check --strict "dist/my_llm_py-${package_version}-py3-none-any.whl" "dist/my_llm_py-${package_version}.tar.gz"
```

`tests/packaging` 在临时源码副本中构建，检查运行时模块、许可证、分发元数据和源码包中的
文档、示例、测试，避免把旧 `dist/` 产物当作当前构建结果。`twine check` 检查元数据和 README
渲染格式，不验证模型调用。修改源码、文档或版本后需要重新构建。

## 隔离安装与公开接口

创建独立临时环境，避免覆盖开发环境。先安装核心包，检查缺失 SDK 的公开错误，
再安装全部 SDK 并检查公开业务入口：

```sh
uv venv /tmp/my-llm-py-package-check --python 3.12
uv pip install --python /tmp/my-llm-py-package-check/bin/python "dist/my_llm_py-${package_version}-py3-none-any.whl"
/tmp/my-llm-py-package-check/bin/python -I - <<'PYTHON'
from my_llm import MyCodec

try:
    MyCodec().encode(messages=[], input_format="openai_chat")
except ModuleNotFoundError as exc:
    assert exc.name == "openai"
    assert "my-llm-py[openai]" in str(exc)
else:
    raise AssertionError("the core-only environment unexpectedly has the OpenAI SDK")
PYTHON

uv pip install --python /tmp/my-llm-py-package-check/bin/python "dist/my_llm_py-${package_version}-py3-none-any.whl[all]"
/tmp/my-llm-py-package-check/bin/python -I - <<'PYTHON'
from importlib.metadata import version

import my_llm

assert my_llm.__version__ == version("my-llm-py")
assert callable(my_llm.MyLLM.generate)
assert all(callable(getattr(my_llm.MyCodec, name)) for name in ("encode", "decode", "convert"))
codec = my_llm.MyCodec()
for protocol in ("openai_chat", "openai_responses", "anthropic"):
    assert codec.encode(messages=[], input_format=protocol).messages == []
print(my_llm.__version__)
print(my_llm.__file__)
PYTHON
```

核心包验证需要没有预装 SDK 的临时环境；重复使用此前装过 `[all]` 的环境无法证明这一点。
`-I` 排除仓库路径与 `PYTHONPATH` 干扰，输出模块路径应位于临时环境的 `site-packages`。
可选依赖的独立回归还可运行 `tests/integration/test_optional_sdks.py`。
这些命令验证安装与导入，不调用远程模型。

## 当前 CI 的范围

[ci.yml](../../.github/workflows/ci.yml) 在 push 和 Pull Request 时运行 Python 3.12、3.13、3.14
的 pytest 与本地示例，3.12 额外运行 Ruff。构建任务等待测试通过后构建、运行严格 Twine 检查、
在隔离环境先安装核心包再安装 `[all]`，并保留发行产物 7 天。

CI 已改用当前公开接口：核心安装验证缺少 SDK 时的明确错误，安装 `[all]` 后验证
`MyLLM.generate` 可调用，以及三协议的 `MyCodec.encode` 可用。
旧接口引用的根因与修复见[审查记录](../reviews/2026-11-23-documentation-alignment.md#后续-ci-修复)。
