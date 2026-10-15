## 构建

使用 Python 3.12 或更高版本及 uv，在仓库根目录运行：

```sh
uv build
```

Hatchling 根据 `pyproject.toml` 构建源码包，再从源码包构建 wheel。当前版本生成：

- `dist/my_llm-0.1.0.tar.gz`：源码包。
- `dist/my_llm-0.1.0-py3-none-any.whl`：不依赖平台的纯 Python wheel。

构建依赖仅用于构建环境，不作为用户安装包时的运行依赖。
wheel 包含 `my_llm` 包和分发元数据，排除目录占位文件。
源码包还包含 README、构建配置、使用文档、本地示例和测试。
`dist/` 已被 Git 忽略。

## 安装验证

在单独的临时虚拟环境中安装本地 wheel，避免覆盖已有开发环境：

```sh
uv venv /tmp/my-llm-package-check --python 3.12
uv pip install --python /tmp/my-llm-package-check/bin/python dist/my_llm-0.1.0-py3-none-any.whl
/tmp/my-llm-package-check/bin/python -I -c 'import my_llm; from importlib.metadata import version; print(version("my-llm")); print(my_llm.__file__)'
```

输出应为 `0.1.0`，且模块路径位于临时虚拟环境的 `site-packages`。
`-I` 使用 Python 隔离模式，避免仓库路径或 `PYTHONPATH` 干扰导入验证。

检查 PyPI 分发元数据和 README 渲染格式：

```sh
uv run twine check --strict dist/my_llm-0.1.0-py3-none-any.whl dist/my_llm-0.1.0.tar.gz
```

修改源文件、文档或版本后重新运行 `uv build`。发布前确定许可证、确认 PyPI
名称及版本可用性；本指南只覆盖本地构建与验证。

## 本地测试

```sh
uv sync --group dev --locked
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run python examples/local_demo.py
```

测试使用真实 SDK、固定事件和本地模拟传输，覆盖配置、消息转换、完整工具调用、
用量、流失败和连接释放，不访问真实模型服务。

配置依据：
[PyPA 的 pyproject.toml 指南](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)和
[Hatch 构建配置文档](https://hatch.pypa.io/latest/config/build/)。
