# my-llm-py 构建与验证

使用接口见 [API 使用指南](../reference/api.md)，配置字段见
[配置指南](../reference/configuration.md)。

## 构建

使用 Python 3.12 或更高版本及 uv，在仓库根目录运行：

```sh
uv build
```

Hatchling 从 `src/my_llm/_version.py` 的 `__version__` 读取版本号，根据
`pyproject.toml` 构建源码包，再从源码包构建 wheel。生成文件如下：

- `dist/my_llm_py-<version>.tar.gz`：源码包。
- `dist/my_llm_py-<version>-py3-none-any.whl`：不依赖平台的纯 Python wheel。

升级版本时只修改 `_version.py`，再运行 `uv lock` 更新锁文件。通过
`my_llm.__version__` 也可读取相同版本；`pyproject.toml` 使用动态版本元数据。

构建依赖仅用于构建环境，不作为用户安装包时的运行依赖。
wheel 包含 `my_llm` 包和分发元数据，排除目录占位文件。
源码包还包含 README、Apache 2.0 许可证、构建配置、使用文档、本地示例和测试。
分发名为 `my-llm-py`，构建产物名称规范化为 `my_llm_py`，导入名为 `my_llm`。
`dist/` 已被 Git 忽略。

## 安装验证

在单独的临时虚拟环境中安装本地 wheel，避免覆盖已有开发环境：

```sh
uv venv /tmp/my-llm-py-package-check --python 3.12
package_version=$(uv run python -c 'from my_llm import __version__; print(__version__)')
uv pip install --python /tmp/my-llm-py-package-check/bin/python "dist/my_llm_py-${package_version}-py3-none-any.whl"
/tmp/my-llm-py-package-check/bin/python -I -c 'import my_llm; from importlib.metadata import version; assert my_llm.__version__ == version("my-llm-py"); print(my_llm.__version__); print(my_llm.__file__)'
```

输出应为 `_version.py` 中声明的版本，且模块路径位于临时虚拟环境的 `site-packages`。
`-I` 使用 Python 隔离模式，避免仓库路径或 `PYTHONPATH` 干扰导入验证。

检查 PyPI 分发元数据和 README 渲染格式：

```sh
uv run twine check --strict "dist/my_llm_py-${package_version}-py3-none-any.whl" "dist/my_llm_py-${package_version}.tar.gz"
```

修改源文件、文档或版本后重新运行 `uv build`。项目使用 Apache-2.0，
wheel 和源码包均携带 LICENSE，分发元数据声明同一许可证。

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
[Hatch 构建配置文档](https://hatch.pypa.io/latest/config/build/)，版本来源参见
[Hatch 版本配置](https://hatch.pypa.io/latest/version/)。

## GitHub 自动测试（CI）

[ci.yml](https://github.com/JJJJSTIYYYY/my-llm/blob/main/.github/workflows/ci.yml) 在分支 push 和 Pull Request 时运行。
使用锁定的依赖在 Ubuntu 上测试 Python 3.12、3.13、3.14；Python 3.12
额外检查 Ruff 规则及格式。所有版本均运行完整 pytest 和本地示例，不需要模型 API Key。

测试通过后，CI 从源码包构建 wheel，执行 `twine check --strict`，在隔离环境安装并
导入 wheel，最后保存 wheel 和源码包为 `python-distributions` artifact，保留 7 天。
