# 构建与发行设计

本页说明构建来源、产物和自动化流程。安装与使用见[API 指南](../reference/api.md)，
隔离安装、元数据校验及复现步骤见[发行包验证](../testing/packaging.md)。

## 构建来源

在仓库根目录运行：

```sh
uv build
```

Hatchling 从 `src/my_llm/_version.py` 的 `__version__` 读取版本，
按 `pyproject.toml` 先构建源码包，再从源码包构建 wheel。保持单一版本来源，避免包元数据与运行时版本分离。
升级版本时修改 `_version.py`，再运行 `uv lock` 同步锁文件。

| 产物 | 内容与用途 |
| --- | --- |
| `dist/my_llm_py-<version>.tar.gz` | 源码、README、许可证、构建配置、文档、示例及测试，用于重建发行包。 |
| `dist/my_llm_py-<version>-py3-none-any.whl` | `my_llm` 运行时包与分发元数据；纯 Python，不绑定平台。 |

分发名是 `my-llm-py`，产物名规范化为 `my_llm_py`，导入名是 `my_llm`。
wheel 和源码包均携带 Apache-2.0 许可证；构建依赖不进入用户运行依赖。
`dist/` 被 Git 忽略。修改源码、文档或版本后需要重新构建，不能使用旧产物证明新工作树可发行。

## 自动化流程

[CI 配置](../../.github/workflows/ci.yml) 在分支 push 和 Pull Request 时触发。
测试任务配置了 Python 3.12、3.13、3.14，使用锁定依赖运行 pytest 和本地示例，
Python 3.12 另运行 Ruff 与格式检查。
构建任务依赖测试任务，随后构建、执行 `twine check --strict`、隔离安装 wheel，
并将发行文件保存为保留七天的 `python-distributions` artifact。

wheel 导入步骤通过 `MyLLM.generate` 和 `MyCodec.encode` 的公开行为检查安装结果，
不读取内部组件映射。旧接口引用的修复见[审查记录](../reviews/2026-11-23-documentation-alignment.md#后续-ci-修复)。
本地验证步骤见[发行包验证](../testing/packaging.md)。

配置依据为 [PyPA 的 pyproject.toml 指南](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)、
[Hatch 构建配置](https://hatch.pypa.io/latest/config/build/)及
[Hatch 版本配置](https://hatch.pypa.io/latest/version/)。
