"""Build fresh distributions from a clean copy, then inspect their contents."""

import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

from my_llm import __version__


@pytest.mark.parametrize("version_override", [None, "9.8.7"])
def test_build_contains_runtime_modules_and_source_tests(tmp_path, version_override):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", ".gitignore"):
        shutil.copy2(root / name, source / name)
    for name in ("src", "docs/guides", "docs/reference", "tests", "examples"):
        shutil.copytree(
            root / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    expected_version = version_override or __version__
    if version_override is not None:
        (source / "src/my_llm/_version.py").write_text(
            f'__version__ = "{version_override}"\n', encoding="utf-8"
        )
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = source / f"dist/my_llm_py-{expected_version}-py3-none-any.whl"
    dist_info = f"my_llm_py-{expected_version}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "my_llm/invoke.py" in names
        assert "my_llm/_version.py" in names
        assert "my_llm/codec/openai/responses/decoding/__init__.py" in names
        for protocol in ("anthropic", "gemini", "ollama"):
            assert f"my_llm/codec/{protocol}/__init__.py" in names
        assert not any(name.startswith("my_llm/types/") for name in names)
        assert "my_llm/exceptions.py" in names
        assert "my_llm/stream/message_stream.py" in names
        assert "my_llm/stream/_state.py" in names
        assert "my_llm/codec/base/codec.py" in names
        assert "my_llm/codec/converter.py" in names
        assert "my_llm/py.typed" in names
        assert all(not name.endswith(".gitkeep") for name in names)
        assert all(name.startswith(("my_llm/", f"{dist_info}/")) for name in names)
        metadata = BytesParser().parsebytes(archive.read(f"{dist_info}/METADATA"))
        assert metadata["Name"] == "my-llm-py"
        assert metadata["Version"] == expected_version
        assert metadata["License-Expression"] == "Apache-2.0"
        assert metadata.get_all("License-File") == ["LICENSE"]
        assert archive.read(f"{dist_info}/licenses/LICENSE") == (root / "LICENSE").read_bytes()
        assert set(metadata.get_all("Project-URL")) == {
            "Repository, https://github.com/JJJJSTIYYYY/my-llm-py",
            "Issues, https://github.com/JJJJSTIYYYY/my-llm-py/issues",
        }
        assert metadata["Requires-Python"] == ">=3.12"
        assert metadata["Description-Content-Type"] == "text/markdown"
        for package in ("openai", "anthropic", "google-genai", "ollama"):
            assert any(
                requirement.startswith(package) for requirement in metadata.get_all("Requires-Dist")
            )
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, {str(wheel)!r}); import my_llm; from importlib.metadata import version; assert my_llm.__version__ == version('my-llm-py') == {expected_version!r}; assert my_llm.__file__.startswith({str(wheel)!r}); assert callable(my_llm.generate); assert callable(my_llm.MyLLM.generate); assert not hasattr(my_llm, 'create'); assert not hasattr(my_llm, 'Invoke')",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    with tarfile.open(source / f"dist/my_llm_py-{expected_version}.tar.gz") as archive:
        names = archive.getnames()
        for name in (
            "src/my_llm/invoke.py",
            "src/my_llm/_version.py",
            "src/my_llm/py.typed",
            "tests/integration/test_sdk_transport.py",
            "docs/reference/api.md",
            "docs/reference/configuration.md",
            "examples/local_demo.py",
            "LICENSE",
        ):
            assert f"my_llm_py-{expected_version}/{name}" in names
        metadata = BytesParser().parsebytes(
            archive.extractfile(f"my_llm_py-{expected_version}/PKG-INFO").read()
        )
        assert metadata["Name"] == "my-llm-py"
        assert metadata["Version"] == expected_version
        assert metadata["License-Expression"] == "Apache-2.0"
        assert (
            archive.extractfile(f"my_llm_py-{expected_version}/LICENSE").read()
            == (root / "LICENSE").read_bytes()
        )
        assert all(not name.endswith(".gitkeep") for name in names)
        assert all("设计文档" not in name for name in names)
