"""Build fresh distributions from a clean copy, then inspect their contents."""

import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def test_build_contains_runtime_types_and_source_tests(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", ".gitignore"):
        shutil.copy2(root / name, source / name)
    for name in ("src", "docs/guides", "docs/reference", "tests", "examples"):
        shutil.copytree(
            root / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = source / "dist/my_llm-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "my_llm/invoke.py" in names
        assert "my_llm/codec/responses.py" in names
        assert "my_llm/types/messages.py" in names
        assert "my_llm/py.typed" in names
        assert all(not name.endswith(".gitkeep") for name in names)
        assert all(name.startswith(("my_llm/", "my_llm-0.1.0.dist-info/")) for name in names)
        metadata = BytesParser().parsebytes(archive.read("my_llm-0.1.0.dist-info/METADATA"))
        assert metadata["Name"] == "my-llm"
        assert metadata["Version"] == "0.1.0"
        assert metadata["Requires-Python"] == ">=3.12"
        assert metadata["Description-Content-Type"] == "text/markdown"
        assert any(
            requirement.startswith("openai") for requirement in metadata.get_all("Requires-Dist")
        )
    with tarfile.open(source / "dist/my_llm-0.1.0.tar.gz") as archive:
        names = archive.getnames()
        for name in (
            "src/my_llm/invoke.py",
            "src/my_llm/py.typed",
            "tests/integration/test_sdk_transport.py",
            "docs/reference/api.md",
            "examples/local_demo.py",
        ):
            assert f"my_llm-0.1.0/{name}" in names
        assert all(not name.endswith(".gitkeep") for name in names)
        assert all("设计文档" not in name for name in names)
