"""Keep native policies with their encoder/decoder and preserve public exports."""

import ast
from pathlib import Path

from tests.helpers.protocols import PROTOCOLS

ROOT = Path(__file__).resolve().parents[2] / "src/my_llm"
OWNERS = ("openai/chat", "openai/responses", "anthropic", "gemini", "ollama")


def native_owner(path):
    for owner in OWNERS:
        if path.is_relative_to(ROOT / "codec" / owner / "encoding") or path.is_relative_to(
            ROOT / "codec" / owner / "decoding"
        ):
            return f"my_llm.codec.{owner.replace('/', '.')}"
    return None


def test_native_conditions_and_sdk_clients_stay_inside_native_owners():
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        if native_owner(path):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
                assert not any(
                    isinstance(child, ast.Constant) and child.value in PROTOCOLS
                    for child in ast.walk(node.test)
                    if isinstance(child, ast.Constant) and isinstance(child.value, str)
                ), path
            if isinstance(node, ast.ImportFrom):
                assert not any(
                    alias.name in {"AsyncOpenAI", "AsyncAnthropic", "AsyncClient", "genai"}
                    for alias in node.names
                ), path


def test_native_implementations_do_not_depend_on_other_native_implementations():
    for path in ROOT.rglob("*.py"):
        owner = native_owner(path)
        if owner is None:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("my_llm.codec."):
                assert node.module.startswith(
                    (owner + ".", "my_llm.codec.shared.", "my_llm.codec.base")
                ), (path, node.module)


def test_encoders_and_decoders_do_not_depend_on_the_opposite_direction():
    for path in (ROOT / "codec").rglob("*.py"):
        owner = native_owner(path)
        if owner is None:
            continue
        direction = "encoding" if "encoding" in path.parts else "decoding"
        opposite = "decoding" if direction == "encoding" else "encoding"
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            assert not any(
                module.startswith((owner + "." + opposite, owner + ".codec"))
                or module == "my_llm.codec.converter"
                for module in modules
            ), path


def test_package_initializers_only_export_interfaces():
    for path in ROOT.rglob("__init__.py"):
        tree = ast.parse(path.read_text())
        assert all(
            isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.Expr))
            for node in tree.body
        ), path
        assert not any(isinstance(node, (ast.Call, ast.IfExp)) for node in ast.walk(tree)), path


def test_runtime_implementations_import_definitions_directly():
    for path in ROOT.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("my_llm"):
                continue
            target = ROOT.parent.joinpath(*node.module.split("."))
            if target.is_dir():
                for alias in node.names:
                    assert (target / f"{alias.name}.py").is_file() or (
                        target / alias.name / "__init__.py"
                    ).is_file(), (path, node.module, alias.name)


def test_shared_test_helpers_do_not_depend_on_test_collection_order():
    tests = ROOT.parents[1] / "tests"
    for path in tests.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                assert all(
                    part != "conftest" and not part.startswith("test_")
                    for part in module.split(".")
                ), (path, module)
