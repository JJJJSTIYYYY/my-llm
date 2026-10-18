"""Keep conversion and SDK execution independent within each native codec."""

import ast
from pathlib import Path

import pytest

from tests.helpers.protocols import PROTOCOLS

ROOT = Path(__file__).resolve().parents[2] / "src/my_llm"
OWNERS = ("openai/chat", "openai/responses", "anthropic", "gemini", "ollama")
SDK_PACKAGES = ("openai", "anthropic", "google", "ollama")
SDK_CLIENTS = {
    "AsyncOpenAI",
    "OpenAI",
    "AsyncAnthropic",
    "Anthropic",
    "AsyncClient",
    "Client",
    "genai",
}


def imported_modules(node):
    if isinstance(node, ast.ImportFrom):
        return [node.module or ""]
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return []


def assert_no_protocol_dispatch(tree, path):
    for node in ast.walk(tree):
        conditions = []
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
            conditions.append(node.test)
        elif isinstance(node, ast.match_case):
            conditions.append(node.pattern)
            if node.guard is not None:
                conditions.append(node.guard)
        for condition in conditions:
            assert not any(
                isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and child.value in PROTOCOLS
                for child in ast.walk(condition)
            ), path


def assert_no_sdk_clients(tree, path):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(
                module.split(".")[0] in SDK_PACKAGES for module in imported_modules(node)
            ), path
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in SDK_PACKAGES:
            assert not any(alias.name in SDK_CLIENTS for alias in node.names), path


def native_owner(path):
    for owner in OWNERS:
        if (
            path.is_relative_to(ROOT / "codec" / owner / "encoding")
            or path.is_relative_to(ROOT / "codec" / owner / "decoding")
            or path == ROOT / "codec" / owner / "transport.py"
        ):
            return f"my_llm.codec.{owner.replace('/', '.')}"
    return None


def test_native_conditions_and_sdk_clients_stay_inside_native_owners():
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        owner = native_owner(path)
        if owner is None:
            assert_no_protocol_dispatch(tree, path)
        if owner is None or path.name != "transport.py":
            assert_no_sdk_clients(tree, path)


@pytest.mark.parametrize(
    "source",
    [
        'if protocol == "anthropic": pass',
        'match protocol:\n    case "anthropic": pass',
        'match protocol:\n    case value if value == "anthropic": pass',
    ],
)
def test_generic_dispatch_check_rejects_protocol_branches(source):
    with pytest.raises(AssertionError):
        assert_no_protocol_dispatch(ast.parse(source), Path("shared.py"))


@pytest.mark.parametrize(
    "source",
    [
        "import openai as oa\nclient = oa.AsyncOpenAI()",
        "from anthropic import AsyncAnthropic as Client",
        "from google import genai",
        "import google.genai as sdk",
    ],
)
def test_sdk_boundary_rejects_import_aliases(source):
    with pytest.raises(AssertionError):
        assert_no_sdk_clients(ast.parse(source), Path("shared.py"))


def test_native_implementations_do_not_depend_on_other_native_implementations():
    for path in ROOT.rglob("*.py"):
        owner = native_owner(path)
        if owner is None:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            for module in imported_modules(node):
                if module.startswith("my_llm.codec."):
                    assert module.startswith(
                        (owner + ".", "my_llm.codec.shared.", "my_llm.codec.base")
                    ), (path, module)


def test_encoders_and_decoders_do_not_depend_on_the_opposite_direction():
    for path in (ROOT / "codec").rglob("*.py"):
        owner = native_owner(path)
        if owner is None or path.name == "transport.py":
            continue
        direction = "encoding" if "encoding" in path.parts else "decoding"
        opposite = "decoding" if direction == "encoding" else "encoding"
        for node in ast.walk(ast.parse(path.read_text())):
            assert not any(
                module.startswith((owner + "." + opposite, owner + ".codec"))
                or module == owner + ".transport"
                or module == "my_llm.codec.converter"
                for module in imported_modules(node)
            ), path


def test_transports_do_not_import_conversion_implementations():
    for owner in OWNERS:
        path = ROOT / "codec" / owner / "transport.py"
        for node in ast.walk(ast.parse(path.read_text())):
            for module in imported_modules(node):
                assert not module.startswith(
                    (
                        f"my_llm.codec.{owner.replace('/', '.')}.encoding",
                        f"my_llm.codec.{owner.replace('/', '.')}.decoding",
                    )
                ), (path, module)


def test_profiles_do_not_depend_on_codec_or_execution_implementations():
    for path in (ROOT / "profile").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            for module in imported_modules(node):
                assert not module.startswith(("my_llm.codec", "my_llm.invoke", "my_llm.stream")), (
                    path,
                    module,
                )


def test_invoke_only_accesses_its_own_private_attributes():
    path = ROOT / "invoke.py"
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            assert isinstance(node.value, ast.Name) and node.value.id == "self", (
                path,
                node.lineno,
                node.attr,
            )


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
            for module in imported_modules(node):
                assert all(
                    part != "conftest" and not part.startswith("test_")
                    for part in module.split(".")
                ), (path, module)
