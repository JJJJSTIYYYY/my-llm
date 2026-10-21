"""Optional SDK isolation in fresh interpreters, independent of test collection."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("installed", [None, "openai", "anthropic", "google.genai", "ollama"])
def test_only_installed_sdks_are_required(installed):
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        installed = sys.argv[1]
        sdks = {"openai", "anthropic", "google.genai", "ollama"}
        blocked = sdks - {installed}

        class MissingSDK(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                for sdk in blocked:
                    if fullname == sdk or fullname.startswith(sdk + "."):
                        raise ModuleNotFoundError(fullname, name=sdk)

        sys.meta_path.insert(0, MissingSDK())

        import my_llm
        import my_llm.codec

        protocols = {
            "openai": ("openai_chat", "openai_responses"),
            "anthropic": ("anthropic",),
            "google.genai": ("gemini",),
            "ollama": ("ollama",),
        }
        exports = {
            "openai": "ChatCompletionsCodec",
            "anthropic": "AnthropicCodec",
            "google.genai": "GeminiCodec",
            "ollama": "OllamaCodec",
        }
        converter = my_llm.MyCodec()
        assert set(converter.codecs) == set(protocols.get(installed, ()))
        for sdk, names in protocols.items():
            for protocol in names:
                if sdk == installed:
                    profile = my_llm.resolve_profiles(
                        my_llm.ProviderProfile(protocol=protocol),
                        my_llm.ModelProfile(model="demo"),
                    )
                    assert profile.protocol == protocol
                    assert converter.decoder(protocol)
                    assert getattr(my_llm.codec, exports[sdk])
                    message = (
                        {"role": "user", "parts": [{"text": "hello"}]}
                        if sdk == "google.genai" else
                        {"role": "user", "content": "hello"}
                    )
                    if protocol == "openai_responses":
                        continue
                    result = converter.convert(
                        messages=[message], input_format=protocol, target_format=protocol
                    )
                    assert result["messages"]
                    continue
                extra = "google-genai" if sdk == "google.genai" else sdk
                for action in (
                    lambda: converter.decoder(protocol),
                    lambda: my_llm.resolve_profiles(
                        my_llm.ProviderProfile(protocol=protocol),
                        my_llm.ModelProfile(model="demo"),
                    ),
                    lambda: getattr(my_llm.codec, exports[sdk]),
                ):
                    try:
                        action()
                    except ModuleNotFoundError as exc:
                        assert f"my-llm-py[{extra}]" in str(exc), str(exc)
                    else:
                        raise AssertionError("missing SDK did not fail")
        assert not any(
            name == sdk or name.startswith(sdk + ".")
            for name in sys.modules for sdk in blocked
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, installed or ""], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_broken_sdk_dependency_is_not_silently_ignored():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BrokenSDK(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "openai":
                    raise ModuleNotFoundError("broken SDK dependency", name="missing_transitive")

        sys.meta_path.insert(0, BrokenSDK())
        try:
            import my_llm
        except ModuleNotFoundError as exc:
            assert exc.name == "missing_transitive"
        else:
            raise AssertionError("broken SDK was silently ignored")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
