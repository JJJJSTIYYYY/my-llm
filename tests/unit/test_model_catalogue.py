import pytest

from my_llm import Attribute, Config, ConfigurationError, ModelProfile, ProviderProfile, generate
from my_llm.profile.resolution import resolve_profiles
from tests.helpers.profiles import parameter
from tests.helpers.protocols import chat_event, response_completed
from tests.helpers.transport import FakeClient


def test_context_window_default():
    assert (
        ModelProfile(model=Attribute(default_value="demo")).max_context_window.default_value
        == 256000
    )


@pytest.mark.parametrize("value", [-1, 0, 1, 128000, 2**80])
def test_context_window_accepts_any_integer(value):
    assert (
        ModelProfile(
            model=Attribute(default_value="demo"), max_context_window=Attribute(default_value=value)
        ).max_context_window.default_value
        == value
    )


@pytest.mark.parametrize("value", [None, "256000", 1.5, True, False])
def test_context_window_rejects_non_integer_values(value):
    with pytest.raises(ConfigurationError, match="max_context_window"):
        ModelProfile(
            model=Attribute(default_value="demo"), max_context_window=Attribute(default_value=value)
        )


def test_provider_models_default_and_independent_lists():
    first, second = ProviderProfile(), ProviderProfile()
    assert first.models is None
    assert first.list_models() == []
    model = ModelProfile(model=Attribute(default_value="demo"))
    first.add_models(model)
    assert first.list_models() == [model]
    assert second.models is None


def test_catalogue_constructor_and_listing_copy_only_the_container():
    model = ModelProfile(model=Attribute(default_value="demo"))
    initial = [model]
    provider = ProviderProfile(models=Attribute(default_value=initial))
    initial.clear()
    listed = provider.list_models()
    assert listed == [model]
    assert listed[0] is model
    listed.append(ModelProfile(model=Attribute(default_value="other")))
    assert provider.list_models() == [model]


def test_add_single_batch_and_replace_matching_id_in_stable_order():
    first, second, third = [
        ModelProfile(model=Attribute(default_value=name)) for name in ("first", "second", "third")
    ]
    provider = ProviderProfile(models=Attribute(default_value=[first]))
    provider.add_models(second)
    provider.add_models([third])
    replacement = ModelProfile(
        model=Attribute(default_value="second"), max_context_window=Attribute(default_value=8192)
    )
    provider.add_models([replacement, ModelProfile(model=Attribute(default_value="fourth"))])
    assert [model.model.default_value for model in provider.list_models()] == [
        "first",
        "second",
        "third",
        "fourth",
    ]
    assert provider.list_models()[1] is replacement
    assert replacement.max_context_window.default_value == 8192


def test_duplicate_ids_in_constructor_and_batch_use_last_configuration():
    first = ModelProfile(
        model=Attribute(default_value="demo"), max_context_window=Attribute(default_value=1)
    )
    updated = ModelProfile(
        model=Attribute(default_value="demo"), max_context_window=Attribute(default_value=2)
    )
    provider = ProviderProfile(models=Attribute(default_value=[first, updated]))
    assert provider.list_models() == [updated]
    provider.add_models([updated, first])
    assert provider.list_models() == [first]


def test_delete_multiple_ids_ignores_unknown_and_retains_remaining_order():
    models = [
        ModelProfile(model=Attribute(default_value=name)) for name in ("first", "second", "third")
    ]
    provider = ProviderProfile(models=Attribute(default_value=models))
    provider.delete_model(["first", "missing", "third", "first"])
    assert provider.list_models() == [models[1]]
    provider.delete_model(["second"])
    assert provider.models.default_value == []
    provider.delete_model(["missing"])
    assert provider.list_models() == []


def test_empty_catalogue_operations():
    provider = ProviderProfile()
    provider.add_models([])
    provider.delete_model([])
    provider.delete_model(["missing"])
    assert provider.models is None
    assert ProviderProfile(models=Attribute(default_value=[])).models.default_value == []


@pytest.mark.parametrize(
    "models",
    [
        "demo",
        (ModelProfile(model=Attribute(default_value="demo")),),
        ["demo"],
        [ModelProfile(model=Attribute(default_value="ok")), None],
    ],
)
def test_invalid_constructor_catalogues(models):
    with pytest.raises(ConfigurationError, match="models"):
        ProviderProfile(models=Attribute(default_value=models))


@pytest.mark.parametrize(
    "incoming",
    [
        None,
        "demo",
        ["demo"],
        [ModelProfile(model=Attribute(default_value="new")), None],
        [ModelProfile(model=Attribute(default_value=" "))],
    ],
)
def test_invalid_additions_are_atomic(incoming):
    original = ModelProfile(model=Attribute(default_value="original"))
    provider = ProviderProfile(models=Attribute(default_value=[original]))
    with pytest.raises(ConfigurationError):
        provider.add_models(incoming)
    assert provider.list_models() == [original]


@pytest.mark.parametrize("model_id", [None, "demo", [1], ("demo",)])
def test_invalid_deletions_do_not_modify_catalogue(model_id):
    original = ModelProfile(model=Attribute(default_value="demo"))
    provider = ProviderProfile(models=Attribute(default_value=[original]))
    with pytest.raises(ConfigurationError, match="model_id"):
        provider.delete_model(model_id)
    assert provider.list_models() == [original]


def test_catalogue_and_context_metadata_are_excluded_from_resolution_and_deepcopy():
    class CatalogueModel(ModelProfile):
        def __deepcopy__(self, memo):
            raise AssertionError("request resolution must not copy the model catalogue")

    provider = ProviderProfile(
        protocol=Config(default_value="openai_chat"),
        models=Attribute(
            default_value=[CatalogueModel(model=Attribute(default_value="catalogue-only"))]
        ),
    )
    resolved = resolve_profiles(
        provider,
        ModelProfile(
            model=Attribute(default_value="selected"),
            max_context_window=Attribute(default_value=-1),
        ),
    )
    assert "models" not in resolved.values
    assert "max_context_window" not in resolved.values
    assert resolved.model == "selected"


@pytest.mark.parametrize("protocol", ["openai_responses", "openai_chat"])
@pytest.mark.parametrize("window", [-1, 0, 1, 256000])
async def test_metadata_does_not_change_request_or_limit_content(protocol, window, monkeypatch):
    provider = ProviderProfile(
        protocol=Config(default_value=protocol),
        models=Attribute(
            default_value=[
                ModelProfile(
                    model=Attribute(default_value="demo"),
                    max_tokens=parameter("max_tokens", 1, protocol=protocol),
                )
            ]
        ),
    )
    model = ModelProfile(
        model=Attribute(default_value="demo"),
        max_tokens=parameter("max_tokens", 64, protocol=protocol),
        max_context_window=Attribute(default_value=window),
    )
    events = (
        [response_completed()] if protocol == "openai_responses" else [chat_event(finish="stop")]
    )
    client = FakeClient(events)
    monkeypatch.setattr("my_llm.codec.openai.chat.transport.AsyncOpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        "my_llm.codec.openai.responses.transport.AsyncOpenAI",
        lambda **kwargs: client,
    )
    text = "content longer than the configured context window " * 10
    await generate(
        stdio_protocol="openai_chat",
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    request = client.requests[0]
    assert "models" not in request and "max_context_window" not in request
    encoded = request["input" if protocol == "openai_responses" else "messages"]
    assert text in str(encoded)
    assert (
        request["max_output_tokens" if protocol == "openai_responses" else "max_completion_tokens"]
        == 64
    )
