"""Instance profile resolution reuses its protocol's lazy codec."""

from collections import Counter

import pytest

from my_llm import Attribute, Config, ModelProfile, MyLLM, ProviderProfile
from my_llm.basic import protocols
from tests.helpers.protocols import PROTOCOLS


@pytest.mark.parametrize("upstream", PROTOCOLS)
def test_llm_constructs_only_one_selected_codec_when_resolving_defaults(monkeypatch, upstream):
    constructions = Counter()
    for protocol, factory in tuple(protocols._CODECS.items()):

        def counted_factory(protocol=protocol, factory=factory):
            constructions[protocol] += 1
            return factory()

        monkeypatch.setitem(protocols._CODECS, protocol, counted_factory)

    llm = MyLLM(
        provider=ProviderProfile(protocol=Config(default_value=upstream)),
        model=ModelProfile(model=Attribute(default_value="demo")),
        stdio_protocol="openai_chat",
    )

    assert constructions == {upstream: 1}
    assert set(llm.codec.codecs.instances) == {upstream}
    assert llm.codec.defaults(upstream) is llm.codec.codecs[upstream].defaults
