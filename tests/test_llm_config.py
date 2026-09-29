import pytest

from utils import llm_config


def test_openai_llm_uses_requested_model():
    llm = llm_config.get_openai_llm(model="gpt-4o-mini", temperature=0.3)
    assert llm.model_name == "gpt-4o-mini" and llm.temperature == 0.3


def test_embeddings_reject_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        llm_config.get_embeddings("nope")


def test_openai_embeddings_construct():
    assert llm_config.get_embeddings("openai") is not None
