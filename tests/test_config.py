"""Tests for the configuration."""

from __future__ import annotations

import pytest

from cvbot_retriever.config import (
    DEFAULT_CHROMA_PORT,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_LOG_LEVEL,
    DEFAULT_TOP_K,
    Settings,
)


def test_from_env_uses_defaults_when_unset() -> None:
    settings = Settings.from_env(env={})

    assert settings.chroma_port == DEFAULT_CHROMA_PORT
    assert settings.collection_name == DEFAULT_COLLECTION_NAME
    assert settings.embedding_model_id == DEFAULT_EMBEDDING_MODEL_ID
    assert settings.llm_model_id == DEFAULT_LLM_MODEL_ID
    assert settings.top_k == DEFAULT_TOP_K
    assert settings.log_level == DEFAULT_LOG_LEVEL


def test_defaults_match_the_embedder_collection() -> None:
    settings = Settings.from_env(env={})

    assert settings.collection_name == "cvbot_documents"
    assert settings.embedding_model_id == "amazon.titan-embed-text-v2:0"


def test_from_env_reads_all_values() -> None:
    settings = Settings.from_env(
        env={
            "CHROMA_HOST": "chroma.internal",
            "CHROMA_PORT": "8443",
            "CHROMA_COLLECTION": "jobs",
            "AWS_REGION": "eu-west-1",
            "EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v1",
            "LLM_MODEL_ID": "amazon.nova-pro-v1:0",
            "TOP_K": "8",
            "LOG_LEVEL": "debug",
        }
    )

    assert settings.chroma_host == "chroma.internal"
    assert settings.chroma_port == 8443
    assert settings.collection_name == "jobs"
    assert settings.aws_region == "eu-west-1"
    assert settings.embedding_model_id == "amazon.titan-embed-text-v1"
    assert settings.llm_model_id == "amazon.nova-pro-v1:0"
    assert settings.top_k == 8
    assert settings.log_level == "DEBUG"


def test_from_env_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHROMA_HOST", "from-process-env")

    assert Settings.from_env().chroma_host == "from-process-env"


def test_non_numeric_top_k_raises() -> None:
    with pytest.raises(ValueError, match="TOP_K"):
        Settings.from_env(env={"TOP_K": "four"})


def test_non_numeric_port_raises() -> None:
    with pytest.raises(ValueError, match="CHROMA_PORT"):
        Settings.from_env(env={"CHROMA_PORT": "eight"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"chroma_host": ""},
        {"collection_name": ""},
        {"chroma_port": 0},
        {"chroma_port": 70000},
        {"aws_region": ""},
        {"embedding_model_id": ""},
        {"llm_model_id": ""},
        {"top_k": 0},
        {"log_level": "TRACE"},
    ],
)
def test_invalid_values_raise(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Settings(**overrides)  # type: ignore[arg-type]


def test_with_overrides_ignores_none() -> None:
    settings = Settings(chroma_host="original", top_k=3)

    updated = settings.with_overrides(chroma_host=None, top_k=7)

    assert updated.chroma_host == "original"
    assert updated.top_k == 7
    assert settings.top_k == 3
