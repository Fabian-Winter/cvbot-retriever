"""Tests for the configuration."""

from __future__ import annotations

import pytest

from cvbot_retriever.config import (
    DEFAULT_CHROMA_PORT,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CONVERSATION_TTL_SECONDS,
    DEFAULT_CORS_ALLOWED_ORIGINS,
    DEFAULT_FILTER_WEIGHT,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_OVERFETCH_FACTOR,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_CONVERSATIONS,
    DEFAULT_RATE_LIMIT_PER_HOUR,
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    DEFAULT_RECENCY_WEIGHT,
    DEFAULT_RECENCY_WINDOW_YEARS,
    DEFAULT_RESPONSE_TOKEN_BUFFER,
    DEFAULT_TOP_K,
    DEFAULT_TRUST_FORWARDED_FOR,
    DEFAULT_WEB_HOST,
    DEFAULT_WEB_PORT,
    RankingConfig,
    Settings,
)


def test_from_env_uses_defaults_when_unset() -> None:
    settings = Settings.from_env(env={})

    assert settings.chroma_port == DEFAULT_CHROMA_PORT
    assert settings.collection_name == DEFAULT_COLLECTION_NAME
    assert settings.llm_model_id == DEFAULT_LLM_MODEL_ID
    assert settings.top_k == DEFAULT_TOP_K
    assert settings.overfetch_factor == DEFAULT_OVERFETCH_FACTOR
    assert settings.filter_weight == DEFAULT_FILTER_WEIGHT
    assert settings.recency_weight == DEFAULT_RECENCY_WEIGHT
    assert settings.recency_window_years == DEFAULT_RECENCY_WINDOW_YEARS
    assert settings.max_context_tokens == DEFAULT_MAX_CONTEXT_TOKENS
    assert settings.response_token_buffer == DEFAULT_RESPONSE_TOKEN_BUFFER
    assert settings.web_host == DEFAULT_WEB_HOST
    assert settings.web_port == DEFAULT_WEB_PORT
    assert settings.log_level == DEFAULT_LOG_LEVEL
    assert settings.rate_limit_per_minute == DEFAULT_RATE_LIMIT_PER_MINUTE
    assert settings.rate_limit_per_hour == DEFAULT_RATE_LIMIT_PER_HOUR
    assert settings.trust_forwarded_for == DEFAULT_TRUST_FORWARDED_FOR
    assert settings.cors_allowed_origins == DEFAULT_CORS_ALLOWED_ORIGINS
    assert settings.conversation_ttl_seconds == DEFAULT_CONVERSATION_TTL_SECONDS
    assert settings.max_conversations == DEFAULT_MAX_CONVERSATIONS


def test_the_default_cors_policy_allows_no_origin() -> None:
    assert Settings.from_env(env={}).cors_allowed_origins == ()


def test_defaults_match_the_embedder_collection() -> None:
    settings = Settings.from_env(env={})

    assert settings.collection_name == "cvbot_documents"


def test_from_env_reads_all_values() -> None:
    settings = Settings.from_env(
        env={
            "CHROMA_HOST": "chroma.internal",
            "CHROMA_PORT": "8443",
            "CHROMA_COLLECTION": "jobs",
            "AWS_REGION": "eu-west-1",
            "LLM_MODEL_ID": "amazon.nova-pro-v1:0",
            "TOP_K": "8",
            "OVERFETCH_FACTOR": "6",
            "FILTER_WEIGHT": "0.4",
            "RECENCY_WEIGHT": "0.3",
            "RECENCY_WINDOW_YEARS": "15",
            "MAX_CONTEXT_TOKENS": "4000",
            "RESPONSE_TOKEN_BUFFER": "500",
            "WEB_HOST": "0.0.0.0",
            "WEB_PORT": "9000",
            "LOG_LEVEL": "debug",
            "RATE_LIMIT_PER_MINUTE": "3",
            "RATE_LIMIT_PER_HOUR": "30",
            "TRUST_FORWARDED_FOR": "false",
            "CORS_ALLOWED_ORIGINS": "https://a.example.com, https://b.example.com",
            "CONVERSATION_TTL_SECONDS": "900",
            "MAX_CONVERSATIONS": "50",
        }
    )

    assert settings.chroma_host == "chroma.internal"
    assert settings.chroma_port == 8443
    assert settings.collection_name == "jobs"
    assert settings.aws_region == "eu-west-1"
    assert settings.llm_model_id == "amazon.nova-pro-v1:0"
    assert settings.top_k == 8
    assert settings.overfetch_factor == 6
    assert settings.filter_weight == 0.4
    assert settings.recency_weight == 0.3
    assert settings.recency_window_years == 15
    assert settings.max_context_tokens == 4000
    assert settings.response_token_buffer == 500
    assert settings.web_host == "0.0.0.0"
    assert settings.web_port == 9000
    assert settings.log_level == "DEBUG"
    assert settings.rate_limit_per_minute == 3
    assert settings.rate_limit_per_hour == 30
    assert settings.trust_forwarded_for is False
    assert settings.cors_allowed_origins == (
        "https://a.example.com",
        "https://b.example.com",
    )
    assert settings.conversation_ttl_seconds == 900
    assert settings.max_conversations == 50


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_boolean_spellings_are_understood(raw: str, expected: bool) -> None:
    settings = Settings.from_env(env={"TRUST_FORWARDED_FOR": raw})

    assert settings.trust_forwarded_for is expected


def test_a_non_boolean_flag_raises() -> None:
    with pytest.raises(ValueError, match="TRUST_FORWARDED_FOR"):
        Settings.from_env(env={"TRUST_FORWARDED_FOR": "maybe"})


def test_empty_cors_entries_are_dropped() -> None:
    settings = Settings.from_env(
        env={"CORS_ALLOWED_ORIGINS": " https://a.example.com , , "}
    )

    assert settings.cors_allowed_origins == ("https://a.example.com",)


def test_a_cors_wildcard_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\*"):
        Settings.from_env(env={"CORS_ALLOWED_ORIGINS": "*"})


@pytest.mark.parametrize(
    "origin",
    ["cv.example.com", "ftp://cv.example.com", "https://cv.example.com/app"],
)
def test_a_malformed_cors_origin_raises(origin: str) -> None:
    with pytest.raises(ValueError, match="cors origin"):
        Settings.from_env(env={"CORS_ALLOWED_ORIGINS": origin})


def test_the_settings_stay_hashable_with_origins() -> None:
    settings = Settings(cors_allowed_origins=["https://cv.example.com"])

    assert hash(settings)
    assert settings.cors_allowed_origins == ("https://cv.example.com",)


def test_from_env_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHROMA_HOST", "from-process-env")

    assert Settings.from_env().chroma_host == "from-process-env"


def test_non_numeric_top_k_raises() -> None:
    with pytest.raises(ValueError, match="TOP_K"):
        Settings.from_env(env={"TOP_K": "four"})


def test_non_numeric_port_raises() -> None:
    with pytest.raises(ValueError, match="CHROMA_PORT"):
        Settings.from_env(env={"CHROMA_PORT": "eight"})


def test_non_numeric_max_context_tokens_raises() -> None:
    with pytest.raises(ValueError, match="MAX_CONTEXT_TOKENS"):
        Settings.from_env(env={"MAX_CONTEXT_TOKENS": "many"})


def test_non_numeric_web_port_raises() -> None:
    with pytest.raises(ValueError, match="WEB_PORT"):
        Settings.from_env(env={"WEB_PORT": "eighty"})


def test_non_numeric_recency_weight_raises() -> None:
    with pytest.raises(ValueError, match="RECENCY_WEIGHT"):
        Settings.from_env(env={"RECENCY_WEIGHT": "hoch"})


def test_response_buffer_must_leave_room_for_the_context() -> None:
    with pytest.raises(ValueError, match="response_token_buffer"):
        Settings.from_env(
            env={"MAX_CONTEXT_TOKENS": "1000", "RESPONSE_TOKEN_BUFFER": "1000"}
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"chroma_host": ""},
        {"collection_name": ""},
        {"chroma_port": 0},
        {"chroma_port": 70000},
        {"aws_region": ""},
        {"llm_model_id": ""},
        {"top_k": 0},
        {"overfetch_factor": 0},
        {"overfetch_factor": 100},
        {"filter_weight": -0.1},
        {"filter_weight": 1.1},
        {"recency_weight": -0.1},
        {"recency_weight": 1.1},
        {"recency_window_years": 0},
        {"max_context_tokens": 0},
        {"response_token_buffer": 0},
        {"max_context_tokens": 100, "response_token_buffer": 200},
        {"web_host": ""},
        {"web_port": 0},
        {"web_port": 70000},
        {"log_level": "TRACE"},
        {"rate_limit_per_minute": 0},
        {"rate_limit_per_minute": 10, "rate_limit_per_hour": 5},
        {"conversation_ttl_seconds": 0},
        {"max_conversations": 0},
        {"cors_allowed_origins": ("*",)},
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


def test_ranking_config_defaults() -> None:
    config = RankingConfig()

    assert config.overfetch_factor == DEFAULT_OVERFETCH_FACTOR
    assert config.filter_weight == DEFAULT_FILTER_WEIGHT
    assert config.recency_weight == DEFAULT_RECENCY_WEIGHT
    assert config.recency_window_years == DEFAULT_RECENCY_WINDOW_YEARS


def test_settings_ranking_config_mirrors_the_settings() -> None:
    settings = Settings.from_env(
        env={
            "OVERFETCH_FACTOR": "6",
            "FILTER_WEIGHT": "0.4",
            "RECENCY_WEIGHT": "0.3",
            "RECENCY_WINDOW_YEARS": "5",
        }
    )

    config = settings.ranking_config()

    assert config == RankingConfig(
        overfetch_factor=6,
        filter_weight=0.4,
        recency_weight=0.3,
        recency_window_years=5,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"overfetch_factor": 0},
        {"overfetch_factor": 100},
        {"filter_weight": -0.1},
        {"filter_weight": 1.1},
        {"recency_weight": -0.1},
        {"recency_weight": 1.1},
        {"recency_window_years": 0},
    ],
)
def test_ranking_config_rejects_invalid_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RankingConfig(**overrides)  # type: ignore[arg-type]
