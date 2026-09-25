"""Retrieval configuration read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from cvbot_core.env import read_bool, read_csv, read_float, read_int, read_str
from cvbot_core.logging_config import DEFAULT_LOG_LEVEL, VALID_LOG_LEVELS
from cvbot_core.overrides import apply_overrides
from cvbot_core.validation import (
    require_at_least,
    require_below,
    require_choice,
    require_float_in_range,
    require_http_origins,
    require_non_empty,
    require_port,
    require_positive,
)
from cvbot_core.vector_store import (
    DEFAULT_CHROMA_HOST,
    DEFAULT_CHROMA_PORT,
    DEFAULT_COLLECTION_NAME,
)

DEFAULT_AWS_REGION = "eu-central-1"
DEFAULT_LLM_MODEL_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_TOP_K = 4
DEFAULT_OVERFETCH_FACTOR = 4
MAX_OVERFETCH_FACTOR = 20
DEFAULT_FILTER_WEIGHT = 0.2
DEFAULT_RECENCY_WEIGHT = 0.2
DEFAULT_RECENCY_WINDOW_YEARS = 10
DEFAULT_MAX_CONTEXT_TOKENS = 32000
DEFAULT_RESPONSE_TOKEN_BUFFER = 2048
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8080
DEFAULT_RATE_LIMIT_PER_MINUTE = 10
DEFAULT_RATE_LIMIT_PER_HOUR = 60
DEFAULT_TRUST_FORWARDED_FOR = True
DEFAULT_CORS_ALLOWED_ORIGINS: tuple[str, ...] = ()
DEFAULT_CONVERSATION_TTL_SECONDS = 1800
DEFAULT_MAX_CONVERSATIONS = 50


@dataclass(frozen=True)
class RankingConfig:
    """Weights and window of the re-ranking stage.

    The single home of the ranking defaults: ``retrieve`` and ``rerank`` take
    this object instead of loose tuning parameters, so the values cannot
    drift between a module default and the configuration.

    Attributes:
        overfetch_factor: How many times ``top_k`` is fetched before the
            similarity, boost and recency scores re-rank the candidates.
            Higher values let a fresher or matching chunk win from further
            down the similarity ranking.
        filter_weight: Largest score the metadata boost can add, relative to
            the similarity score, which lies between 0 and 1. Scaled by the
            share of the extracted fields a chunk matches, so the bonus stays
            bounded no matter how many fields were extracted.
        recency_weight: Largest score the recency bonus can add, derived from
            the ``startdate``/``enddate``/``status`` metadata at query time.
            ``0`` turns the bonus off and leaves the pure similarity order.
        recency_window_years: How many years back the recency bonus decays
            linearly to zero.
    """

    overfetch_factor: int = DEFAULT_OVERFETCH_FACTOR
    filter_weight: float = DEFAULT_FILTER_WEIGHT
    recency_weight: float = DEFAULT_RECENCY_WEIGHT
    recency_window_years: int = DEFAULT_RECENCY_WINDOW_YEARS

    def __post_init__(self) -> None:
        """Validates the ranking parameters.

        Raises:
            ValueError: If a value is outside the accepted range.
        """
        require_positive(self.overfetch_factor, "overfetch_factor")
        require_below(
            self.overfetch_factor,
            MAX_OVERFETCH_FACTOR,
            "overfetch_factor",
            "the supported maximum",
        )
        require_float_in_range(self.filter_weight, 0.0, 1.0, "filter_weight")
        require_float_in_range(self.recency_weight, 0.0, 1.0, "recency_weight")
        require_positive(self.recency_window_years, "recency_window_years")


@dataclass(frozen=True)
class Settings:
    """Runtime configuration of the retrieval pipeline.

    ``collection_name`` must match the value used by cvbot-embedder, otherwise
    the collection cannot be found. The embedding model itself is not
    configured here: it is read from the collection metadata cvbot-embedder
    wrote, so query and index vectors are always built with the same model.

    Attributes:
        chroma_host: Hostname of the ChromaDB container (AWS Fargate).
        chroma_port: Port of the ChromaDB container.
        collection_name: Name of the collection that is queried.
        aws_region: AWS region of the Bedrock client.
        llm_model_id: Bedrock model ID used to generate the answer.
        top_k: Number of chunks retrieved per question.
        overfetch_factor: How many times ``top_k`` is fetched before the
            similarity, boost and recency scores re-rank the candidates.
            Higher values let a fresher or matching chunk win from further
            down the similarity ranking.
        filter_weight: Largest score the metadata boost can add, relative to
            the similarity score, which lies between 0 and 1. Scaled by the
            share of the extracted fields a chunk matches.
        recency_weight: Largest score the recency bonus can add, derived from
            the ``startdate``/``enddate``/``status`` metadata at query time.
            ``0`` turns the bonus off and leaves the pure similarity order.
        recency_window_years: How many years back the recency bonus decays
            linearly to zero.
        max_context_tokens: Upper bound for the whole context sent to the LLM
            (system prompt, retrieved chunks and conversation history).
        response_token_buffer: Part of ``max_context_tokens`` that is kept free
            for the answer of the model.
        web_host: Interface the web application binds to.
        web_port: Port the web application listens on.
        log_level: Verbosity of the log output.
        rate_limit_per_minute: Answered questions a single client may request
            within one minute before it is rejected.
        rate_limit_per_hour: Answered questions a single client may request
            within one hour.
        trust_forwarded_for: Whether the client address may be taken from the
            ``X-Forwarded-For`` header. Only safe behind a trusted proxy such
            as the API Gateway; when disabled, the peer address is used.
        cors_allowed_origins: Origins allowed to call the JSON API from a
            browser. Empty means same-origin only.
        conversation_ttl_seconds: Idle time after which a conversation is
            dropped from the store.
        max_conversations: Upper bound of conversations kept in memory; the
            least recently used one is dropped beyond it.
    """

    chroma_host: str = DEFAULT_CHROMA_HOST
    chroma_port: int = DEFAULT_CHROMA_PORT
    collection_name: str = DEFAULT_COLLECTION_NAME
    aws_region: str = DEFAULT_AWS_REGION
    llm_model_id: str = DEFAULT_LLM_MODEL_ID
    top_k: int = DEFAULT_TOP_K
    overfetch_factor: int = DEFAULT_OVERFETCH_FACTOR
    filter_weight: float = DEFAULT_FILTER_WEIGHT
    recency_weight: float = DEFAULT_RECENCY_WEIGHT
    recency_window_years: int = DEFAULT_RECENCY_WINDOW_YEARS
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    response_token_buffer: int = DEFAULT_RESPONSE_TOKEN_BUFFER
    web_host: str = DEFAULT_WEB_HOST
    web_port: int = DEFAULT_WEB_PORT
    log_level: str = DEFAULT_LOG_LEVEL
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE
    rate_limit_per_hour: int = DEFAULT_RATE_LIMIT_PER_HOUR
    trust_forwarded_for: bool = DEFAULT_TRUST_FORWARDED_FOR
    # Tuple, not list: a frozen dataclass generates __hash__ over its fields.
    cors_allowed_origins: tuple[str, ...] = DEFAULT_CORS_ALLOWED_ORIGINS
    conversation_ttl_seconds: int = DEFAULT_CONVERSATION_TTL_SECONDS
    max_conversations: int = DEFAULT_MAX_CONVERSATIONS

    def __post_init__(self) -> None:
        """Validates the configuration.

        Raises:
            ValueError: If a value is outside the accepted range.
        """
        require_non_empty(self.chroma_host, "chroma_host")
        require_non_empty(self.collection_name, "collection_name")
        require_port(self.chroma_port, "chroma_port")
        require_non_empty(self.aws_region, "aws_region")
        require_non_empty(self.llm_model_id, "llm_model_id")
        require_positive(self.top_k, "top_k")
        # The ranking parameters are validated by RankingConfig alone, so the
        # rules cannot drift apart between the two configurations.
        self.ranking_config()
        require_positive(self.max_context_tokens, "max_context_tokens")
        require_positive(self.response_token_buffer, "response_token_buffer")
        require_below(
            self.response_token_buffer,
            self.max_context_tokens,
            "response_token_buffer",
            "max_context_tokens",
        )
        require_non_empty(self.web_host, "web_host")
        require_port(self.web_port, "web_port")
        require_choice(self.log_level, VALID_LOG_LEVELS, "log_level")
        require_positive(self.rate_limit_per_minute, "rate_limit_per_minute")
        require_at_least(
            self.rate_limit_per_hour,
            self.rate_limit_per_minute,
            "rate_limit_per_hour",
            "rate_limit_per_minute",
        )
        require_positive(
            self.conversation_ttl_seconds, "conversation_ttl_seconds"
        )
        require_positive(self.max_conversations, "max_conversations")
        require_http_origins(self.cors_allowed_origins, "cors_allowed_origins")
        # Keeps the frozen dataclass hashable when callers pass a list.
        object.__setattr__(
            self, "cors_allowed_origins", tuple(self.cors_allowed_origins)
        )

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        """Builds the configuration from environment variables.

        Variables that are not set fall back to the module defaults.

        Args:
            env: Optional mapping used instead of ``os.environ`` (for tests).

        Returns:
            The validated configuration.

        Raises:
            ValueError: If a variable cannot be parsed or is outside the
                accepted range.
        """
        source = os.environ if env is None else env
        return cls(
            chroma_host=read_str(source, "CHROMA_HOST", DEFAULT_CHROMA_HOST),
            chroma_port=read_int(source, "CHROMA_PORT", DEFAULT_CHROMA_PORT),
            collection_name=read_str(
                source, "CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME
            ),
            aws_region=read_str(source, "AWS_REGION", DEFAULT_AWS_REGION),
            llm_model_id=read_str(source, "LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID),
            top_k=read_int(source, "TOP_K", DEFAULT_TOP_K),
            overfetch_factor=read_int(
                source,
                "OVERFETCH_FACTOR",
                DEFAULT_OVERFETCH_FACTOR,
            ),
            filter_weight=read_float(
                source, "FILTER_WEIGHT", DEFAULT_FILTER_WEIGHT
            ),
            recency_weight=read_float(
                source, "RECENCY_WEIGHT", DEFAULT_RECENCY_WEIGHT
            ),
            recency_window_years=read_int(
                source,
                "RECENCY_WINDOW_YEARS",
                DEFAULT_RECENCY_WINDOW_YEARS,
            ),
            max_context_tokens=read_int(
                source, "MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS
            ),
            response_token_buffer=read_int(
                source, "RESPONSE_TOKEN_BUFFER", DEFAULT_RESPONSE_TOKEN_BUFFER
            ),
            web_host=read_str(source, "WEB_HOST", DEFAULT_WEB_HOST),
            web_port=read_int(source, "WEB_PORT", DEFAULT_WEB_PORT),
            log_level=read_str(source, "LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
            rate_limit_per_minute=read_int(
                source, "RATE_LIMIT_PER_MINUTE", DEFAULT_RATE_LIMIT_PER_MINUTE
            ),
            rate_limit_per_hour=read_int(
                source, "RATE_LIMIT_PER_HOUR", DEFAULT_RATE_LIMIT_PER_HOUR
            ),
            trust_forwarded_for=read_bool(
                source, "TRUST_FORWARDED_FOR", DEFAULT_TRUST_FORWARDED_FOR
            ),
            cors_allowed_origins=read_csv(
                source, "CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ALLOWED_ORIGINS
            ),
            conversation_ttl_seconds=read_int(
                source,
                "CONVERSATION_TTL_SECONDS",
                DEFAULT_CONVERSATION_TTL_SECONDS,
            ),
            max_conversations=read_int(
                source, "MAX_CONVERSATIONS", DEFAULT_MAX_CONVERSATIONS
            ),
        )

    def ranking_config(self) -> RankingConfig:
        """Extracts the ranking parameters of this configuration.

        Returns:
            The validated ranking configuration for ``retrieve``.
        """
        return RankingConfig(
            overfetch_factor=self.overfetch_factor,
            filter_weight=self.filter_weight,
            recency_weight=self.recency_weight,
            recency_window_years=self.recency_window_years,
        )

    def with_overrides(self, **overrides: Any) -> "Settings":
        """Returns a copy with the given fields replaced.

        ``None`` values are ignored so that unset CLI arguments do not override
        the configuration coming from the environment.

        Args:
            **overrides: Field names and their new values.

        Returns:
            A new, validated ``Settings`` instance.
        """
        return apply_overrides(self, **overrides)
