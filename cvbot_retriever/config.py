"""Retrieval configuration read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

DEFAULT_CHROMA_HOST = "localhost"
DEFAULT_CHROMA_PORT = 8000
DEFAULT_COLLECTION_NAME = "cvbot_documents"
DEFAULT_AWS_REGION = "eu-central-1"
DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_LLM_MODEL_ID = "amazon.nova-lite-v1:0"
DEFAULT_TOP_K = 4
DEFAULT_MAX_CONTEXT_TOKENS = 8000
DEFAULT_RESPONSE_TOKEN_BUFFER = 1024
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8080
DEFAULT_LOG_LEVEL = "INFO"

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})


@dataclass(frozen=True)
class Settings:
    """Runtime configuration of the retrieval pipeline.

    ``collection_name`` and ``embedding_model_id`` must match the values used by
    cvbot-embedder, otherwise the query vectors are incompatible with the
    indexed ones.

    Attributes:
        chroma_host: Hostname of the ChromaDB container (AWS Fargate).
        chroma_port: Port of the ChromaDB container.
        collection_name: Name of the collection that is queried.
        aws_region: AWS region of the Bedrock client.
        embedding_model_id: Bedrock model ID used to embed the question.
        llm_model_id: Bedrock model ID used to generate the answer.
        top_k: Number of chunks retrieved per question.
        max_context_tokens: Upper bound for the whole context sent to the LLM
            (system prompt, retrieved chunks and conversation history).
        response_token_buffer: Part of ``max_context_tokens`` that is kept free
            for the answer of the model.
        web_host: Interface the web application binds to.
        web_port: Port the web application listens on.
        log_level: Verbosity of the log output.
    """

    chroma_host: str = DEFAULT_CHROMA_HOST
    chroma_port: int = DEFAULT_CHROMA_PORT
    collection_name: str = DEFAULT_COLLECTION_NAME
    aws_region: str = DEFAULT_AWS_REGION
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    llm_model_id: str = DEFAULT_LLM_MODEL_ID
    top_k: int = DEFAULT_TOP_K
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    response_token_buffer: int = DEFAULT_RESPONSE_TOKEN_BUFFER
    web_host: str = DEFAULT_WEB_HOST
    web_port: int = DEFAULT_WEB_PORT
    log_level: str = DEFAULT_LOG_LEVEL

    def __post_init__(self) -> None:
        """Validates the configuration.

        Raises:
            ValueError: If a value is outside the accepted range.
        """
        if not self.chroma_host:
            raise ValueError("chroma_host must not be empty")
        if not self.collection_name:
            raise ValueError("collection_name must not be empty")
        if not 1 <= self.chroma_port <= 65535:
            raise ValueError(f"chroma_port outside 1-65535: {self.chroma_port}")
        if not self.aws_region:
            raise ValueError("aws_region must not be empty")
        if not self.embedding_model_id:
            raise ValueError("embedding_model_id must not be empty")
        if not self.llm_model_id:
            raise ValueError("llm_model_id must not be empty")
        if self.top_k < 1:
            raise ValueError(f"top_k must be positive: {self.top_k}")
        if self.max_context_tokens < 1:
            raise ValueError(
                f"max_context_tokens must be positive: {self.max_context_tokens}"
            )
        if self.response_token_buffer < 1:
            raise ValueError(
                "response_token_buffer must be positive: "
                f"{self.response_token_buffer}"
            )
        if self.response_token_buffer >= self.max_context_tokens:
            raise ValueError(
                "response_token_buffer must be smaller than max_context_tokens: "
                f"{self.response_token_buffer} >= {self.max_context_tokens}"
            )
        if not self.web_host:
            raise ValueError("web_host must not be empty")
        if not 1 <= self.web_port <= 65535:
            raise ValueError(f"web_port outside 1-65535: {self.web_port}")
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(f"unknown log_level: {self.log_level}")

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
            chroma_host=source.get("CHROMA_HOST", DEFAULT_CHROMA_HOST),
            chroma_port=_int(source, "CHROMA_PORT", DEFAULT_CHROMA_PORT),
            collection_name=source.get(
                "CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME
            ),
            aws_region=source.get("AWS_REGION", DEFAULT_AWS_REGION),
            embedding_model_id=source.get(
                "EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID
            ),
            llm_model_id=source.get("LLM_MODEL_ID", DEFAULT_LLM_MODEL_ID),
            top_k=_int(source, "TOP_K", DEFAULT_TOP_K),
            max_context_tokens=_int(
                source, "MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS
            ),
            response_token_buffer=_int(
                source, "RESPONSE_TOKEN_BUFFER", DEFAULT_RESPONSE_TOKEN_BUFFER
            ),
            web_host=source.get("WEB_HOST", DEFAULT_WEB_HOST),
            web_port=_int(source, "WEB_PORT", DEFAULT_WEB_PORT),
            log_level=source.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
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
        effective = {
            key: value for key, value in overrides.items() if value is not None
        }
        return replace(self, **effective)


def _int(env: dict[str, str] | Any, key: str, default: int) -> int:
    """Reads an integer from the environment.

    Args:
        env: Mapping of variable names to values.
        key: Name of the variable.
        default: Value used if the variable is not set.

    Returns:
        The parsed value or ``default``.

    Raises:
        ValueError: If the value is not an integer.
    """
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} is not an integer: {raw!r}") from exc
