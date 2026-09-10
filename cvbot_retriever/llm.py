"""Bedrock client for the answer generation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import boto3

from .config import Settings
from .conversation import ROLE_USER, Message

LOGGER = logging.getLogger(__name__)


class BedrockLLMClient:
    """Generates text with a Bedrock model through the Converse API.

    The client knows nothing about prompt construction: it receives a finished
    message list and passes it on unchanged. System prompt, retrieved context
    and truncation of the history are handled by the calling pipeline.
    """

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        """Initializes the client.

        AWS credentials are resolved through the usual boto3 chain (environment
        variables, profile, IAM role of the Fargate task).

        Args:
            settings: Runtime configuration holding model ID and region.
            client: Optional ``bedrock-runtime`` client used instead of a newly
                created one (for tests).
        """
        self._model_id = settings.llm_model_id
        self._client = client or boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )
        LOGGER.info(
            "Bedrock LLM: model_id=%s region=%s",
            settings.llm_model_id,
            settings.aws_region,
        )

    def generate(self, messages: Sequence[Message], system: str) -> str:
        """Sends a conversation to the model and returns its answer.

        Args:
            messages: The turns of the conversation in chronological order,
                ending with the current user message.
            system: The system prompt, sent as a separate Converse block so
                that the model can tell it apart from the user input. It is
                mandatory: without it the persona and the injection guardrails
                would silently be missing.

        Returns:
            The generated text.

        Raises:
            ValueError: If the system prompt is empty, if no message is given
                or if the last one is not a user turn.
        """
        if not system.strip():
            raise ValueError("system prompt must not be empty")
        if not messages:
            raise ValueError("messages must not be empty")
        if messages[-1].role != ROLE_USER:
            raise ValueError("the last message must be a user message")

        request: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": [message.to_converse() for message in messages],
            "system": [{"text": system}],
        }

        LOGGER.debug(
            "invoking %s with %d message(s)", self._model_id, len(messages)
        )
        response = self._client.converse(**request)
        return _extract_text(response)


def _extract_text(response: dict[str, Any]) -> str:
    """Reads the generated text out of a Converse response.

    Args:
        response: The response returned by the Converse API.

    Returns:
        All text blocks of the answer, joined by newlines.
    """
    content = response["output"]["message"]["content"]
    return "\n".join(block["text"] for block in content if "text" in block)
