"""Bedrock client for the answer generation."""

from __future__ import annotations

import logging
from typing import Any

import boto3

from .config import Settings

LOGGER = logging.getLogger(__name__)


class BedrockLLMClient:
    """Generates text with a Bedrock model through the Converse API.

    The client knows nothing about prompt construction: it receives a finished
    prompt and passes it on unchanged. Later iterations can therefore add a
    system prompt, retrieved context and conversation history without touching
    this class.
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

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Sends a prompt to the model and returns its answer.

        Args:
            prompt: The finished user prompt.
            system: Optional system prompt sent as a separate Converse block so
                that the model can tell it apart from the user input.

        Returns:
            The generated text.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        request: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
        }
        if system:
            request["system"] = [{"text": system}]

        LOGGER.debug(
            "invoking %s with %d prompt character(s)",
            self._model_id,
            len(prompt),
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
