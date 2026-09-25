"""Bedrock client for the answer generation."""

from __future__ import annotations

import json
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

    def generate(
        self,
        messages: Sequence[Message],
        system: str,
        inference_config: dict[str, Any] | None = None,
    ) -> str:
        """Sends a conversation to the model and returns its answer.

        Args:
            messages: The turns of the conversation in chronological order,
                ending with the current user message.
            system: The system prompt, sent as a separate Converse block so
                that the model can tell it apart from the user input. It is
                mandatory: without it the persona and the injection guardrails
                would silently be missing.
            inference_config: Optional ``inferenceConfig`` block (for example
                ``{"temperature": 0}``) applied to this single request. Left
                unset, the model answers with its own default, which is what
                the answer generation relies on for a natural style.

        Returns:
            The generated text.

        Raises:
            ValueError: If the system prompt is empty, if no message is given
                or if the last one is not a user turn.
        """
        text, _ = self._converse(messages, system, inference_config=inference_config)
        return text

    def generate_json(
        self,
        messages: Sequence[Message],
        system: str,
        json_schema: dict[str, Any],
        schema_name: str,
        inference_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Sends a conversation whose answer has to be structured JSON.

        Uses the structured output of the Converse API: the schema is sent as
        ``outputConfig.textFormat``, so the model is constrained while it
        generates and cannot answer in free-form prose. The caller is still
        responsible for validating the values, since a schema can only
        enumerate what is known in advance.

        Args:
            messages: The turns of the conversation in chronological order,
                ending with the current user message.
            system: The system prompt, sent as a separate Converse block.
            json_schema: The JSON schema the answer has to conform to.
            schema_name: Name of the schema, sent along for logging.
            inference_config: Optional ``inferenceConfig`` block for this
                single request; extraction calls pass ``{"temperature": 0}``
                so that the same question yields the same JSON.

        Returns:
            The parsed JSON object the model generated.

        Raises:
            ValueError: If the system prompt is empty, if no message is given,
                if the last one is not a user turn, or if the answer is not a
                JSON object despite the schema.
        """
        text, _ = self._converse(
            messages,
            system,
            json_schema=json_schema,
            schema_name=schema_name,
            inference_config=inference_config,
        )
        return _parse_json_object(text)

    def _converse(
        self,
        messages: Sequence[Message],
        system: str,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "structured_output",
        inference_config: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Runs a Converse request and returns text plus raw response.

        Args:
            messages: The turns of the conversation, ending with a user turn.
            system: The system prompt.
            json_schema: Optional JSON schema the answer has to conform to.
            schema_name: Name of the schema, only used when one is given.
            inference_config: Optional ``inferenceConfig`` block, applied to
                this request only, so that other callers of the same client
                keep the model defaults.

        Returns:
            The generated text and the untouched Converse response.

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
        if json_schema is not None:
            request["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "name": schema_name,
                            "schema": json.dumps(json_schema),
                        }
                    },
                }
            }
        if inference_config is not None:
            # Copy so a caller-owned dict (e.g. a module-level constant)
            # cannot be mutated through the request later on.
            request["inferenceConfig"] = dict(inference_config)

        LOGGER.debug(
            "invoking %s with %d message(s), json_schema=%s",
            self._model_id,
            len(messages),
            json_schema is not None,
        )
        response = self._client.converse(**request)
        _log_response(response)
        return _extract_text(response), response


def _log_response(response: dict[str, Any]) -> None:
    """Logs the Converse response for diagnosis.

    Args:
        response: The response returned by the Converse API.
    """
    try:
        message = response.get("output", {}).get("message", {})
        content = message.get("content") or []
        block_types = sorted(
            {
                key
                for block in content
                if isinstance(block, dict)
                for key in block
                if key != "text"
            }
        )
        text = _extract_text(response)
        stop_reason = message.get("stopReason", "")
        usage = response.get("usage", {})
        LOGGER.debug(
            "bedrock response:"
            "stop_reason=%s block_types=%s tokens=%s/%s text_len=%d text=%r",
            stop_reason or "?",
            ",".join(block_types) or "text",
            usage.get("inputTokens", "?"),
            usage.get("outputTokens", "?"),
            len(text),
            " ".join(text.split()),
        )
    except Exception:
        # Diagnosis must never be the reason a request fails.
        LOGGER.debug("could not log bedrock response shape", exc_info=True)


def _extract_text(response: dict[str, Any]) -> str:
    """Reads the generated text out of a Converse response.

    Args:
        response: The response returned by the Converse API.

    Returns:
        All text blocks of the answer, joined by newlines.
    """
    content = response["output"]["message"]["content"]
    return "\n".join(block["text"] for block in content if "text" in block)


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parses the answer of a structured-output request into a mapping.

    Args:
        text: The generated text, expected to be one JSON object.

    Returns:
        The parsed JSON object.

    Raises:
        ValueError: If the text is not valid JSON or not an object.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"model answer is no valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("model answer is no JSON object")
    return parsed
